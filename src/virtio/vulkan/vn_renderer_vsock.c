/*
 * Copyright 2019 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * Copy-based venus transport ("vsock backend").
 *
 * Both existing backends share resource memory with the renderer:
 * virtgpu passes dma-buf fds through the kernel, vtest passes memfds through
 * SCM_RIGHTS.  Either way the guest and the renderer must live in the same
 * kernel, because an fd is a reference to a kernel object that only exists
 * there.
 *
 * Android's AVF (and anything else that only gives us a byte stream) cannot
 * do that.  This backend therefore gives every resource two independent
 * copies, one per side, and moves contents explicitly:
 *
 *  - the ring shmem is pushed after each submit (command bytes plus tail)
 *    and pulled before polling head/status;
 *  - host-visible device memory is pushed before the renderer can read it
 *    (tracked as "mapped since last submit") and pulled on invalidate;
 *  - reply shmems are pulled after the ring consumed the command.
 *
 * The renderer side protocol lives in vkbackend (/home/lakitu/code/vkbackend)
 * and is described in that repository's README.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#ifdef __linux__
#include <linux/vm_sockets.h>
#endif

#include "drm-uapi/virtgpu_drm.h"
#include "util/list.h"
#include "util/os_misc.h"
#include "util/sparse_array.h"
#include "util/u_process.h"
#include "virtio/virtio-gpu/venus_hw.h"

#include "vn_renderer_internal.h"

#define VSOCK_PCI_VENDOR_ID 0x1af4
#define VSOCK_PCI_DEVICE_ID 0x1050

#define VSOCK_PROTOCOL_VERSION 1

/* the only venus context we create */
#define VSOCK_CTX_ID 1

enum vsock_msg_type {
   VSOCK_MSG_CREATE_RENDERER = 1,
   VSOCK_MSG_PROTOCOL_VERSION = 2,
   VSOCK_MSG_GET_PARAM = 3,
   VSOCK_MSG_GET_CAPSET = 4,
   VSOCK_MSG_CONTEXT_INIT = 5,
   VSOCK_MSG_RES_CREATE = 6,
   VSOCK_MSG_RES_UNREF = 7,
   VSOCK_MSG_RES_MAP = 8,
   VSOCK_MSG_RES_WRITE = 9,
   VSOCK_MSG_RES_READ = 10,
   VSOCK_MSG_SUBMIT = 11,
   VSOCK_MSG_SYNC_CREATE = 12,
   VSOCK_MSG_SYNC_UNREF = 13,
   VSOCK_MSG_SYNC_READ = 14,
   VSOCK_MSG_SYNC_WRITE = 15,
   VSOCK_MSG_SYNC_WAIT = 16,
};

enum vsock_resp_type {
   VSOCK_RESP_OK = 0x80,
   VSOCK_RESP_ERR = 0x81,
   VSOCK_RESP_DATA = 0x82,
   VSOCK_RESP_TIMEOUT = 0x83,
};

enum vsock_param {
   VSOCK_PARAM_MAX_TIMELINE_COUNT = 1,
};

/* resource creation flags, interpreted by the renderer side */
enum vsock_blob_flags {
   VSOCK_BLOB_MAPPABLE = 1 << 0,
   VSOCK_BLOB_SHAREABLE = 1 << 1,
};

/* 16-byte header, little endian, followed by len bytes of payload */
struct vsock_msg_hdr {
   uint32_t type;
   uint32_t seq;
   uint32_t len;
   uint32_t arg;
};

struct vsock_res_create {
   uint64_t size;
   uint64_t blob_id;
};

/* a (sync_id, value) pair as sent on the wire */
struct vsock_wire_sync {
   uint32_t sync_id;
   uint32_t val_lo;
   uint32_t val_hi;
};

struct vsock;

struct vsock_shmem {
   struct vn_renderer_shmem base;
};

struct vsock_bo {
   struct vn_renderer_bo base;

   uint32_t blob_flags;

   /* list of bos mapped by the client and not uploaded yet */
   struct list_head dirty_head;
   bool is_dirty;
};

struct vsock_sync {
   struct vn_renderer_sync base;
};

struct vsock {
   struct vn_renderer base;

   struct vn_instance *instance;

   mtx_t sock_mutex;
   int sock_fd;

   uint32_t seq_out;
   uint32_t protocol_version;
   uint32_t max_timeline_count;

   struct {
      uint32_t id;
      uint32_t version;
      struct virgl_renderer_capset_venus data;
   } capset;

   /* bytes of shmem transfers, for debugging */
   uint64_t stat_upload_bytes;
   uint64_t stat_download_bytes;

   struct util_sparse_array shmem_array;
   struct util_sparse_array bo_array;

   /* bos whose client-side contents may not have reached the renderer */
   struct list_head dirty_bos;

   struct vn_renderer_shmem_cache shmem_cache;
};

static void
vsock_abort(struct vsock *vsock, const char *what)
{
   vn_log(vsock->instance, "vsock transport: %s", what);
   abort();
}

static void
vsock_read(struct vsock *vsock, void *buf, size_t size)
{
   do {
      const ssize_t ret = read(vsock->sock_fd, buf, size);
      if (unlikely(ret <= 0)) {
         if (ret < 0 && errno == EINTR)
            continue;
         vsock_abort(vsock, "lost connection to renderer");
      }

      buf += ret;
      size -= (size_t)ret;
   } while (size);
}

static void
vsock_write(struct vsock *vsock, const void *buf, size_t size)
{
   do {
      const ssize_t ret = write(vsock->sock_fd, buf, size);
      if (unlikely(ret <= 0)) {
         if (ret < 0 && errno == EINTR)
            continue;
         vsock_abort(vsock, "lost connection to renderer");
      }

      buf += ret;
      size -= (size_t)ret;
   } while (size);
}

/*
 * Send a request and read the response header.  The caller must have taken
 * sock_mutex and is responsible for consuming any response payload.
 */
static struct vsock_msg_hdr
vsock_call_locked(struct vsock *vsock,
                  uint32_t type,
                  uint32_t arg,
                  const void *payload,
                  size_t len)
{
   struct vsock_msg_hdr req = {
      .type = type,
      .seq = ++vsock->seq_out,
      .len = len,
      .arg = arg,
   };

   vsock_write(vsock, &req, sizeof(req));
   if (len)
      vsock_write(vsock, payload, len);

   struct vsock_msg_hdr resp;
   vsock_read(vsock, &resp, sizeof(resp));

   if (resp.seq != req.seq)
      vsock_abort(vsock, "response out of order");

   return resp;
}

static void
vsock_expect_ok_locked(struct vsock *vsock,
                       uint32_t type,
                       uint32_t arg,
                       const void *payload,
                       size_t len)
{
   const struct vsock_msg_hdr resp =
      vsock_call_locked(vsock, type, arg, payload, len);
   if (resp.type != VSOCK_RESP_OK)
      vsock_abort(vsock, "request failed");
}

static uint32_t
vsock_call_u32_locked(struct vsock *vsock,
                      uint32_t type,
                      uint32_t arg,
                      const void *payload,
                      size_t len)
{
   const struct vsock_msg_hdr resp =
      vsock_call_locked(vsock, type, arg, payload, len);
   if (resp.type != VSOCK_RESP_OK)
      vsock_abort(vsock, "request failed");

   return resp.arg;
}

static void
vsock_read_payload_locked(struct vsock *vsock,
                          const struct vsock_msg_hdr *resp,
                          void *data,
                          size_t size)
{
   assert(resp->type == VSOCK_RESP_DATA);

   const size_t copy = MIN2(size, (size_t)resp->len);
   if (copy)
      vsock_read(vsock, data, copy);

   if (resp->len > copy) {
      char temp[256];
      size_t left = resp->len - copy;
      while (left) {
         const size_t chunk = MIN2(left, sizeof(temp));
         vsock_read(vsock, temp, chunk);
         left -= chunk;
      }
   }
}

static void
vsock_vcmd_create_renderer(struct vsock *vsock, const char *name)
{
   const size_t size = strlen(name) + 1;
   mtx_lock(&vsock->sock_mutex);
   vsock_expect_ok_locked(vsock, VSOCK_MSG_CREATE_RENDERER, 0, name, size);
   mtx_unlock(&vsock->sock_mutex);
}

static uint32_t
vsock_vcmd_protocol_version(struct vsock *vsock)
{
   mtx_lock(&vsock->sock_mutex);
   const uint32_t version =
      vsock_call_u32_locked(vsock, VSOCK_MSG_PROTOCOL_VERSION, 0, NULL, 0);
   mtx_unlock(&vsock->sock_mutex);

   return version;
}

static uint32_t
vsock_vcmd_get_param(struct vsock *vsock, uint32_t param)
{
   mtx_lock(&vsock->sock_mutex);
   const uint32_t val =
      vsock_call_u32_locked(vsock, VSOCK_MSG_GET_PARAM, param, NULL, 0);
   mtx_unlock(&vsock->sock_mutex);

   return val;
}

static bool
vsock_vcmd_get_capset(struct vsock *vsock,
                      uint32_t id,
                      uint32_t version,
                      void *capset,
                      size_t capset_size)
{
   mtx_lock(&vsock->sock_mutex);
   const struct vsock_msg_hdr resp = vsock_call_locked(
      vsock, VSOCK_MSG_GET_CAPSET, id, &version, sizeof(version));
   if (resp.type == VSOCK_RESP_ERR) {
      mtx_unlock(&vsock->sock_mutex);
      return false;
   }

   vsock_read_payload_locked(vsock, &resp, capset, capset_size);
   mtx_unlock(&vsock->sock_mutex);

   return true;
}

static void
vsock_vcmd_context_init(struct vsock *vsock, uint32_t capset_id)
{
   mtx_lock(&vsock->sock_mutex);
   vsock_expect_ok_locked(vsock, VSOCK_MSG_CONTEXT_INIT, capset_id, NULL, 0);
   mtx_unlock(&vsock->sock_mutex);
}

static uint32_t
vsock_vcmd_resource_create_locked(struct vsock *vsock,
                                  uint32_t blob_flags,
                                  VkDeviceSize size,
                                  vn_object_id blob_id)
{
   const struct vsock_res_create req = {
      .size = size,
      .blob_id = blob_id,
   };

   return vsock_call_u32_locked(vsock, VSOCK_MSG_RES_CREATE, blob_flags, &req,
                                sizeof(req));
}

static void
vsock_vcmd_resource_unref_locked(struct vsock *vsock, uint32_t res_id)
{
   vsock_expect_ok_locked(vsock, VSOCK_MSG_RES_UNREF, res_id, NULL, 0);
}

static void
vsock_vcmd_resource_map_locked(struct vsock *vsock,
                               uint32_t res_id,
                               uint64_t *out_size)
{
   const struct vsock_msg_hdr resp =
      vsock_call_locked(vsock, VSOCK_MSG_RES_MAP, res_id, NULL, 0);
   uint64_t size = 0;
   vsock_read_payload_locked(vsock, &resp, &size, sizeof(size));
   *out_size = size;
}

static void
vsock_vcmd_resource_write_locked(struct vsock *vsock,
                                 uint32_t res_id,
                                 uint32_t offset,
                                 const void *data,
                                 size_t size)
{
   uint32_t hdr[2] = { offset, (uint32_t)size };
   struct vsock_msg_hdr req = {
      .type = VSOCK_MSG_RES_WRITE,
      .seq = ++vsock->seq_out,
      .len = sizeof(hdr) + size,
      .arg = res_id,
   };

   vsock_write(vsock, &req, sizeof(req));
   vsock_write(vsock, hdr, sizeof(hdr));
   if (size)
      vsock_write(vsock, data, size);

   struct vsock_msg_hdr resp;
   vsock_read(vsock, &resp, sizeof(resp));
   if (resp.seq != req.seq || resp.type != VSOCK_RESP_OK)
      vsock_abort(vsock, "resource write failed");

   vsock->stat_upload_bytes += size;
}

static void
vsock_vcmd_resource_read_locked(struct vsock *vsock,
                                uint32_t res_id,
                                uint32_t offset,
                                void *data,
                                size_t size)
{
   const uint32_t req[2] = { offset, (uint32_t)size };
   const struct vsock_msg_hdr resp = vsock_call_locked(
      vsock, VSOCK_MSG_RES_READ, res_id, req, sizeof(req));
   vsock_read_payload_locked(vsock, &resp, data, size);

   vsock->stat_download_bytes += size;
}

static void
vsock_vcmd_submit_locked(struct vsock *vsock,
                         const struct vn_renderer_submit_batch *batch)
{
   if (!batch->cs_size && !batch->sync_count)
      return;

   const size_t payload_size = sizeof(uint32_t) * 2 +
                               sizeof(struct vsock_wire_sync) *
                                  batch->sync_count +
                               batch->cs_size;
   uint8_t *payload = malloc(payload_size);
   if (!payload)
      vsock_abort(vsock, "out of memory for submit");

   uint8_t *p = payload;
   memcpy(p, &(const uint32_t){ VSOCK_CTX_ID }, sizeof(uint32_t));
   p += sizeof(uint32_t);
   memcpy(p, &batch->sync_count, sizeof(uint32_t));
   p += sizeof(uint32_t);

   for (uint32_t i = 0; i < batch->sync_count; i++) {
      const uint64_t val = batch->sync_values[i];
      const struct vsock_wire_sync sync = {
         .sync_id = batch->syncs[i]->sync_id,
         .val_lo = (uint32_t)val,
         .val_hi = (uint32_t)(val >> 32),
      };
      memcpy(p, &sync, sizeof(sync));
      p += sizeof(sync);
   }

   if (batch->cs_size) {
      memcpy(p, batch->cs_data, batch->cs_size);
      p += batch->cs_size;
   }

   vsock_expect_ok_locked(vsock, VSOCK_MSG_SUBMIT, batch->ring_idx, payload,
                          payload_size);
   free(payload);
}

static uint32_t
vsock_vcmd_sync_create_locked(struct vsock *vsock, uint64_t initial_val)
{
   return vsock_call_u32_locked(vsock, VSOCK_MSG_SYNC_CREATE, 0, &initial_val,
                                sizeof(initial_val));
}

static void
vsock_vcmd_sync_unref_locked(struct vsock *vsock, uint32_t sync_id)
{
   vsock_expect_ok_locked(vsock, VSOCK_MSG_SYNC_UNREF, sync_id, NULL, 0);
}

static uint64_t
vsock_vcmd_sync_read_locked(struct vsock *vsock, uint32_t sync_id)
{
   const struct vsock_msg_hdr resp =
      vsock_call_locked(vsock, VSOCK_MSG_SYNC_READ, sync_id, NULL, 0);
   uint64_t val = 0;
   vsock_read_payload_locked(vsock, &resp, &val, sizeof(val));

   return val;
}

static void
vsock_vcmd_sync_write_locked(struct vsock *vsock,
                             uint32_t sync_id,
                             uint64_t val)
{
   vsock_expect_ok_locked(vsock, VSOCK_MSG_SYNC_WRITE, sync_id, &val,
                          sizeof(val));
}

static VkResult
vsock_vcmd_sync_wait_locked(struct vsock *vsock,
                            uint32_t flags,
                            uint32_t timeout_ms,
                            struct vn_renderer_sync *const *syncs,
                            const uint64_t *vals,
                            uint32_t count)
{
   const size_t payload_size =
      sizeof(uint32_t) * 2 + sizeof(struct vsock_wire_sync) * count;
   uint8_t *payload = malloc(payload_size);
   if (!payload)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   uint8_t *p = payload;
   memcpy(p, &timeout_ms, sizeof(uint32_t));
   p += sizeof(uint32_t);
   memcpy(p, &count, sizeof(uint32_t));
   p += sizeof(uint32_t);

   for (uint32_t i = 0; i < count; i++) {
      const uint64_t val = vals[i];
      const struct vsock_wire_sync sync = {
         .sync_id = syncs[i]->sync_id,
         .val_lo = (uint32_t)val,
         .val_hi = (uint32_t)(val >> 32),
      };
      memcpy(p, &sync, sizeof(sync));
      p += sizeof(sync);
   }

   const struct vsock_msg_hdr resp =
      vsock_call_locked(vsock, VSOCK_MSG_SYNC_WAIT, flags, payload,
                        payload_size);
   free(payload);

   if (resp.type == VSOCK_RESP_OK)
      return VK_SUCCESS;
   if (resp.type == VSOCK_RESP_TIMEOUT)
      return VK_TIMEOUT;

   return VK_ERROR_DEVICE_LOST;
}

static void
vsock_upload_locked(struct vsock *vsock, struct vsock_bo *bo, uint32_t offset,
                    size_t size)
{
   const uint8_t *ptr = bo->base.mmap_ptr;
   assert(ptr);

   vsock_vcmd_resource_write_locked(vsock, bo->base.res_id, offset,
                                    ptr + offset, size);
}

static void
vsock_flush_dirty_bos_locked(struct vsock *vsock)
{
   if (list_is_empty(&vsock->dirty_bos))
      return;

   list_for_each_entry_safe (struct vsock_bo, bo, &vsock->dirty_bos,
                              dirty_head) {
      /* upload the whole mapped range; there is no write tracking */
      if (bo->base.mmap_ptr && bo->base.mmap_size)
         vsock_upload_locked(vsock, bo, 0, bo->base.mmap_size);
      list_del(&bo->dirty_head);
      bo->is_dirty = false;
   }
}

static void *
vsock_bo_map(struct vn_renderer *renderer,
             struct vn_renderer_bo *_bo,
             void *placed_addr)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_bo *bo = (struct vsock_bo *)_bo;

   if (!bo->base.mmap_ptr && bo->base.mmap_size) {
      const int prot = PROT_READ | PROT_WRITE;
      const int flags =
         MAP_PRIVATE | MAP_ANONYMOUS | (placed_addr ? MAP_FIXED_NOREPLACE : 0);
      void *ptr = mmap(placed_addr, bo->base.mmap_size, prot, flags, -1, 0);
      if (ptr == MAP_FAILED) {
         vn_log(vsock->instance, "failed to allocate %zu bytes: %s",
                bo->base.mmap_size, strerror(errno));
         return NULL;
      }
      bo->base.mmap_ptr = ptr;

      /* the renderer has its own copy; make sure it is initialized before
       * the client starts writing (the client may rely on zero-filled
       * memory)
       */
      mtx_lock(&vsock->sock_mutex);
      vsock_vcmd_resource_write_locked(vsock, bo->base.res_id, 0, ptr,
                                       bo->base.mmap_size);
      mtx_unlock(&vsock->sock_mutex);
   }

   /* everything the client writes until the next submit must be uploaded,
    * because there is no way to track writes to a plain mmap
    */
   mtx_lock(&vsock->sock_mutex);
   if (!bo->is_dirty) {
      list_addtail(&bo->dirty_head, &vsock->dirty_bos);
      bo->is_dirty = true;
   }
   mtx_unlock(&vsock->sock_mutex);

   return bo->base.mmap_ptr;
}

static void
vsock_bo_flush(struct vn_renderer *renderer,
               struct vn_renderer_bo *_bo,
               VkDeviceSize offset,
               VkDeviceSize size)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_bo *bo = (struct vsock_bo *)_bo;

   if (!bo->base.mmap_ptr || !size)
      return;

   mtx_lock(&vsock->sock_mutex);
   vsock_upload_locked(vsock, bo, (uint32_t)offset, (size_t)size);
   mtx_unlock(&vsock->sock_mutex);
}

static void
vsock_bo_invalidate(struct vn_renderer *renderer,
                    struct vn_renderer_bo *_bo,
                    VkDeviceSize offset,
                    VkDeviceSize size)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_bo *bo = (struct vsock_bo *)_bo;

   if (!bo->base.mmap_ptr || !size)
      return;

   mtx_lock(&vsock->sock_mutex);
   vsock_vcmd_resource_read_locked(vsock, bo->base.res_id, (uint32_t)offset,
                                   bo->base.mmap_ptr + offset, (size_t)size);
   mtx_unlock(&vsock->sock_mutex);
}

static int
vsock_bo_export_dma_buf(UNUSED struct vn_renderer *renderer,
                        UNUSED struct vn_renderer_bo *bo)
{
   /* resources live in two address spaces; there is no fd to hand out */
   return -1;
}

static bool
vsock_bo_destroy(struct vn_renderer *renderer, struct vn_renderer_bo *_bo)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_bo *bo = (struct vsock_bo *)_bo;

   mtx_lock(&vsock->sock_mutex);
   if (bo->is_dirty) {
      list_del(&bo->dirty_head);
      bo->is_dirty = false;
   }
   vsock_vcmd_resource_unref_locked(vsock, bo->base.res_id);
   mtx_unlock(&vsock->sock_mutex);

   if (bo->base.mmap_ptr)
      munmap(bo->base.mmap_ptr, bo->base.mmap_size);

   return true;
}

static uint32_t
vsock_bo_blob_flags(VkMemoryPropertyFlags flags,
                    VkExternalMemoryHandleTypeFlags external_handles)
{
   uint32_t blob_flags = 0;
   if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      blob_flags |= VSOCK_BLOB_MAPPABLE;
   if (external_handles)
      blob_flags |= VSOCK_BLOB_SHAREABLE;
   return blob_flags;
}

static VkResult
vsock_bo_create_from_device_memory(
   struct vn_renderer *renderer,
   struct vn_renderer_submit_batch *batch,
   VkDeviceSize size,
   vn_object_id mem_id,
   VkMemoryPropertyFlags flags,
   VkExternalMemoryHandleTypeFlags external_handles,
   struct vn_renderer_bo **out_bo)
{
   struct vsock *vsock = (struct vsock *)renderer;
   const uint32_t blob_flags = vsock_bo_blob_flags(flags, external_handles);

   mtx_lock(&vsock->sock_mutex);
   if (batch)
      vsock_vcmd_submit_locked(vsock, batch);
   const uint32_t res_id =
      vsock_vcmd_resource_create_locked(vsock, blob_flags, size, mem_id);
   if (!res_id) {
      mtx_unlock(&vsock->sock_mutex);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   mtx_unlock(&vsock->sock_mutex);

   struct vsock_bo *bo = util_sparse_array_get(&vsock->bo_array, res_id);
   *bo = (struct vsock_bo){
      .base = {
         .refcount = VN_REFCOUNT_INIT(1),
         .res_id = res_id,
         .mmap_size = size,
      },
      .blob_flags = blob_flags,
   };

   *out_bo = &bo->base;
   return VK_SUCCESS;
}

static void
vsock_shmem_destroy_now(struct vn_renderer *renderer,
                        struct vn_renderer_shmem *_shmem)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_shmem *shmem = (struct vsock_shmem *)_shmem;

   mtx_lock(&vsock->sock_mutex);
   vsock_vcmd_resource_unref_locked(vsock, shmem->base.res_id);
   mtx_unlock(&vsock->sock_mutex);

   munmap(shmem->base.mmap_ptr, shmem->base.mmap_size);
}

static void
vsock_shmem_destroy(struct vn_renderer *renderer,
                    struct vn_renderer_shmem *shmem)
{
   struct vsock *vsock = (struct vsock *)renderer;

   if (vn_renderer_shmem_cache_add(&vsock->shmem_cache, shmem))
      return;

   vsock_shmem_destroy_now(&vsock->base, shmem);
}

static struct vn_renderer_shmem *
vsock_shmem_create(struct vn_renderer *renderer, size_t size)
{
   struct vsock *vsock = (struct vsock *)renderer;

   struct vn_renderer_shmem *cached_shmem =
      vn_renderer_shmem_cache_get(&vsock->shmem_cache, size);
   if (cached_shmem) {
      cached_shmem->refcount = VN_REFCOUNT_INIT(1);
      return cached_shmem;
   }

   void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   if (ptr == MAP_FAILED)
      return NULL;

   mtx_lock(&vsock->sock_mutex);
   const uint32_t res_id =
      vsock_vcmd_resource_create_locked(vsock, VSOCK_BLOB_MAPPABLE, size, 0);
   if (res_id) {
      /* the renderer starts with a zeroed buffer as well; no need to upload
       * anything here
       */
      uint64_t host_size = 0;
      vsock_vcmd_resource_map_locked(vsock, res_id, &host_size);
   }
   mtx_unlock(&vsock->sock_mutex);

   if (!res_id) {
      munmap(ptr, size);
      return NULL;
   }

   struct vsock_shmem *shmem = util_sparse_array_get(&vsock->shmem_array, res_id);
   *shmem = (struct vsock_shmem){
      .base = {
         .refcount = VN_REFCOUNT_INIT(1),
         .res_id = res_id,
         .mmap_size = size,
         .mmap_ptr = ptr,
      },
   };

   return &shmem->base;
}

static void
vsock_shmem_flush(struct vn_renderer *renderer,
                  struct vn_renderer_shmem *shmem,
                  size_t offset,
                  size_t size)
{
   struct vsock *vsock = (struct vsock *)renderer;

   if (!size)
      return;

   mtx_lock(&vsock->sock_mutex);
   vsock_vcmd_resource_write_locked(vsock, shmem->res_id, (uint32_t)offset,
                                    (uint8_t *)shmem->mmap_ptr + offset, size);
   mtx_unlock(&vsock->sock_mutex);
}

static void
vsock_shmem_invalidate(struct vn_renderer *renderer,
                       struct vn_renderer_shmem *shmem,
                       size_t offset,
                       size_t size)
{
   struct vsock *vsock = (struct vsock *)renderer;

   if (!size)
      return;

   mtx_lock(&vsock->sock_mutex);
   vsock_vcmd_resource_read_locked(vsock, shmem->res_id, (uint32_t)offset,
                                   (uint8_t *)shmem->mmap_ptr + offset, size);
   mtx_unlock(&vsock->sock_mutex);
}

static VkResult
vsock_sync_write(struct vn_renderer *renderer,
                 struct vn_renderer_sync *_sync,
                 uint64_t val)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_sync *sync = (struct vsock_sync *)_sync;

   mtx_lock(&vsock->sock_mutex);
   vsock_vcmd_sync_write_locked(vsock, sync->base.sync_id, val);
   mtx_unlock(&vsock->sock_mutex);

   return VK_SUCCESS;
}

static VkResult
vsock_sync_read(struct vn_renderer *renderer,
                struct vn_renderer_sync *_sync,
                uint64_t *val)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_sync *sync = (struct vsock_sync *)_sync;

   mtx_lock(&vsock->sock_mutex);
   *val = vsock_vcmd_sync_read_locked(vsock, sync->base.sync_id);
   mtx_unlock(&vsock->sock_mutex);

   return VK_SUCCESS;
}

static VkResult
vsock_sync_reset(struct vn_renderer *renderer, struct vn_renderer_sync *sync)
{
   return vsock_sync_write(renderer, sync, 0);
}

static void
vsock_sync_destroy(struct vn_renderer *renderer,
                   struct vn_renderer_sync *_sync)
{
   struct vsock *vsock = (struct vsock *)renderer;
   struct vsock_sync *sync = (struct vsock_sync *)_sync;

   mtx_lock(&vsock->sock_mutex);
   vsock_vcmd_sync_unref_locked(vsock, sync->base.sync_id);
   mtx_unlock(&vsock->sock_mutex);

   free(sync);
}

static VkResult
vsock_sync_create(struct vn_renderer *renderer,
                  uint64_t initial_val,
                  struct vn_renderer_sync **out_sync)
{
   struct vsock *vsock = (struct vsock *)renderer;

   struct vsock_sync *sync = calloc(1, sizeof(*sync));
   if (!sync)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   mtx_lock(&vsock->sock_mutex);
   sync->base.sync_id = vsock_vcmd_sync_create_locked(vsock, initial_val);
   mtx_unlock(&vsock->sock_mutex);

   *out_sync = &sync->base;
   return VK_SUCCESS;
}

static VkResult
vsock_submit(struct vn_renderer *renderer,
             const struct vn_renderer_submit_batch *batch)
{
   struct vsock *vsock = (struct vsock *)renderer;

   mtx_lock(&vsock->sock_mutex);
   vsock_flush_dirty_bos_locked(vsock);
   vsock_vcmd_submit_locked(vsock, batch);
   mtx_unlock(&vsock->sock_mutex);

   return VK_SUCCESS;
}

static VkResult
vsock_wait(struct vn_renderer *renderer, const struct vn_renderer_wait *wait)
{
   struct vsock *vsock = (struct vsock *)renderer;
   const uint32_t flags = wait->wait_any ? 1 : 0;
   const int poll_timeout = vn_timeout_to_poll_timeout(wait->timeout);
   const uint32_t timeout_ms =
      poll_timeout < 0 ? UINT32_MAX : (uint32_t)poll_timeout;

   mtx_lock(&vsock->sock_mutex);
   const VkResult result = vsock_vcmd_sync_wait_locked(
      vsock, flags, timeout_ms, wait->syncs, wait->sync_values,
      wait->sync_count);
   mtx_unlock(&vsock->sock_mutex);

   return result;
}

static void
vsock_init_renderer_info(struct vsock *vsock)
{
   struct vn_renderer_info *info = &vsock->base.info;

   info->pci.vendor_id = VSOCK_PCI_VENDOR_ID;
   info->pci.device_id = VSOCK_PCI_DEVICE_ID;

   info->has_dma_buf_import = false;
   info->has_external_sync = false;
   info->has_timeline_sync = !VN_PERF(NO_TIMELINE_SYNC);
   info->has_implicit_fencing = false;

   const struct virgl_renderer_capset_venus *capset = &vsock->capset.data;
   info->wire_format_version = capset->wire_format_version;
   info->vk_xml_version = capset->vk_xml_version;
   info->vk_ext_command_serialization_spec_version =
      capset->vk_ext_command_serialization_spec_version;
   info->vk_mesa_venus_protocol_spec_version =
      capset->vk_mesa_venus_protocol_spec_version;
   assert(capset->supports_blob_id_0);

   STATIC_ASSERT(sizeof(info->vk_extension_mask) >=
                 sizeof(capset->vk_extension_mask1));
   memcpy(info->vk_extension_mask, capset->vk_extension_mask1,
          sizeof(capset->vk_extension_mask1));

   assert(capset->allow_vk_wait_syncs);
   assert(capset->supports_multiple_timelines);
   info->max_timeline_count = vsock->max_timeline_count;
}

static void
vsock_destroy(struct vn_renderer *renderer,
              const VkAllocationCallbacks *alloc)
{
   struct vsock *vsock = (struct vsock *)renderer;

   vn_renderer_shmem_cache_fini(&vsock->shmem_cache);

   if (vsock->sock_fd >= 0) {
      shutdown(vsock->sock_fd, SHUT_RDWR);
      close(vsock->sock_fd);
   }

   mtx_destroy(&vsock->sock_mutex);
   util_sparse_array_finish(&vsock->shmem_array);
   util_sparse_array_finish(&vsock->bo_array);

   vk_free(alloc, vsock);
}

static int
vsock_connect_unix(struct vsock *vsock, const char *path)
{
   struct sockaddr_un un;
   int sock;

   sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
   if (sock < 0) {
      vn_log(vsock->instance, "failed to create a socket: %s",
             strerror(errno));
      return -1;
   }

   memset(&un, 0, sizeof(un));
   un.sun_family = AF_UNIX;
   snprintf(un.sun_path, sizeof(un.sun_path), "%s", path);

   if (connect(sock, (struct sockaddr *)&un, sizeof(un)) == -1) {
      vn_log(vsock->instance, "failed to connect to %s: %s", path,
             strerror(errno));
      close(sock);
      return -1;
   }

   return sock;
}

static int
vsock_connect_vsock(struct vsock *vsock, const char *addr)
{
#ifdef __linux__
   struct sockaddr_vm vm;
   int sock;

   uint32_t cid = 0;
   uint32_t port = 0;
   if (sscanf(addr, "%u:%u", &cid, &port) != 2 || !port) {
      vn_log(vsock->instance, "invalid VENUS_COPY_VSOCK '%s'", addr);
      return -1;
   }

   sock = socket(AF_VSOCK, SOCK_STREAM | SOCK_CLOEXEC, 0);
   if (sock < 0) {
      vn_log(vsock->instance, "failed to create a vsock socket: %s",
             strerror(errno));
      return -1;
   }

   memset(&vm, 0, sizeof(vm));
   vm.svm_family = AF_VSOCK;
   vm.svm_cid = cid;
   vm.svm_port = port;

   if (connect(sock, (struct sockaddr *)&vm, sizeof(vm)) == -1) {
      vn_log(vsock->instance, "failed to connect to vsock %s: %s", addr,
             strerror(errno));
      close(sock);
      return -1;
   }

   return sock;
#else
   vn_log(vsock->instance, "vsock is not supported on this platform");
   return -1;
#endif
}

static VkResult
vsock_init(struct vsock *vsock)
{
   const char *socket_path = os_get_option("VENUS_COPY_SOCKET");
   const char *vsock_addr = os_get_option("VENUS_COPY_VSOCK");

   util_sparse_array_init(&vsock->shmem_array, sizeof(struct vsock_shmem),
                          1024);
   util_sparse_array_init(&vsock->bo_array, sizeof(struct vsock_bo), 1024);

   mtx_init(&vsock->sock_mutex, mtx_plain);
   list_inithead(&vsock->dirty_bos);

   if (!socket_path && !vsock_addr) {
      vn_log(vsock->instance, "neither VENUS_COPY_SOCKET nor "
             "VENUS_COPY_VSOCK is set");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   vsock->sock_fd = socket_path ? vsock_connect_unix(vsock, socket_path)
                                : vsock_connect_vsock(vsock, vsock_addr);
   if (vsock->sock_fd < 0)
      return VK_ERROR_INITIALIZATION_FAILED;

   const char *renderer_name = util_get_process_name();
   if (!renderer_name)
      renderer_name = "venus";
   vsock_vcmd_create_renderer(vsock, renderer_name);

   const uint32_t version = vsock_vcmd_protocol_version(vsock);
   if (version != VSOCK_PROTOCOL_VERSION) {
      vn_log(vsock->instance, "unsupported transport version %u", version);
      return VK_ERROR_INITIALIZATION_FAILED;
   }
   vsock->protocol_version = version;

   vsock->max_timeline_count =
      vsock_vcmd_get_param(vsock, VSOCK_PARAM_MAX_TIMELINE_COUNT);
   if (!vsock->max_timeline_count) {
      vn_log(vsock->instance, "no timeline support");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   vsock->capset.id = VIRTGPU_DRM_CAPSET_VENUS;
   vsock->capset.version = 0;
   if (!vsock_vcmd_get_capset(vsock, vsock->capset.id,
                              vsock->capset.version, &vsock->capset.data,
                              sizeof(vsock->capset.data))) {
      vn_log(vsock->instance, "no venus capset");
      return VK_ERROR_INITIALIZATION_FAILED;
   }

   vn_renderer_shmem_cache_init(&vsock->shmem_cache, &vsock->base,
                                vsock_shmem_destroy_now);

   vsock_vcmd_context_init(vsock, vsock->capset.id);

   vsock_init_renderer_info(vsock);

   vsock->base.ops.destroy = vsock_destroy;
   vsock->base.ops.submit = vsock_submit;
   vsock->base.ops.wait = vsock_wait;

   vsock->base.shmem_ops.create = vsock_shmem_create;
   vsock->base.shmem_ops.destroy = vsock_shmem_destroy;
   vsock->base.shmem_ops.flush = vsock_shmem_flush;
   vsock->base.shmem_ops.invalidate = vsock_shmem_invalidate;

   vsock->base.bo_ops.create_from_device_memory =
      vsock_bo_create_from_device_memory;
   vsock->base.bo_ops.create_from_dma_buf = NULL;
   vsock->base.bo_ops.destroy = vsock_bo_destroy;
   vsock->base.bo_ops.export_dma_buf = vsock_bo_export_dma_buf;
   vsock->base.bo_ops.export_sync_file =
      vn_renderer_bo_export_sync_file_internal;
   vsock->base.bo_ops.map = vsock_bo_map;
   vsock->base.bo_ops.flush = vsock_bo_flush;
   vsock->base.bo_ops.invalidate = vsock_bo_invalidate;

   vsock->base.sync_ops.create = vsock_sync_create;
   vsock->base.sync_ops.create_from_syncobj = NULL;
   vsock->base.sync_ops.destroy = vsock_sync_destroy;
   vsock->base.sync_ops.export_syncobj = NULL;
   vsock->base.sync_ops.reset = vsock_sync_reset;
   vsock->base.sync_ops.read = vsock_sync_read;
   vsock->base.sync_ops.write = vsock_sync_write;

   return VK_SUCCESS;
}

VkResult
vn_renderer_create_vsock(struct vn_instance *instance,
                         const VkAllocationCallbacks *alloc,
                         struct vn_renderer **renderer)
{
   struct vsock *vsock = vk_zalloc(alloc, sizeof(*vsock), VN_DEFAULT_ALIGN,
                                   VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!vsock)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   vsock->instance = instance;
   vsock->sock_fd = -1;

   VkResult result = vsock_init(vsock);
   if (result != VK_SUCCESS) {
      vsock_destroy(&vsock->base, alloc);
      return result;
   }

   *renderer = &vsock->base;

   if (VN_DEBUG(INIT))
      vn_log(vsock->instance, "vsock (copy) backend initialized");

   return VK_SUCCESS;
}
