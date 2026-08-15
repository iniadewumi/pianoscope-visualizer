/**
 * Persistent local cache for remote video clips (Cache Storage).
 * First play can stream from the network; later plays use a blob URL from cache.
 */

window.VideoCache = (function () {
    "use strict";

    const CACHE_NAME = 'psv-videos-v1';
    const inflight = new Map();

    function supported() {
        return typeof caches !== 'undefined';
    }

    function isCacheable(url) {
        return typeof url === 'string' && /^https?:\/\//i.test(url);
    }

    async function openCache() {
        if (!supported()) return null;
        return caches.open(CACHE_NAME);
    }

    async function has(url) {
        if (!isCacheable(url)) return false;
        const cache = await openCache();
        if (!cache) return false;
        const match = await cache.match(url, { ignoreSearch: false });
        return Boolean(match);
    }

    async function getBlobUrl(url) {
        if (!isCacheable(url)) return null;
        const cache = await openCache();
        if (!cache) return null;

        const response = await cache.match(url);
        if (!response || !response.ok) return null;

        const blob = await response.blob();
        if (!blob || !blob.size) return null;
        return URL.createObjectURL(blob);
    }

    /**
     * Fetch a remote clip and store the full response. Dedupes concurrent puts
     * for the same URL. Returns true on success.
     */
    async function put(url, { onProgress } = {}) {
        if (!isCacheable(url)) return false;
        if (!supported()) return false;

        if (inflight.has(url)) return inflight.get(url);

        const job = (async () => {
            try {
                if (await has(url)) return true;

                const response = await fetch(url, { mode: 'cors', credentials: 'omit' });
                if (!response.ok) {
                    throw new Error(`Cache fetch failed (${response.status})`);
                }

                // Prefer a cloned body so we can report byte progress when possible.
                const lengthHeader = response.headers.get('content-length');
                const total = lengthHeader ? parseInt(lengthHeader, 10) : 0;

                if (onProgress && response.body && typeof ReadableStream !== 'undefined') {
                    const reader = response.body.getReader();
                    const chunks = [];
                    let received = 0;

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        chunks.push(value);
                        received += value.byteLength;
                        onProgress({ received, total: total || 0 });
                    }

                    const body = new Blob(chunks, {
                        type: response.headers.get('content-type') || 'video/mp4'
                    });
                    const cache = await openCache();
                    await cache.put(url, new Response(body, {
                        status: 200,
                        statusText: 'OK',
                        headers: {
                            'Content-Type': body.type,
                            'Content-Length': String(body.size)
                        }
                    }));
                } else {
                    const cache = await openCache();
                    await cache.put(url, response.clone());
                    if (onProgress && total) onProgress({ received: total, total });
                }

                return true;
            } catch (error) {
                console.warn('VideoCache.put failed:', url, error);
                return false;
            } finally {
                inflight.delete(url);
            }
        })();

        inflight.set(url, job);
        return job;
    }

    async function remove(url) {
        if (!isCacheable(url)) return false;
        const cache = await openCache();
        if (!cache) return false;
        return cache.delete(url);
    }

    async function clear() {
        if (!supported()) return false;
        return caches.delete(CACHE_NAME);
    }

    async function cachedUrls() {
        const cache = await openCache();
        if (!cache) return [];
        const keys = await cache.keys();
        return keys.map((request) => request.url);
    }

    /**
     * Resolve a playable URL: cached blob when available, otherwise the network src.
     * Optionally warm the cache in the background after a network miss.
     */
    async function resolveForPlayback(url, { warm = true } = {}) {
        if (!isCacheable(url)) {
            return { playUrl: url, fromCache: false, isObjectUrl: false };
        }

        const blobUrl = await getBlobUrl(url);
        if (blobUrl) {
            return { playUrl: blobUrl, fromCache: true, isObjectUrl: true };
        }

        if (warm) {
            // Fire-and-forget; next play can hit cache.
            put(url).catch(() => {});
        }

        return { playUrl: url, fromCache: false, isObjectUrl: false };
    }

    return {
        supported,
        isCacheable,
        has,
        getBlobUrl,
        put,
        remove,
        clear,
        cachedUrls,
        resolveForPlayback
    };
})();
