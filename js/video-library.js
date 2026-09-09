/**
 * Video Library for Pianoscope Visualizer
 *
 * Discovers clips in media/videos/ at runtime (no hardcoded catalog), then
 * generates poster frames + metadata on demand.
 *
 * Visualizer source stills live in media/images/ (UI assets stay in images/).
 */

window.VideoLibrary = (function () {
    "use strict";

    const THUMB_CACHE_PREFIX = 'psv:thumb:v1:';
    const THUMB_WIDTH = 320;
    const THUMB_HEIGHT = 180;
    const MEDIA_DIR = 'media';
    const VIDEOS_DIR = `${MEDIA_DIR}/videos`;
    const IMAGES_DIR = `${MEDIA_DIR}/images`;
    const VIDEO_EXT = /\.(mp4|webm|mov|ogg|ogv|m4v)$/i;
    const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|avif)$/i;

    const VIDEO_KIND = { dir: VIDEOS_DIR, ext: VIDEO_EXT, kind: 'video', jsonKeys: ['videos', 'files'] };
    const IMAGE_KIND = { dir: IMAGES_DIR, ext: IMAGE_EXT, kind: 'image', jsonKeys: ['images', 'files'] };

    function isImageSrc(src) {
        if (!src || typeof src !== 'string') return false;
        return IMAGE_EXT.test(src.split('?')[0].split('#')[0]);
    }

    function decodeHtml(value) {
        const textarea = document.createElement('textarea');
        textarea.innerHTML = value;
        return textarea.value;
    }

    function isAbsoluteUrl(value) {
        return /^https?:\/\//i.test(value);
    }

    function fileNameFromHref(href) {
        let path = decodeHtml(href).split('?')[0].split('#')[0];
        try {
            path = decodeURIComponent(path);
        } catch (error) {
            // Keep the raw path if it was not percent-encoded.
        }
        const name = path.replace(/\\/g, '/').split('/').filter(Boolean).pop();
        if (!name || name === '.' || name === '..') return null;
        return name;
    }

    function sortEntries(entries) {
        return entries.slice().sort((a, b) =>
            a.title.localeCompare(b.title, undefined, { numeric: true, sensitivity: 'base' })
        );
    }

    function entriesFromNames(names, spec) {
        const unique = [...new Set(names.filter((name) => name && spec.ext.test(name)))];
        return sortEntries(unique.map((name) => ({
            src: `${spec.dir}/${name}`,
            title: titleFromSrc(name),
            kind: spec.kind
        })));
    }

    function entryFromItem(item, spec) {
        if (typeof item === 'string') {
            if (isAbsoluteUrl(item)) {
                if (!spec.ext.test(item.split('?')[0])) return null;
                return { src: item, title: titleFromSrc(item), kind: spec.kind };
            }
            const name = fileNameFromHref(item);
            if (!name || !spec.ext.test(name)) return null;
            return { src: `${spec.dir}/${name}`, title: titleFromSrc(name), kind: spec.kind };
        }

        if (item && typeof item === 'object') {
            const raw = item.src || item.url || item.name || item.file || '';
            if (!raw) return null;

            if (isAbsoluteUrl(raw)) {
                if (!spec.ext.test(raw.split('?')[0])) return null;
                return {
                    src: raw,
                    title: item.title || titleFromSrc(raw),
                    kind: spec.kind
                };
            }

            const name = fileNameFromHref(raw);
            if (!name || !spec.ext.test(name)) return null;
            return {
                src: `${spec.dir}/${name}`,
                title: item.title || titleFromSrc(name),
                kind: spec.kind
            };
        }

        return null;
    }

    /** Prefer local / relative clips when the same filename also exists on R2. */
    function mergeEntries(...lists) {
        const byKey = new Map();

        lists.flat().forEach((entry) => {
            if (!entry || !entry.src) return;
            const key = (fileNameFromHref(entry.src) || entry.src).toLowerCase();
            const existing = byKey.get(key);
            if (!existing) {
                byKey.set(key, entry);
                return;
            }
            if (!isAbsoluteUrl(entry.src) && isAbsoluteUrl(existing.src)) {
                byKey.set(key, entry);
            }
        });

        return sortEntries([...byKey.values()]);
    }

    function parseDirectoryHtml(html, spec) {
        const names = [];
        const hrefRe = /href\s*=\s*["']([^"']+)["']/gi;
        let match;
        while ((match = hrefRe.exec(html))) {
            const name = fileNameFromHref(match[1]);
            if (name) names.push(name);
        }
        return entriesFromNames(names, spec);
    }

    /**
     * npx serve and python's http.server both emit an HTML index for media/*.
     * GitHub Pages does not, so this returns null there and we fall through.
     */
    async function listFromDirectory(spec) {
        try {
            const response = await fetch(`${spec.dir}/`, { cache: 'no-store' });
            if (!response.ok) return null;

            const type = response.headers.get('content-type') || '';
            if (type.includes('application/json')) {
                const data = await response.json();
                return entriesFromJson(data, spec);
            }

            const entries = parseDirectoryHtml(await response.text(), spec);
            return entries.length ? entries : null;
        } catch (error) {
            return null;
        }
    }

    function entriesFromJson(data, spec) {
        if (!data) return null;

        let raw = null;
        if (Array.isArray(data)) {
            raw = data;
        } else {
            for (const key of spec.jsonKeys) {
                if (Array.isArray(data[key])) {
                    raw = data[key];
                    break;
                }
            }
        }

        if (!raw) return null;

        const entries = raw.map((item) => entryFromItem(item, spec)).filter(Boolean);
        return entries.length ? sortEntries(entries) : null;
    }

    async function listFromManifest(spec) {
        try {
            const response = await fetch(`${spec.dir}/index.json`, { cache: 'no-store' });
            if (!response.ok) return null;
            return entriesFromJson(await response.json(), spec);
        } catch (error) {
            return null;
        }
    }

    function githubPagesRepo() {
        const { hostname, pathname } = location;
        if (!hostname.endsWith('.github.io')) return null;

        const owner = hostname.slice(0, -'.github.io'.length);
        if (!owner) return null;

        const segments = pathname.split('/').filter(Boolean);
        const repo = segments[0] || `${owner}.github.io`;
        return { owner, repo };
    }

    /**
     * GitHub Pages has no directory listing, but the public Contents API can
     * enumerate whatever is committed under media/videos or media/images.
     */
    async function listFromGitHub(spec) {
        const repo = githubPagesRepo();
        if (!repo) return null;

        try {
            const response = await fetch(
                `https://api.github.com/repos/${repo.owner}/${repo.repo}/contents/${spec.dir}`,
                { headers: { Accept: 'application/vnd.github+json' } }
            );
            if (!response.ok) return null;

            const data = await response.json();
            if (!Array.isArray(data)) return null;

            const names = data
                .filter((item) => item && item.type === 'file' && item.name)
                .map((item) => item.name);

            const entries = entriesFromNames(names, spec);
            return entries.length ? entries : null;
        } catch (error) {
            return null;
        }
    }

    async function listKind(spec) {
        const [directory, manifest, github] = await Promise.all([
            listFromDirectory(spec),
            listFromManifest(spec),
            listFromGitHub(spec)
        ]);
        return mergeEntries(directory || [], manifest || [], github || []);
    }

    async function list() {
        const [videos, images] = await Promise.all([
            listKind(VIDEO_KIND),
            listKind(IMAGE_KIND)
        ]);
        return sortEntries([...(videos || []), ...(images || [])]);
    }

    async function probe() {
        return list();
    }

    function readCachedThumb(src) {
        try {
            const raw = localStorage.getItem(THUMB_CACHE_PREFIX + src);
            return raw ? JSON.parse(raw) : null;
        } catch (error) {
            return null;
        }
    }

    function writeCachedThumb(src, data) {
        try {
            localStorage.setItem(THUMB_CACHE_PREFIX + src, JSON.stringify(data));
        } catch (error) {
            // Quota exceeded or storage disabled; posters simply regenerate next visit.
        }
    }

    /**
     * Load a clip far enough to grab one representative frame plus its metadata.
     * Resolves with a null poster when the frame cannot be read (e.g. a
     * cross-origin source that taints the canvas).
     */
    function generateImageThumbnail(src) {
        return new Promise((resolve) => {
            const img = new Image();
            let settled = false;

            const finish = (data) => {
                if (settled) return;
                settled = true;
                resolve(data);
            };

            const timeout = setTimeout(() => {
                finish({ poster: null, duration: 0, width: 0, height: 0, kind: 'image' });
            }, 15000);

            img.crossOrigin = 'anonymous';
            img.onload = () => {
                clearTimeout(timeout);
                finish({
                    poster: src,
                    duration: 0,
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                    kind: 'image'
                });
            };
            img.onerror = () => {
                clearTimeout(timeout);
                finish({ poster: null, duration: 0, width: 0, height: 0, kind: 'image' });
            };
            img.src = src;
        });
    }

    function generateThumbnail(src, { useCache = true, kind } = {}) {
        if (kind === 'image' || isImageSrc(src)) {
            return generateImageThumbnail(src);
        }

        if (useCache) {
            const cached = readCachedThumb(src);
            if (cached) return Promise.resolve(cached);
        }

        return new Promise((resolve) => {
            const video = document.createElement('video');
            let settled = false;

            const finish = (data) => {
                if (settled) return;
                settled = true;
                clearTimeout(timeout);
                video.removeAttribute('src');
                video.load();
                resolve(data);
            };

            const timeout = setTimeout(() => finish({ poster: null, duration: 0, width: 0, height: 0 }), 15000);

            video.muted = true;
            video.playsInline = true;
            video.preload = 'auto';
            video.crossOrigin = 'anonymous';

            video.addEventListener('loadedmetadata', () => {
                // A quarter in avoids fade-ins and black leader frames.
                const target = Math.min(video.duration * 0.25, 3);
                video.currentTime = Number.isFinite(target) ? target : 0;
            });

            video.addEventListener('seeked', () => {
                const meta = {
                    poster: null,
                    duration: video.duration,
                    width: video.videoWidth,
                    height: video.videoHeight
                };

                try {
                    const canvas = document.createElement('canvas');
                    canvas.width = THUMB_WIDTH;
                    canvas.height = THUMB_HEIGHT;
                    const ctx = canvas.getContext('2d');

                    // Cover-fit the frame into the thumbnail so posters share a shape.
                    const scale = Math.max(THUMB_WIDTH / video.videoWidth, THUMB_HEIGHT / video.videoHeight);
                    const drawWidth = video.videoWidth * scale;
                    const drawHeight = video.videoHeight * scale;
                    ctx.drawImage(
                        video,
                        (THUMB_WIDTH - drawWidth) / 2,
                        (THUMB_HEIGHT - drawHeight) / 2,
                        drawWidth,
                        drawHeight
                    );

                    meta.poster = canvas.toDataURL('image/jpeg', 0.72);
                } catch (error) {
                    meta.poster = null;
                }

                if (useCache && meta.poster) writeCachedThumb(src, meta);
                finish(meta);
            }, { once: true });

            video.addEventListener('error', () => {
                finish({ poster: null, duration: 0, width: 0, height: 0 });
            });

            video.src = src;
            video.load();
        });
    }

    function formatDuration(seconds) {
        if (!Number.isFinite(seconds) || seconds <= 0) return '';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function titleFromSrc(src) {
        const file = src.split('/').pop().split('?')[0];
        return decodeURIComponent(file.replace(/\.[^.]+$/, ''));
    }

    return {
        MEDIA_DIR,
        VIDEOS_DIR,
        IMAGES_DIR,
        isImageSrc,
        list,
        probe,
        generateThumbnail,
        formatDuration,
        titleFromSrc
    };
})();
