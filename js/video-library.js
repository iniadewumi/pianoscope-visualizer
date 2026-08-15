/**
 * Video Library for Pianoscope Visualizer
 *
 * Discovers clips in videos/ at runtime (no hardcoded catalog), then generates
 * poster frames + metadata on demand.
 */

window.VideoLibrary = (function () {
    "use strict";

    const THUMB_CACHE_PREFIX = 'psv:thumb:v1:';
    const THUMB_WIDTH = 320;
    const THUMB_HEIGHT = 180;
    const VIDEOS_DIR = 'videos';
    const VIDEO_EXT = /\.(mp4|webm|mov|ogg|ogv|m4v)$/i;

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

    function entriesFromNames(names) {
        const unique = [...new Set(names.filter((name) => name && VIDEO_EXT.test(name)))];
        return sortEntries(unique.map((name) => ({
            src: `${VIDEOS_DIR}/${name}`,
            title: titleFromSrc(name)
        })));
    }

    function entryFromItem(item) {
        if (typeof item === 'string') {
            if (isAbsoluteUrl(item)) {
                if (!VIDEO_EXT.test(item.split('?')[0])) return null;
                return { src: item, title: titleFromSrc(item) };
            }
            const name = fileNameFromHref(item);
            if (!name || !VIDEO_EXT.test(name)) return null;
            return { src: `${VIDEOS_DIR}/${name}`, title: titleFromSrc(name) };
        }

        if (item && typeof item === 'object') {
            const raw = item.src || item.url || item.name || item.file || '';
            if (!raw) return null;

            if (isAbsoluteUrl(raw)) {
                if (!VIDEO_EXT.test(raw.split('?')[0])) return null;
                return {
                    src: raw,
                    title: item.title || titleFromSrc(raw)
                };
            }

            const name = fileNameFromHref(raw);
            if (!name || !VIDEO_EXT.test(name)) return null;
            return {
                src: `${VIDEOS_DIR}/${name}`,
                title: item.title || titleFromSrc(name)
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

    function parseDirectoryHtml(html) {
        const names = [];
        const hrefRe = /href\s*=\s*["']([^"']+)["']/gi;
        let match;
        while ((match = hrefRe.exec(html))) {
            const name = fileNameFromHref(match[1]);
            if (name) names.push(name);
        }
        return entriesFromNames(names);
    }

    /**
     * npx serve and python's http.server both emit an HTML index for videos/.
     * GitHub Pages does not, so this returns null there and we fall through.
     */
    async function listFromDirectory() {
        try {
            const response = await fetch(`${VIDEOS_DIR}/`, { cache: 'no-store' });
            if (!response.ok) return null;

            const type = response.headers.get('content-type') || '';
            if (type.includes('application/json')) {
                const data = await response.json();
                return entriesFromJson(data);
            }

            const entries = parseDirectoryHtml(await response.text());
            return entries.length ? entries : null;
        } catch (error) {
            return null;
        }
    }

    function entriesFromJson(data) {
        if (!data) return null;

        const raw = Array.isArray(data)
            ? data
            : Array.isArray(data.files)
                ? data.files
                : Array.isArray(data.videos)
                    ? data.videos
                    : null;

        if (!raw) return null;

        const entries = raw.map(entryFromItem).filter(Boolean);
        return entries.length ? sortEntries(entries) : null;
    }

    async function listFromManifest() {
        try {
            const response = await fetch(`${VIDEOS_DIR}/index.json`, { cache: 'no-store' });
            if (!response.ok) return null;
            return entriesFromJson(await response.json());
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
     * enumerate whatever is committed under videos/.
     */
    async function listFromGitHub() {
        const repo = githubPagesRepo();
        if (!repo) return null;

        try {
            const response = await fetch(
                `https://api.github.com/repos/${repo.owner}/${repo.repo}/contents/${VIDEOS_DIR}`,
                { headers: { Accept: 'application/vnd.github+json' } }
            );
            if (!response.ok) return null;

            const data = await response.json();
            if (!Array.isArray(data)) return null;

            const names = data
                .filter((item) => item && item.type === 'file' && item.name)
                .map((item) => item.name);

            const entries = entriesFromNames(names);
            return entries.length ? entries : null;
        } catch (error) {
            return null;
        }
    }

    async function list() {
        const [directory, manifest, github] = await Promise.all([
            listFromDirectory(),
            listFromManifest(),
            listFromGitHub()
        ]);
        return mergeEntries(directory || [], manifest || [], github || []);
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
    function generateThumbnail(src, { useCache = true } = {}) {
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
        list,
        probe,
        generateThumbnail,
        formatDuration,
        titleFromSrc
    };
})();
