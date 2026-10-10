"use strict";
/* Offline reading: the page, data and every picture already opened stay readable without signal.
   Pages and data go network-first (a new deploy shows at once); pictures and fonts go cache-first.
   Each build has its own cache name, so a deploy drops the previous build's pictures. */
const V = "__V__";
const CORE = `core-${V}`, PICS = `pics-${V}`, FONTS = "fonts-1";

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CORE).then((c) => c.addAll(["./", `app.js?v=${V}`, `style.css?v=${V}`, `data.json?v=${V}`])));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CORE && k !== PICS && k !== FONTS).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.hostname === "fonts.googleapis.com" || url.hostname === "fonts.gstatic.com") {
    e.respondWith(cacheFirst(req, FONTS));
  } else if (url.origin === self.location.origin && /\.(webp|png|jpg|svg)$/.test(url.pathname)) {
    e.respondWith(cacheFirst(req, PICS));
  } else if (url.origin === self.location.origin) {
    e.respondWith(networkFirst(req));
  }
});
async function cacheFirst(req, name) {
  const cache = await caches.open(name);
  const hit = await cache.match(req);
  if (hit) return hit;
  const res = await fetch(req);
  if (res.ok || res.type === "opaque") cache.put(req, res.clone());
  return res;
}
async function networkFirst(req) {
  const cache = await caches.open(CORE);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put(req, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(req, { ignoreSearch: req.mode === "navigate" });
    if (hit) return hit;
    throw err;
  }
}
