/* Mlektic fluid playback.
 *
 * Plotly redraws the whole figure, and MathJax re-typesets every equation,
 * whenever an animation frame changes layout annotations. This controller keeps
 * the figure exactly as authored and only changes how frames are applied:
 *
 * 1. The first time a frame is shown, Plotly draws it normally (redraw: true)
 *    and the rendered <g class="annotation"> nodes are kept.
 * 2. Later visits move trace data with a redraw-free animate and put those exact
 *    nodes back, so equations are pixel-identical and MathJax does not run.
 * 3. A hidden twin plot pre-renders the remaining frames while the page is idle.
 * 4. MathJax output is memoised, so equations that repeat across frames are
 *    typeset once.
 *
 * The slider and Play/Pause stay Plotly's own controls: `execute: false` (a
 * documented Plotly attribute) hands their commands to this controller while
 * Plotly keeps drawing and syncing them. Anything this controller does not
 * recognise is left to Plotly unchanged.
 */
(function () {
  "use strict";
  var VERSION = 1;
  if (window.__mlekticFluid && window.__mlekticFluid.version >= VERSION) return;

  var GL = { surface: 1, scatter3d: 1, mesh3d: 1, volume: 1, isosurface: 1, scattergl: 1, cone: 1, streamtube: 1 };
  var CAMERA_ONLY = /^(scene\d*(\.|$)|dragmode$|hovermode$)/;
  function idle(cb) {
    return window.requestIdleCallback ? window.requestIdleCallback(cb, { timeout: 1000 }) : setTimeout(cb, 1);
  }

  // ── MathJax 2 output cache ──────────────────────────────────────────────
  // Plotly queues six callbacks per equation: setConfig, setRenderer,
  // initiate (creates the math-output-* div and asks MathJax to typeset it),
  // finalize, resetRenderer, resetConfig. Initiate runs first, synchronously,
  // only to learn the font size and TeX source. A cache hit replays MathJax's
  // own output and finalizes at once (the config callbacks would cancel out).
  // A miss queues Plotly's sequence in its usual order around the typeset.
  function installMathCache() {
    var MJ = window.MathJax;
    if (!MJ || !MJ.Hub || !MJ.Hub.Queue || !MJ.Hub.Typeset) return false;
    if (MJ.__mlekticCache) return true;
    var H = MJ.Hub, cache = new Map(), peek = null;
    var origQueue = H.Queue, origTypeset = H.Typeset;
    MJ.__mlekticCache = cache;
    function key(el) { return el.style.fontSize + "|" + el.textContent; }

    H.Typeset = function (el) {
      if (peek && el && el.id && el.id.indexOf("math-output-") === 0) {
        var hit = cache.get(key(el));
        if (hit !== undefined) { el.innerHTML = hit; peek.hit = true; return null; }
        peek.el = el;
        return null;
      }
      return origTypeset.apply(H, arguments);
    };
    H.Queue = function () {
      var a = arguments;
      var plotlyShape = a.length === 6 && Array.prototype.every.call(a, function (f) { return typeof f === "function"; });
      if (!plotlyShape) return origQueue.apply(H, a);
      var p = peek = {};
      try { a[2](); } finally { peek = null; }
      if (p.hit) { a[3](); return null; }
      if (!p.el) return origQueue.call(H, a[0], a[1], a[3], a[4], a[5]);
      var el = p.el, k = key(el);
      return origQueue.call(H, a[0], a[1], ["Typeset", H, el], function () {
        if (el.querySelector(".MathJax_SVG")) cache.set(k, el.innerHTML);
      }, a[3], a[4], a[5]);
    };
    return true;
  }
  if (!installMathCache()) {
    var tries = 0, timer = setInterval(function () {
      if (installMathCache() || ++tries > 200) clearInterval(timer);
    }, 25);
  }

  // ── frame helpers ───────────────────────────────────────────────────────
  function opts(redraw, ms) {
    return { mode: "immediate", frame: { duration: 0, redraw: redraw }, transition: { duration: ms || 0, easing: "linear" } };
  }
  function traceIndex(f, j) { return f.traces ? f.traces[j] : j; }
  function isGL(gd, f, j) {
    var t = gd.data[traceIndex(f, j)];
    return !!GL[(t && t.type) || (f.data[j] && f.data[j].type) || "scatter"];
  }
  function hasGL(gd, f) {
    return (f.data || []).some(function (_, j) { return isGL(gd, f, j); });
  }
  function hasAnnotations(f) { return !!(f.layout && f.layout.annotations); }
  function eligible(gd, f) {
    var lay = f.layout || {};
    var allowed = hasGL(gd, f) ? { annotations: 1, scene: 1 } : { annotations: 1 };
    if (Object.keys(lay).some(function (k) { return !allowed[k]; })) return false;
    if (gd._context.editable) return false;
    // Cached nodes carry no event handlers and no per-size clip paths.
    return !(gd._fullLayout.annotations || []).some(function (a) {
      return a.width || a.height || a.captureevents || a.hovertext;
    });
  }
  function flatten(v, prefix, out) {
    Object.keys(v).forEach(function (k) {
      var x = v[k], path = prefix ? prefix + "." + k : k;
      if (x && typeof x === "object" && !Array.isArray(x) && !ArrayBuffer.isView(x) && !("bdata" in x)) flatten(x, path, out);
      else out[path] = x;
    });
    return out;
  }
  function clone(x) {
    if (Array.isArray(x)) return x.map(clone);
    if (ArrayBuffer.isView(x)) return x.slice();
    if (x && typeof x === "object") {
      var out = {};
      Object.keys(x).forEach(function (k) { out[k] = clone(x[k]); });
      return out;
    }
    return x;
  }
  // WebGL traces cannot move without a replot. Replot once with MathJax off so
  // the equations are not re-typeset; the cached nodes are swapped in after.
  function glUpdate(gd, f) {
    var flat = (f.data || []).map(function (d) { return flatten(d, "", {}); });
    var keys = {};
    flat.forEach(function (d) { Object.keys(d).forEach(function (k) { if (k !== "type") keys[k] = 1; }); });
    var traceUpdate = {};
    Object.keys(keys).forEach(function (k) { traceUpdate[k] = flat.map(function (d) { return d[k]; }); });
    var layoutUpdate = f.layout && f.layout.scene ? flatten(f.layout.scene, "scene", {}) : {};
    var typeset = gd._context.typesetMath;
    gd._context.typesetMath = false;
    try {
      return Plotly.update(gd, traceUpdate, layoutUpdate, f.traces);
    } finally {
      gd._context.typesetMath = typeset;
    }
  }
  function harvest(gd, f) {
    var out = {}, layer = gd._fullLayout._infolayer.node();
    f.layout.annotations.forEach(function (_, i) {
      var n = layer.querySelector('.annotation[data-index="' + i + '"]');
      if (n) out[i] = n.outerHTML;
    });
    return out;
  }
  function swap(gd, html) {
    var layer = gd._fullLayout._infolayer.node();
    var tmp = document.createElementNS("http://www.w3.org/2000/svg", "g");
    Object.keys(html).forEach(function (i) {
      var cur = layer.querySelector('.annotation[data-index="' + i + '"]');
      tmp.innerHTML = html[i];
      var next = tmp.firstChild;
      if (cur) layer.replaceChild(next, cur); else layer.appendChild(next);
    });
  }
  function mergeInto(target, src) {
    Object.keys(src).forEach(function (k) {
      var v = src[k];
      if (v && typeof v === "object" && !Array.isArray(v) && target[k] && typeof target[k] === "object") mergeInto(target[k], v);
      else target[k] = v;
    });
  }
  // Keep Plotly's state equal to the frame so any later full redraw (resize,
  // relayout, export) reproduces the same picture.
  function syncLayout(gd, f) {
    if (!hasAnnotations(f)) return;
    f.layout.annotations.forEach(function (a, i) {
      if (gd.layout.annotations && gd.layout.annotations[i]) mergeInto(gd.layout.annotations[i], a);
      if (gd._fullLayout.annotations && gd._fullLayout.annotations[i]) mergeInto(gd._fullLayout.annotations[i], a);
    });
  }
  // Plotly's slider follows _currentFrame through this event.
  function announce(gd, name, f, frameOpts, transitionOpts) {
    gd._fullLayout._currentFrame = name;
    gd.emit("plotly_animatingframe", { name: name, frame: f, animation: { frame: frameOpts, transition: transitionOpts } });
  }

  // ── controls ────────────────────────────────────────────────────────────
  function frameTarget(gd, args) {
    var t = args && args[0];
    if (!Array.isArray(t) || t.length !== 1 || t[0] === null || t[0] === undefined) return null;
    var name = String(t[0]);
    return gd._transitionData._frameHash[name] ? name : null;
  }
  function isPlay(args) { return !!args && args[0] === null; }
  function isPause(args) { return !!args && Array.isArray(args[0]) && args[0].length === 1 && args[0][0] === null; }
  function handOverControls(gd) {
    var touched = 0;
    [gd.layout, gd._fullLayout].forEach(function (lay) {
      (lay.sliders || []).forEach(function (s) {
        (s.steps || []).forEach(function (st) {
          if (st.method === "animate" && frameTarget(gd, st.args)) { st.execute = false; touched++; }
        });
      });
      (lay.updatemenus || []).forEach(function (m) {
        (m.buttons || []).forEach(function (b) {
          if (b.method === "animate" && (isPlay(b.args) || isPause(b.args))) { b.execute = false; touched++; }
        });
      });
    });
    return touched > 0;
  }
  function playSettings(gd) {
    var out = { frame: 500, transition: 0, redraw: true, fromcurrent: false, reverse: false };
    (gd.layout.updatemenus || []).forEach(function (m) {
      (m.buttons || []).forEach(function (b) {
        var o = b.method === "animate" && isPlay(b.args) && (b.args[1] || {});
        if (!o) return;
        var frame = o.frame || {}, transition = o.transition || {};
        if (frame.duration !== undefined) out.frame = frame.duration;
        if (frame.redraw !== undefined) out.redraw = !!frame.redraw;
        out.transition = transition.duration || 0;
        out.fromcurrent = !!o.fromcurrent;
        out.reverse = o.direction === "reverse";
      });
    });
    return out;
  }

  function attach(gd, options) {
    options = options || {};
    var cache = new Map();
    var frames = gd._transitionData._frames;
    var names = frames.map(function (f) { return String(f.name); });
    var play = playSettings(gd);
    var token = 0, busy = false, pendingName = null, playing = false;
    var generation = 0, warmTimer = null, warming = false;
    var size = sizeKey();

    function sizeKey() { return gd._fullLayout.width + "x" + gd._fullLayout.height; }
    // Cached nodes hold pixel positions, so anything that can move an
    // annotation (resize, zoom, legend toggles) discards them.
    function invalidate() {
      cache.clear();
      generation++;
      if (options.prewarm === false) return;
      clearTimeout(warmTimer);
      warmTimer = setTimeout(function () { idle(prewarm); }, 400);
    }
    function checkSize() {
      var k = sizeKey();
      if (k !== size) { size = k; invalidate(); }
    }
    function stockAnimating() {
      var q = gd._transitionData._frameQueue;
      return !!(q && q.length);
    }

    function apply(name, ms, redraw) {
      var f = gd._transitionData._frameHash[name];
      if (!f) return Promise.resolve();
      checkSize();
      if (!eligible(gd, f)) return Plotly.animate(gd, [name], opts(redraw !== false, ms));
      var snap = cache.get(name);
      if (!snap && !hasAnnotations(f) && !hasGL(gd, f)) snap = {};  // data-only frame: nothing to typeset
      if (!snap) {
        var gen = generation;
        return Plotly.animate(gd, [name], opts(true, 0)).then(function () {
          if (gen === generation) cache.set(name, harvest(gd, f));
        });
      }
      if (hasGL(gd, f)) {
        syncLayout(gd, f);
        return glUpdate(gd, f).then(function () {
          swap(gd, snap);
          announce(gd, name, f, { duration: 0, redraw: true }, { duration: 0 });
        });
      }
      // Swap equations first so they change in the same paint as the data starts moving.
      swap(gd, snap);
      syncLayout(gd, f);
      var frameOpts = { duration: 0, redraw: false }, transitionOpts = { duration: ms || 0 };
      announce(gd, name, f, frameOpts, transitionOpts);
      if (!(f.data && f.data.length)) return Promise.resolve();
      // An anonymous data frame clears _currentFrame, so restore the name afterwards.
      return Plotly.animate(gd, { data: f.data, traces: f.traces }, opts(false, ms)).then(function () {
        announce(gd, name, f, frameOpts, transitionOpts);
      });
    }
    // Coalesce slider drags: only the newest requested frame is drawn.
    function request(name) {
      pendingName = name;
      if (busy) return;
      busy = true;
      (function drain() {
        var n = pendingName;
        pendingName = null;
        var next = function () { if (pendingName !== null) drain(); else busy = false; };
        apply(n, 0, true).then(next, next);
      })();
    }
    function stop() { playing = false; token++; }
    // Same frame order and start rules as Plotly.animate(gd, null, options).
    function startPlayback() {
      stop();
      var my = ++token;
      playing = true;
      var order = play.reverse ? names.slice().reverse() : names;
      var i = 0;
      if (play.fromcurrent) {
        var idx = order.indexOf(String(gd._fullLayout._currentFrame));
        if (idx > 0 && idx < order.length - 1) i = idx + 1;
      }
      (function tick() {
        if (my !== token) return;
        if (i >= order.length) { playing = false; return; }
        var t0 = performance.now();
        var done = function () {
          if (my !== token) return;
          setTimeout(tick, Math.max(0, play.frame - (performance.now() - t0)));
        };
        apply(order[i++], Math.min(play.transition, play.frame), play.redraw).then(done, done);
      })();
    }

    if (handOverControls(gd)) {
      gd.on("plotly_sliderchange", function (e) {
        if (!e || !e.interaction || !e.step) return;
        var name = frameTarget(gd, e.step.args);
        if (name !== null && e.step.execute === false) { stop(); request(name); }
      });
      gd.on("plotly_buttonclicked", function (e) {
        var b = e && e.button;
        if (!b || b.method !== "animate" || b.execute !== false) return;
        if (isPlay(b.args)) startPlayback();
        else if (isPause(b.args)) stop();
      });
    }
    gd.on("plotly_relayout", function (e) {
      var keys = Object.keys(e || {});
      if (keys.length && keys.every(function (k) { return CAMERA_ONLY.test(k); })) return;
      invalidate();
    });
    gd.on("plotly_restyle", invalidate);

    function prewarm() {
      if (warming) { clearTimeout(warmTimer); warmTimer = setTimeout(function () { idle(prewarm); }, 400); return; }
      var gen = generation;
      var todo = frames.filter(function (f) { return hasAnnotations(f) && eligible(gd, f) && !cache.has(String(f.name)); });
      if (!todo.length || !document.body.contains(gd)) return;
      warming = true;
      var twin = document.createElement("div");
      twin.setAttribute("aria-hidden", "true");
      twin.style.cssText = "position:absolute;left:-20000px;top:0;visibility:hidden;pointer-events:none;width:" +
        gd._fullLayout.width + "px;height:" + gd._fullLayout.height + "px";
      document.body.appendChild(twin);
      var mainUid = gd._fullLayout._uid;
      function finish() {
        try { Plotly.purge(twin); } catch (err) { /* already gone */ }
        twin.remove();
        warming = false;
      }
      // The twin keeps every trace so legend and margins, and therefore
      // annotation positions, match the main plot. Its frames carry the
      // annotations plus 2D trace data (which can move axis-referenced
      // annotations); WebGL data does not affect them and is skipped.
      function twinFrame(f) {
        var out = { name: f.name, layout: { annotations: f.layout.annotations } };
        var data = [], traces = [];
        (f.data || []).forEach(function (d, j) {
          if (!isGL(gd, f, j)) { data.push(d); traces.push(traceIndex(f, j)); }
        });
        if (data.length) { out.data = data; out.traces = traces; }
        return out;
      }
      var context = Object.assign({}, gd._context, { responsive: false });
      Plotly.newPlot(twin, clone(gd.data), clone(gd.layout), context)
        .then(function () { return Plotly.addFrames(twin, todo.map(twinFrame)); })
        .then(function () {
          var twinUid = twin._fullLayout._uid, i = 0;
          (function next() {
            if (gen !== generation || i >= todo.length || !document.body.contains(gd)) { finish(); return; }
            // Never compete with the student for the main thread.
            if (playing || busy || stockAnimating()) { setTimeout(next, 120); return; }
            var f = todo[i++], name = String(f.name);
            if (cache.has(name)) { idle(next); return; }
            Plotly.animate(twin, [f.name], opts(true, 0)).then(function () {
              if (gen === generation && !cache.has(name)) {
                var snap = harvest(twin, f);
                Object.keys(snap).forEach(function (k) { snap[k] = snap[k].split(twinUid).join(mainUid); });
                cache.set(name, snap);
              }
              idle(next);
            }, finish);
          })();
        })
        .catch(finish);
    }

    var api = { apply: apply, request: request, prewarm: prewarm, cache: cache, play: startPlayback, stop: stop };
    if (options.prewarm !== false) idle(prewarm);
    if (options.autoplay) startPlayback();
    return api;
  }

  function mount(gd, options) {
    if (!gd || !window.Plotly || gd.__mlekticFluid) return gd && gd.__mlekticFluid;
    var trans = gd._transitionData;
    if (!trans || !trans._frames || !trans._frames.length || !gd._fullLayout) return null;
    gd.__mlekticFluid = attach(gd, options);
    return gd.__mlekticFluid;
  }

  window.__mlekticFluid = { version: VERSION, mount: mount, attach: attach };
})();
