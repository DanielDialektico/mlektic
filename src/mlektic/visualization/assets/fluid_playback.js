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
 * 3. MathJax output is memoised, and the equations of frames not shown yet are
 *    typeset one at a time while the page is idle, so a first visit replays
 *    MathJax's own output instead of waiting for it.
 *
 * The slider and Play/Pause stay Plotly's own controls: `execute: false` (a
 * documented Plotly attribute) hands their commands to this controller while
 * Plotly keeps drawing and syncing them. Anything this controller does not
 * recognise is left to Plotly unchanged.
 */
(function () {
  "use strict";
  var VERSION = 4;
  if (window.__mlekticFluid && window.__mlekticFluid.version >= VERSION) return;

  // Traces a redraw-free animate cannot move: WebGL traces, and raster traces
  // (heatmap, contour) whose image is only rebuilt by a replot.
  var GL = {
    surface: 1, scatter3d: 1, mesh3d: 1, volume: 1, isosurface: 1, scattergl: 1, cone: 1, streamtube: 1,
    heatmap: 1, contour: 1, histogram2d: 1, histogram2dcontour: 1, image: 1,
  };
  var CAMERA_ONLY = /^(scene\d*(\.|$)|dragmode$|hovermode$)/;
  // Typesetting ahead waits until the student has left the figure alone for
  // this long, so it never queues MathJax work in front of a visible step.
  var QUIET_MS = 1000;
  var INLINE_MATH = [["$", "$"], ["\\(", "\\)"]];
  var FIND_TEX = /([^$]*)([$]+[^$]*[$]+)([^$]*)/;
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

  // The key Plotly's initiate step produces: integer font size and the TeX
  // source with angle brackets escaped (svg_text_utils texToSVG).
  function texJob(text, size) {
    var m = typeof text === "string" && text.match(FIND_TEX);
    if (!m || !size) return null;
    var tex = m[2].replace(/(<|&lt;|&#60;)/g, "\\lt ").replace(/(>|&gt;|&#62;)/g, "\\gt ");
    var px = parseInt(size, 10);
    return { size: px, tex: tex, key: px + "px|" + tex };
  }
  // Typeset one equation through MathJax's queue with Plotly's settings and
  // keep the output under Plotly's key.
  var aheadCount = 0;
  function typesetAhead(job, done) {
    var MJ = window.MathJax, H = MJ.Hub, cache = MJ.__mlekticCache;
    if (cache.has(job.key)) { done(); return; }
    var div = document.createElement("div"), saved, finished = false;
    div.id = "mlektic-math-ahead-" + (++aheadCount);
    // Off-screen, not merely hidden: MathJax may mark its own output visible,
    // and a page that shares one document (VS Code) would then show it.
    div.setAttribute("aria-hidden", "true");
    div.style.cssText = "position:fixed;left:-10000px;top:0;visibility:hidden;pointer-events:none;";
    div.style.fontSize = job.size + "px";
    div.textContent = job.tex;
    document.body.appendChild(div);
    // Whatever MathJax does, the scratch node leaves and the queue moves on.
    function finish() {
      if (finished) return;
      finished = true;
      if (div.parentNode) div.parentNode.removeChild(div);
      done();
    }
    setTimeout(finish, 8000);
    H.Queue(function () {
      saved = { delay: H.processSectionDelay, config: {
        messageStyle: H.config.messageStyle,
        tex2jax: { inlineMath: H.config.tex2jax && H.config.tex2jax.inlineMath },
        displayAlign: H.config.displayAlign,
      } };
      if (H.processSectionDelay !== undefined) H.processSectionDelay = 0;
      return H.Config({ messageStyle: "none", tex2jax: { inlineMath: INLINE_MATH }, displayAlign: "left" });
    }, ["Typeset", H, div], function () {
      if (div.querySelector(".MathJax_SVG") && !cache.has(job.key)) cache.set(job.key, div.innerHTML);
      if (div.parentNode) div.parentNode.removeChild(div);
    }, function () {
      if (saved && saved.delay !== undefined) H.processSectionDelay = saved.delay;
      return saved ? H.Config(saved.config) : undefined;
    }, finish);
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
    var generation = 0;
    var size = sizeKey();
    var lastInput = -Infinity;
    function touched(e) {
      if (e.type !== "pointermove" || e.buttons) lastInput = performance.now();
    }
    ["pointerdown", "pointermove", "wheel", "keydown", "touchstart"].forEach(function (type) {
      gd.addEventListener(type, touched, { capture: true, passive: true });
    });

    function sizeKey() { return gd._fullLayout.width + "x" + gd._fullLayout.height; }
    // Cached nodes hold pixel positions, so anything that can move an
    // annotation (resize, zoom, legend toggles) discards them.
    function invalidate() {
      cache.clear();
      generation++;
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
    // A frame whose Plotly or MathJax promise never settles (a shared webview
    // that throttles rendering, a second Plotly.js copy) must not end playback
    // or lock the slider: past the limit the controller moves on regardless.
    function bounded(name, ms, redraw, limit) {
      return new Promise(function (resolve) {
        var settled = false;
        function settle() { if (!settled) { settled = true; resolve(); } }
        setTimeout(settle, limit);
        var result;
        try { result = apply(name, ms, redraw); } catch (err) { settle(); throw err; }
        Promise.resolve(result).then(settle, settle);
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
        var next = function () {
          // Slider moves made while the listeners were gone were never heard.
          if (rebind() && pendingName === null) {
            var target = sliderTarget();
            if (target !== null && target !== n) pendingName = target;
          }
          if (pendingName !== null) drain(); else busy = false;
        };
        try {
          bounded(n, 0, true, 4000).then(next);
        } catch (err) {
          // A synchronous failure must not leave the controller busy forever.
          busy = false;
          throw err;
        }
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
          rebind();
          if (my !== token) return;
          setTimeout(tick, Math.max(0, play.frame - (performance.now() - t0)));
        };
        try {
          bounded(order[i++], Math.min(play.transition, play.frame), play.redraw, Math.max(4000, 4 * play.frame)).then(done);
        } catch (err) {
          playing = false;
          throw err;
        }
      })();
    }

    function onSlider(e) {
      if (!e || !e.interaction || !e.step) return;
      var name = frameTarget(gd, e.step.args);
      if (name !== null && e.step.execute === false) { stop(); request(name); }
    }
    function onButton(e) {
      var b = e && e.button;
      if (!b || b.method !== "animate" || b.execute !== false) return;
      if (isPlay(b.args)) startPlayback();
      else if (isPause(b.args)) stop();
    }
    function onRelayout(e) {
      var keys = Object.keys(e || {});
      if (keys.length && keys.every(function (k) { return CAMERA_ONLY.test(k); })) return;
      invalidate();
    }
    if (handOverControls(gd)) {
      gd.on("plotly_sliderchange", onSlider);
      gd.on("plotly_buttonclicked", onButton);
    }
    gd.on("plotly_relayout", onRelayout);
    gd.on("plotly_restyle", invalidate);
    var emitters = { pub: gd._ev, internal: gd._internalEv };
    // A notebook page can hold several copies of Plotly.js, one per output, and
    // window.Plotly is the last one loaded. When it redraws a figure that
    // another copy created (a 3-D frame does), it gives the figure fresh event
    // emitters and drops every listener: this controller's, so the slider and
    // Play stop responding, and Plotly's own, so the slider stops following
    // the animation. Carry them over whenever the emitters have changed.
    function carry(from, to, skipKnownEvents) {
      if (!from || !to || from === to || !from._events) return;
      Object.keys(from._events).forEach(function (name) {
        // Plotly may already have re-registered its own observers on the new one.
        if (skipKnownEvents && to.listeners(name).length) return;
        from.listeners(name).forEach(function (fn) {
          if (to.listeners(name).indexOf(fn) < 0) to.on(name, fn);
        });
      });
    }
    function rebind() {
      if (gd._ev === emitters.pub && gd._internalEv === emitters.internal) return false;
      carry(emitters.pub, gd._ev, false);
      carry(emitters.internal, gd._internalEv, true);
      emitters = { pub: gd._ev, internal: gd._internalEv };
      return true;
    }
    function sliderTarget() {
      var found = null;
      (gd._fullLayout.sliders || []).some(function (s) {
        var step = s.steps && s.steps[s.active];
        found = step && step.method === "animate" && step.execute === false ? frameTarget(gd, step.args) : null;
        return found !== null;
      });
      return found;
    }

    // Every distinct equation in the frames, in frame order, under the font
    // size Plotly will use for it.
    function texJobs() {
      var fl = gd._fullLayout, base = fl.annotations || [], seen = {}, jobs = [];
      frames.forEach(function (f) {
        if (!hasAnnotations(f)) return;
        f.layout.annotations.forEach(function (a, i) {
          var size = (a.font && a.font.size) || (base[i] && base[i].font && base[i].font.size) || fl.font.size;
          var job = texJob(a.text, size);
          if (job && !seen[job.key]) { seen[job.key] = 1; jobs.push(job); }
        });
      });
      return jobs;
    }
    function prewarm() {
      if (gd._context.typesetMath === false || !window.MathJax) return;
      var jobs = texJobs(), i = 0, waits = 0;
      (function next() {
        if (i >= jobs.length || !document.body.contains(gd)) return;
        if (!window.MathJax.__mlekticCache) {
          if (++waits < 50) setTimeout(next, 200);
          return;
        }
        // Never queue MathJax work in front of something the student sees.
        if (playing || busy || stockAnimating() || performance.now() - lastInput < QUIET_MS) {
          setTimeout(next, 200);
          return;
        }
        typesetAhead(jobs[i++], function () { idle(next); });
      })();
    }

    var api = { apply: apply, request: request, prewarm: prewarm, cache: cache, play: startPlayback, stop: stop };
    if (options.prewarm !== false) idle(prewarm);
    if (options.autoplay) startPlayback();
    return api;
  }

  // ── typed arrays inside frames ──────────────────────────────────────────
  // Plotly decodes base64 typed arrays ({dtype, bdata, shape}) in the figure's
  // data, but not in its frames. Decode them once so every playback path
  // (animate, update, replot) receives real arrays.
  var TYPED = { u1: Uint8Array, i1: Int8Array, u2: Uint16Array, i2: Int16Array,
    u4: Uint32Array, i4: Int32Array, f4: Float32Array, f8: Float64Array };
  function decodeSpec(v) {
    if (!v || typeof v !== "object" || ArrayBuffer.isView(v)) return v;
    if (typeof v.bdata === "string" && TYPED[v.dtype]) {
      var raw = atob(v.bdata), bytes = new Uint8Array(raw.length);
      for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      var flat = new TYPED[v.dtype](bytes.buffer);
      var shape = v.shape === undefined ? [flat.length] : String(v.shape).split(",").map(Number);
      if (shape.length !== 2) return flat;
      var rows = [];
      for (var r = 0; r < shape[0]; r++) rows.push(Array.prototype.slice.call(flat, r * shape[1], (r + 1) * shape[1]));
      return rows;
    }
    if (Array.isArray(v)) {
      for (var j = 0; j < v.length; j++) v[j] = decodeSpec(v[j]);
      return v;
    }
    Object.keys(v).forEach(function (k) { v[k] = decodeSpec(v[k]); });
    return v;
  }
  function decodeFrames(gd) {
    (gd._transitionData._frames || []).forEach(function (f) {
      if (f && f.data) f.data = f.data.map(decodeSpec);
    });
  }

  function mount(gd, options) {
    if (!gd || !window.Plotly || gd.__mlekticFluid) return gd && gd.__mlekticFluid;
    var trans = gd._transitionData;
    if (!trans || !trans._frames || !trans._frames.length || !gd._fullLayout) return null;
    decodeFrames(gd);
    gd.__mlekticFluid = attach(gd, options);
    return gd.__mlekticFluid;
  }

  window.__mlekticFluid = { version: VERSION, mount: mount, attach: attach };
})();
