(function () {
  'use strict';
  // Dot-matrix band under the footer: slowly rolling "dunes" of dots, drawn only while visible.
  // The pointer and the walking robot push dots aside like footprints in sand; the marks fade back in a few seconds.
  var canvas = document.querySelector('[data-footer-field]');
  if (!canvas || !canvas.getContext) return;
  var ctx = canvas.getContext('2d');
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var GAP = 10;
  var DOT = 2;
  var FRAME_MS = 50;   // ~20 fps is enough for a slow drift
  var width = 0;
  var height = 0;
  var color = '#888';
  var visible = false;
  var raf = 0;
  var last = 0;
  var time = 0;
  var walker = canvas.parentNode.querySelector('.site-footer-walker');
  var TRAIL_MAX = 32;
  var TRAIL_MS = 2600;
  var MARK_GAP = 8;     // px between recorded footprints
  var trail = [];       // { x, y, at, radius, strength }
  var lastMark = {};

  function field(x, y, t) {
    var v = Math.sin(x * 0.045 + t * 0.6) + Math.sin(x * 0.013 - t * 0.35 + y * 0.05) + Math.sin((x + y) * 0.03 + t * 0.25);
    return v / 6 + 0.5;   // 0..1
  }

  function mark(source, x, y, radius, strength) {
    var prev = lastMark[source];
    if (prev && Math.abs(prev.x - x) + Math.abs(prev.y - y) < MARK_GAP) return;
    var point = { x: x, y: y, at: performance.now(), radius: radius, strength: strength };
    lastMark[source] = point;
    trail.push(point);
    if (trail.length > TRAIL_MAX) trail.shift();
  }

  // Offset of one dot from every live footprint: pushed outward, and thinned near the center.
  function displace(x, y, now, out) {
    out.dx = 0; out.dy = 0; out.fade = 1;
    for (var i = 0; i < trail.length; i++) {
      var p = trail[i];
      var life = 1 - (now - p.at) / TRAIL_MS;
      if (life <= 0) continue;
      var dx = x - p.x, dy = y - p.y;
      var d = Math.sqrt(dx * dx + dy * dy);
      if (d >= p.radius) continue;
      var push = (1 - d / p.radius) * life * p.strength;
      if (d > 0.01) { out.dx += dx / d * push; out.dy += dy / d * push; }
      out.fade = Math.min(out.fade, 1 - (1 - d / p.radius) * life * 0.7);
    }
  }

  var offset = { dx: 0, dy: 0, fade: 1 };

  function draw() {
    var now = performance.now();
    while (trail.length && now - trail[0].at > TRAIL_MS) trail.shift();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = color;
    for (var y = GAP / 2; y < height; y += GAP) {
      var depth = y / height;
      var threshold = 1.05 - depth * 0.75;   // sparse at the top, denser toward the bottom
      for (var x = GAP / 2; x < width; x += GAP) {
        var v = field(x, y, time);
        if (v <= threshold) continue;
        var alpha = Math.min(1, (v - threshold) * 3) * (0.25 + depth * 0.55);
        if (trail.length) {
          displace(x, y, now, offset);
          ctx.globalAlpha = alpha * offset.fade;
          ctx.fillRect(Math.round(x + offset.dx), Math.round(y + offset.dy), DOT, DOT);
        } else {
          ctx.globalAlpha = alpha;
          ctx.fillRect(Math.round(x), Math.round(y), DOT, DOT);
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth;
    height = canvas.clientHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function loop(now) {
    raf = 0;
    if (!visible || reduce.matches) return;
    if (now - last >= FRAME_MS) {
      last = now;
      time = now / 1000;
      trackWalker();
      draw();
    }
    raf = requestAnimationFrame(loop);
  }

  function trackWalker() {
    if (!walker) return;
    var box = canvas.getBoundingClientRect();
    var feet = walker.getBoundingClientRect();
    if (!feet.width) return;
    mark('walker', feet.left + feet.width / 2 - box.left, feet.bottom - box.top - 2, 18, 4);
  }

  function onPointer(event) {
    if (reduce.matches) return;
    var box = canvas.getBoundingClientRect();
    mark('pointer', event.clientX - box.left, event.clientY - box.top, 44, 9);
  }

  function update() {
    if (visible && !reduce.matches && !raf) raf = requestAnimationFrame(loop);
  }

  function readColor() {
    color = getComputedStyle(canvas).color;
    draw();
  }

  readColor();
  resize();
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener('resize', resize);
  new MutationObserver(readColor).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  reduce.addEventListener('change', update);
  canvas.parentNode.addEventListener('pointermove', onPointer);
  canvas.parentNode.addEventListener('pointerdown', onPointer);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      update();
    }).observe(canvas);
  }
})();
