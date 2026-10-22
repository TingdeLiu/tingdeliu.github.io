(function () {
  'use strict';
  // Dot-matrix band under the footer: slowly rolling "dunes" of dots, drawn only while visible.
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

  function field(x, y, t) {
    var v = Math.sin(x * 0.045 + t * 0.6) + Math.sin(x * 0.013 - t * 0.35 + y * 0.05) + Math.sin((x + y) * 0.03 + t * 0.25);
    return v / 6 + 0.5;   // 0..1
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = color;
    for (var y = GAP / 2; y < height; y += GAP) {
      var depth = y / height;
      var threshold = 1.05 - depth * 0.75;   // sparse at the top, denser toward the bottom
      for (var x = GAP / 2; x < width; x += GAP) {
        var v = field(x, y, time);
        if (v <= threshold) continue;
        ctx.globalAlpha = Math.min(1, (v - threshold) * 3) * (0.25 + depth * 0.55);
        ctx.fillRect(Math.round(x), Math.round(y), DOT, DOT);
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
      draw();
    }
    raf = requestAnimationFrame(loop);
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
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      update();
    }).observe(canvas);
  }
})();
