(function () {
  'use strict';
  // 404 page as a failed navigation episode: the missing path seeds a maze, the robot explores until it hits a dead end.
  var root = document.querySelector('[data-nf]');
  if (!root) return;
  var canvas = root.querySelector('[data-nf-map]');
  var log = root.querySelector('[data-nf-log]');
  var metrics = root.querySelector('[data-nf-metrics]');
  var ndtw = root.querySelector('[data-nf-ndtw]');
  var ctx = canvas.getContext && canvas.getContext('2d');
  if (!ctx) return;
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var COLS = 15;
  var ROWS = 7;
  var STEP_MS = 170;
  var WALK_MS = 5000;  // long walks speed up so an episode never drags
  var path = decodeURIComponent(window.location.pathname) || '/';
  var english = /^\/en(\/|$)/.test(path);
  var episode = 0;
  var maze = null;
  var walk = [];
  var progress = 0;   // cells walked, fractional while moving
  var bump = 0;       // collision shake, 1 → 0
  var start = 0;
  var stepMs = STEP_MS;
  var raf = 0;
  var width = 0;
  var cell = 0;
  var colors = {};

  if (english) {
    document.documentElement.lang = 'en';
    document.querySelectorAll('[data-nf-copy]').forEach(function (el) { el.hidden = el.getAttribute('data-nf-copy') !== 'en'; });
  }

  function hash(text) {
    var h = 2166136261;
    for (var i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }

  function rng(seed) {
    return function () {
      seed = (seed + 0x6d2b79f5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function shuffle(list, random) {
    for (var i = list.length - 1; i > 0; i--) {
      var j = Math.floor(random() * (i + 1));
      var tmp = list[i]; list[i] = list[j]; list[j] = tmp;
    }
    return list;
  }

  var DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

  // Recursive backtracker; open[y][x] holds a bit per direction index.
  function buildMaze(random) {
    var open = [];
    for (var y = 0; y < ROWS; y++) { open.push(new Array(COLS).fill(0)); }
    var seen = open.map(function (row) { return row.map(function () { return false; }); });
    var stack = [[0, ROWS - 1]];
    seen[ROWS - 1][0] = true;
    while (stack.length) {
      var top = stack[stack.length - 1];
      var next = shuffle([0, 1, 2, 3], random).filter(function (d) {
        var nx = top[0] + DIRS[d][0], ny = top[1] + DIRS[d][1];
        return nx >= 0 && ny >= 0 && nx < COLS && ny < ROWS && !seen[ny][nx];
      });
      if (!next.length) { stack.pop(); continue; }
      var d = next[0];
      var nx = top[0] + DIRS[d][0], ny = top[1] + DIRS[d][1];
      open[top[1]][top[0]] |= 1 << d;
      open[ny][nx] |= 1 << (d ^ 1);
      seen[ny][nx] = true;
      stack.push([nx, ny]);
    }
    return open;
  }

  // Greedy exploration without backtracking: it always ends in a dead end, which is the point.
  function explore(random) {
    var visited = {};
    var at = [0, ROWS - 1];
    var cells = [at];
    visited[at] = true;
    for (;;) {
      var options = shuffle([0, 1, 2, 3], random).filter(function (d) {
        var nx = at[0] + DIRS[d][0], ny = at[1] + DIRS[d][1];
        return (maze[at[1]][at[0]] & (1 << d)) && !visited[[nx, ny]];
      });
      if (!options.length) return cells;
      at = [at[0] + DIRS[options[0]][0], at[1] + DIRS[options[0]][1]];
      visited[at] = true;
      cells.push(at);
    }
  }

  function readColors() {
    var style = getComputedStyle(canvas);
    ['ink', 'muted', 'rule', 'accent', 'paper'].forEach(function (name) {
      colors[name] = style.getPropertyValue('--' + name).trim() || '#888';
    });
  }

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth;
    cell = width / COLS;
    var height = Math.round(cell * ROWS);
    canvas.style.height = height + 'px';
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function center(c) { return [(c[0] + 0.5) * cell, (c[1] + 0.5) * cell]; }

  function draw() {
    if (!maze || !cell) return;
    ctx.clearRect(0, 0, width, ROWS * cell);
    var dot = Math.max(1.5, cell * 0.08);
    ctx.fillStyle = colors.muted;
    ctx.globalAlpha = 0.35;
    for (var y = 0; y < ROWS; y++) {
      for (var x = 0; x < COLS; x++) {
        ctx.fillRect((x + 0.5) * cell - dot / 2, (y + 0.5) * cell - dot / 2, dot, dot);
      }
    }
    ctx.globalAlpha = 1;

    // Walls: draw the east and south side of every cell, plus the outer west and north edges.
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 2;
    ctx.lineCap = 'square';
    ctx.beginPath();
    ctx.rect(1, 1, width - 2, ROWS * cell - 2);
    for (y = 0; y < ROWS; y++) {
      for (x = 0; x < COLS; x++) {
        if (x < COLS - 1 && !(maze[y][x] & 1)) { ctx.moveTo((x + 1) * cell, y * cell); ctx.lineTo((x + 1) * cell, (y + 1) * cell); }
        if (y < ROWS - 1 && !(maze[y][x] & 4)) { ctx.moveTo(x * cell, (y + 1) * cell); ctx.lineTo((x + 1) * cell, (y + 1) * cell); }
      }
    }
    ctx.stroke();

    var whole = Math.floor(progress);
    var frac = progress - whole;
    var here = center(walk[whole]);
    if (whole < walk.length - 1) {
      var ahead = center(walk[whole + 1]);
      here = [here[0] + (ahead[0] - here[0]) * frac, here[1] + (ahead[1] - here[1]) * frac];
    }

    // Trajectory so far.
    ctx.strokeStyle = colors.accent;
    ctx.lineWidth = Math.max(2, cell * 0.12);
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    var first = center(walk[0]);
    ctx.moveTo(first[0], first[1]);
    for (var i = 1; i <= whole; i++) { var p = center(walk[i]); ctx.lineTo(p[0], p[1]); }
    ctx.lineTo(here[0], here[1]);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Start marker.
    ctx.fillStyle = colors.accent;
    ctx.fillRect(first[0] - cell * 0.18, first[1] - cell * 0.18, cell * 0.36, cell * 0.36);

    // Robot, shaking when it bumps into the dead end.
    var shake = bump > 0 ? Math.sin(bump * 40) * bump * cell * 0.12 : 0;
    var r = cell * 0.3;
    ctx.fillStyle = colors.paper;
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(here[0] + shake, here[1], r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = colors.accent;
    ctx.fillRect(here[0] + shake - r * 0.45, here[1] - r * 0.2, r * 0.3, r * 0.3);
    ctx.fillRect(here[0] + shake + r * 0.15, here[1] - r * 0.2, r * 0.3, r * 0.3);

    if (progress >= walk.length - 1 && bump <= 0) {
      ctx.fillStyle = colors.accent;
      ctx.font = '600 ' + Math.round(cell * 0.6) + 'px ' + getComputedStyle(log).fontFamily;
      ctx.textAlign = 'center';
      ctx.fillText('?', here[0], here[1] - r - 6);
    }
  }

  function short(text) { return text.length > 48 ? text.slice(0, 45) + '...' : text; }

  function writeLog(lines) { log.textContent = lines.join('\n'); }

  function finish() {
    var steps = walk.length - 1;
    var random = rng(hash(path) + episode * 7919 + 1);
    ndtw.textContent = (0.05 + random() * 0.25).toFixed(2);
    writeLog([
      '$ navigate --instruction "Go to ' + short(path) + ' and stop."',
      '[episode ' + episodeId() + '] scene: tingdeliu.github.io',
      '[step ' + steps + '] dead end, collision with wall',
      '[step ' + steps + '] STOP',
      '[eval] target not in scene graph'
    ]);
    metrics.hidden = false;
  }

  function episodeId() { return '0x' + ((hash(path) + episode * 2654435761) >>> 0).toString(16).slice(-4).padStart(4, '0'); }

  function loop(now) {
    raf = 0;
    var elapsed = Math.max(0, now - start);   // the frame timestamp can predate start
    var last = walk.length - 1;
    progress = Math.min(last, elapsed / stepMs);
    if (progress >= last) {
      var since = elapsed - last * stepMs;
      bump = Math.max(0, 1 - since / 500);
      if (bump <= 0) { draw(); finish(); return; }
    }
    draw();
    raf = requestAnimationFrame(loop);
  }

  function run() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    var random = rng(hash(path) + episode * 2654435761);
    maze = buildMaze(random);
    walk = explore(random);
    stepMs = Math.min(STEP_MS, WALK_MS / Math.max(1, walk.length - 1));
    metrics.hidden = true;
    writeLog([
      '$ navigate --instruction "Go to ' + short(path) + ' and stop."',
      '[episode ' + episodeId() + '] scene: tingdeliu.github.io',
      '[step 0] exploring...'
    ]);
    if (reduce.matches) {
      progress = walk.length - 1;
      bump = 0;
      draw();
      finish();
      return;
    }
    progress = 0;
    bump = 0;
    start = performance.now();
    raf = requestAnimationFrame(loop);
  }

  document.querySelectorAll('[data-nf-retry]').forEach(function (button) {
    button.addEventListener('click', function () { episode += 1; run(); });
  });

  readColors();
  new MutationObserver(function () { readColors(); draw(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(canvas);
  else window.addEventListener('resize', resize);
  run();
  resize();
})();
