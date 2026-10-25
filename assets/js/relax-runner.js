(function () {
  'use strict';
  // Pixel runner on /relax/: the dino jumps crates, cones and low drones, must not jump into high ones,
  // and can pick up shields that smash whatever it runs into for a few seconds.
  var stage = document.querySelector('[data-relax-stage]');
  if (!stage) return;
  var canvas = stage.querySelector('[data-relax-canvas]');
  var message = stage.querySelector('[data-relax-message]');
  var scoreEl = document.querySelector('[data-relax-score]');
  var bestEl = document.querySelector('[data-relax-best]');
  var levelEl = document.querySelector('[data-relax-level]');
  var shieldBtn = document.querySelector('[data-relax-shield]');
  var skinBtn = stage.querySelector('[data-relax-skin]');
  var skinIcon = skinBtn.querySelector('canvas');
  var chargesEl = document.querySelector('[data-relax-charges]');
  var ctx = canvas.getContext('2d');

  var H = 120;              // logical height in game pixels
  var GROUND = 104;
  var DINO_X = 20;
  var GRAVITY = 1100;
  var HOLD_GRAVITY = 650;   // lighter gravity while the jump key is held and the dino is rising
  var JUMP_SPEED = 290;
  var CUT_SPEED = 140;      // releasing early caps the upward speed, giving a short hop
  var BUFFER_MS = 120;      // a press shortly before landing still jumps
  var MAX_SPEED = 340;
  var SHIELD_SECONDS = 4;
  var MAX_CHARGES = 3;
  var SMASH_BONUS = 15;
  var BEST_KEY = 'relax-runner-best';
  var LEVEL_TIPS = ['', '高空无人机：别跳', '低空无人机要跳，还有连排障碍', '无人机开始上下浮动', '节奏组合：跳与不跳交替', '冲刺无人机迎面而来'];

  // Silhouette only: # body, b belly, s spike, e eye, w eye shine. The ink outline is traced at load time.
  var BODY = [
    '...........#######..',
    '..........#########.',
    '..........######we#.',
    '..........######ee#.',
    '..........#########.',
    '..........#########.',
    '..........#####.....',
    '.....s.s.s#######...',
    '....############....',
    '..#############b#...',
    '##############bb.##.',
    '...###########bb#...',
    '.....#########b#....',
    '......#########.....'
  ];
  var LEGS = [
    ['......##...##.......', '......#.....#.......', '......##....##......'],
    ['.......##.##........', '.......#...#........', '......##...##.......'],
    ['......##..##........', '......##..##........', '....................']
  ];
  var FRAMES = LEGS.map(function (legs) { return trace(BODY.concat(legs)); });
  var DEAD = FRAMES[0].slice();
  DEAD[3] = DEAD[3].slice(0, 17) + 'e#' + DEAD[3].slice(19);   // eyes become a pixel "x"
  DEAD[4] = DEAD[4].slice(0, 17) + '#e' + DEAD[4].slice(19);
  var SLEEP = FRAMES[0].slice();
  SLEEP[3] = SLEEP[3].slice(0, 17) + '##' + SLEEP[3].slice(19);  // eyes shut
  var DINO_W = FRAMES[0][0].length;
  var DINO_H = FRAMES[0].length;

  // Tin-toy robot, the rarest character from the dice in the stage corner. Same size as the dino. m metal, v dark panel, o eye,
  // a accent light, w dial face, g joints and arms.
  var ROBOT_BODY = [
    '.............a......',
    '.............g......',
    '.........mmmmmmmmm..',
    '.........moommmoom..',
    '.........moommmoom..',
    '.........mmmmmmmmm..',
    '.........mmvgvgvmm..',
    '.........mmmmmmmmm..',
    '............ggg.....',
    '........gmmmmmmmmmg.',
    '........gmmvvvvvmmg.',
    '........gmmvawavmmg.',
    '........g.mmmmmmm.g.',
    '..........mmmmmmm...'
  ];
  var ROBOT_LEGS = [
    ['..........gg..gg....', '.........gg....gg...', '........mmm....mmm..'],
    ['..........g....g....', '..........g....g....', '.........mm....mm...'],
    ['..........gg..gg....', '..........mm..mm....', '....................']
  ];
  var ROBOT_FRAMES = ROBOT_LEGS.map(function (legs) { return trace(ROBOT_BODY.concat(legs)); });
  var ROBOT_SLEEP = ROBOT_FRAMES[0].slice();
  ROBOT_SLEEP[4] = ROBOT_SLEEP[4].replace(/o/g, 'm');   // eyes shut to a dark line
  ROBOT_SLEEP[5] = ROBOT_SLEEP[5].replace(/o/g, 'v');
  // Jet flame under each foot while airborne, two flicker frames. a accent shell, w hot core.
  var FLAMES = [['.a.', 'awa', 'awa', '.a.'], ['aaa', 'awa', '.w.', '.a.']];
  var FLAME_X = [10, 14];
  var FLAME_CORE = '#ffd08a';
  // hit: insets from the left, right and top of the sprite box, skipping tails and antennas.
  // More characters, same 20 x 17 grid and letters as the robot.
  var HOVER_BODY = [
    '..........mmmmm.....',
    '........mmmmmmmmm...',
    '.......mmvvvvvvvmm..',
    '.......mvvvvvvvvvm..',
    '.......mvvovvvovvm..',
    '.......mvvovvvovvm..',
    '.......mmvvvvvvvmm..',
    '........mmmmmmmmm...',
    '......g..mmmmmmm..g.',
    '......gg.mmaammm.gg.',
    '.........mmmmmmm....',
    '..........mmmmm.....',
    '...........ggg......',
    '....................'
  ];
  var HOVER_LEGS = [   // no legs: a jet flame that flickers, bigger in the air
    ['..........aaaaa.....', '...........aaa......', '....................'],
    ['...........aaa......', '..........aaaaa.....', '...........aaa......'],
    ['..........aaaaa.....', '.........aaaaaaa....', '..........aa.aa.....']
  ];
  var TANK_BODY = [
    '....................',
    '........mmmm.mmmm...',
    '.......mvvvvmvvvvm..',
    '.......mvvwvmvvwvm..',
    '.......mvvvvmvvvvm..',
    '........mmmm.mmmm...',
    '...........gg.......',
    '...........gg.......',
    '......mmmmmmmmmmm...',
    '......mmaammmmmmm.g.',
    '......mmmmmmmmmmmgg.',
    '......mmmmmmmmmmm...',
    '......mmmmmmmmmmm...',
    '....................'
  ];
  var TANK_LEGS = [   // treads roll by shifting their studs
    ['.....vvvvvvvvvvvvv..', '....vgvvgvvgvvgvvgv.', '.....vvvvvvvvvvvvv..'],
    ['.....vvvvvvvvvvvvv..', '....vvgvvgvvgvvgvvv.', '.....vvvvvvvvvvvvv..'],
    ['.....vvvvvvvvvvvvv..', '....vgvvgvvgvvgvvgv.', '.....vvvvvvvvvvvvv..']
  ];
  var DOG_BODY = [
    '....................',
    '....................',
    '....................',
    '...............mmm..',
    '..............mmmmmm',
    '..............mvovvm',
    '..............mmmmm.',
    '..a..........mm.....',
    '...g.mmmmmmmmmm.....',
    '....mmmmmmmmmmmm....',
    '....mmaammmmmmmm....',
    '....mmmmmmmmmmmm....',
    '.....gg......gg.....',
    '.....gg......gg.....'
  ];
  var DOG_LEGS = [
    ['.....g........g.....', '....g..........g....', '...mm.........mm....'],
    ['......g......g......', '.......g......g.....', '......mm.....mm.....'],
    ['......g......g......', '.....g........g.....', '....................']
  ];

  // Golden capybara with an orange on its head. y gold, d dark gold, h shine, e eye, a orange, g leaf.
  var CAPY_BODY = [
    '.............g......',
    '............aaa.....',
    '...........aaaaa....',
    '...........aaaaa....',
    '............aaa.....',
    '...........dyyyyyy..',
    '..........yyyyyyyyy.',
    '..........yyyyyeyyyy',
    '...yyyyyyyyyyyyyyydd',
    '..yhhyyyyyyyyyyyyyy.',
    '.yhyyyyyyyyyyyyyyyy.',
    '.yyyyyyyyyyyyyyyyy..',
    '.dyyyyyyyyyyyyyyyy..',
    '..ddyyyyyyyyyyyyy...'
  ];
  var CAPY_LEGS = [
    ['...yy........yy.....', '...yy........yy.....', '...dd........dd.....'],
    ['....yy......yy......', '....yy......yy......', '....dd......dd......'],
    ['...yy........yy.....', '...dd........dd.....', '....................']
  ];
  var GOLD = { y: '#e0b03c', d: '#9c7419', h: '#fff3c4' };

  function build(body, legs) { return legs.map(function (l) { return trace(body.concat(l)); }); }
  function swap(rows, from, to) { return rows.map(function (r) { return r.replace(from, to); }); }
  var HOVER = build(HOVER_BODY, HOVER_LEGS);
  var TANK = build(TANK_BODY, TANK_LEGS);
  var DOG = build(DOG_BODY, DOG_LEGS);
  var CAPY = build(CAPY_BODY, CAPY_LEGS);

  // Every character plays the same; only the look and how rare it is differ. weight is out of 100.
  // hit: insets from the left, right and top of the sprite box, skipping tails and antennas.
  var SKINS = {
    dino: { name: '小恐龙', rank: 'N 普通', weight: 40, frames: FRAMES, dead: DEAD, sleep: SLEEP, hit: [5, 3, 2] },
    tank: { name: '履带小车', rank: 'N 普通', weight: 25, frames: TANK, dead: swap(TANK[0], /[wa]/g, 'g'), sleep: swap(TANK[0], /w/g, 'v'), hit: [5, 2, 2] },
    hover: { name: '悬浮蛋仔', rank: 'R 稀有', weight: 18, frames: HOVER, dead: swap(HOVER[0], /[oa]/g, 'g'), sleep: swap(HOVER[0], /o/g, 'v'), hit: [8, 3, 1] },
    dog: { name: '机器狗', rank: 'SR 史诗', weight: 12, frames: DOG, dead: swap(DOG[0], /[oa]/g, 'g'), sleep: swap(DOG[0], /o/g, 'm'), hit: [5, 1, 4] },
    robot: {
      name: '铁皮机器人', rank: 'SSR 传说', weight: 4,
      frames: ROBOT_FRAMES,
      dead: swap(ROBOT_FRAMES[0], /[oa]/g, 'g'),   // lights go out
      sleep: ROBOT_SLEEP,
      hit: [9, 3, 3]
    },
    // The rarest one sheds golden sparkles while it runs. The orange on its head doesn't count as a hit.
    capy: { name: '金色卡皮巴拉', rank: 'UR 金色传说', weight: 1, frames: CAPY, dead: swap(CAPY[0], /e/g, 'd'), sleep: swap(CAPY[0], /e/g, 'd'), hit: [3, 1, 6] }
  };
  var ROLL_MS = 700;
  var SKIN_KEY = 'relax-runner-skin';
  var SLEEP_MS = 30000;
  // Night sky between 23:00 and 05:00 local time (or with ?night in the URL).
  var NIGHT = { paper: '#161a26', panel: '#252a3a', ink: '#e9e6da', muted: '#8d91a3', rule: '#2d3245', accent: '#e39572', selection: '#6e4a3c' };
  var R2_BONUS = 30;
  var R2_ACCEPT_BONUS = 50;
  // Shield pickup, drawn with the same palette as the dino.
  var SHIELD = trace([
    '#######',
    '#ww####',
    '#w#####',
    '#######',
    '.#####.',
    '..###..',
    '...#...'
  ]);

  var colors = {};
  var W = 320;
  var scale = 2;
  var state = 'ready';      // ready | running | paused | over
  var dino = { y: 0, vy: 0, grounded: true };
  var obstacles = [];
  var items = [];
  var particles = [];
  var clouds = [];
  var speed = 0;
  var distance = 0;
  var bonus = 0;
  var elapsed = 0;
  var nextGap = 0;
  var charges = 0;
  var shieldLeft = 0;
  var currentLevel = 0;
  var tipUntil = 0;
  var lastGroupHigh = false;
  var holding = false;
  var bufferedAt = -1;
  var overAt = 0;
  var best = 0;
  var raf = 0;
  var last = 0;
  var lastMilestone = 0;
  var skin = 'dino';
  var rolling = false;
  var idleSince = performance.now();
  var night = false;
  var stars = [];

  try { best = parseInt(localStorage.getItem(BEST_KEY), 10) || 0; } catch (e) { best = 0; }
  try { var saved = localStorage.getItem(SKIN_KEY); if (SKINS[saved]) skin = saved; } catch (e) { /* keep the dino */ }

  function isNight() {
    var h = new Date().getHours();
    return /[?&]night\b/.test(location.search) || h >= 23 || h < 5;
  }

  function readyText() {
    return night ? '这么晚还在读论文？跑两步就去睡吧 · 按空格开始' : '按空格或点击画面开始';
  }

  // Short in-game notices (level ups, Reviewer 2) that clear themselves after a moment.
  function toast(text) {
    setMessage(text);
    tipUntil = elapsed + 1.6;
  }

  function pad(n) { return ('00000' + Math.floor(n)).slice(-5); }
  function score() { return Math.floor(distance / 8) + bonus; }

  // Levels never stop: each one asks for 50 more points than the last (0, 150, 350, 600, 900, 1250, ...).
  function levelStart(n) { return 150 * n + 25 * n * (n - 1); }
  function level() {
    var s = score();
    var n = 0;
    while (levelStart(n + 1) <= s) n++;
    return n;
  }
  function levelProgress(n) { return (score() - levelStart(n)) / (levelStart(n + 1) - levelStart(n)); }

  function readColors() {
    var css = getComputedStyle(document.documentElement);
    ['paper', 'panel', 'ink', 'muted', 'rule', 'accent', 'selection'].forEach(function (name) {
      colors[name] = night ? NIGHT[name] : css.getPropertyValue('--' + name).trim() || '#888';
    });
    stage.classList.toggle('is-night', night);
    drawDie();
    draw();
  }

  function resize() {
    var width = stage.clientWidth;
    scale = width >= 720 ? 3 : 2;
    W = Math.ceil(width / scale);
    canvas.width = W;
    canvas.height = H;
    canvas.style.width = W * scale + 'px';
    canvas.style.height = H * scale + 'px';
    ctx.imageSmoothingEnabled = false;
    stars = [];
    for (var i = 0; i < W / 9; i++) stars.push({ x: Math.floor(Math.random() * W), y: 3 + Math.floor(Math.random() * 40), phase: Math.random() * 6 });
    draw();
  }

  function setMessage(text) {
    message.textContent = text;
    message.hidden = !text;
  }

  function reset() {
    dino.y = 0; dino.vy = 0; dino.grounded = true;
    obstacles = [];
    items = [];
    particles = [];
    speed = 105;
    distance = 0;
    bonus = 0;
    elapsed = 0;
    nextGap = W * 0.8;
    charges = 0;
    shieldLeft = 0;
    currentLevel = 0;
    tipUntil = 0;
    lastGroupHigh = false;
    lastMilestone = 0;
    clouds = [];
    for (var i = 0; i < 4; i++) clouds.push({ x: Math.random() * W, y: 10 + Math.random() * 40 });
  }

  function start() {
    if (state === 'over' && performance.now() - overAt < 400) return;   // a stray tap right after a crash should not restart
    if (state !== 'paused') {
      reset();
      if (isNight() !== night) { night = !night; readColors(); }
    }
    state = 'running';
    setMessage('');
    last = performance.now();
    updateHud();
    if (!raf) raf = requestAnimationFrame(loop);
  }

  function pause() {
    if (state !== 'running') return;
    state = 'paused';
    holding = false;
    tipUntil = 0;
    setMessage('已暂停 · 按空格或点击继续');
    updateHud();
  }

  function crash(o) {
    state = 'over';
    overAt = performance.now();
    holding = false;
    var s = score();
    var byReviewer = o && o.type === 'r2';
    var lead = byReviewer ? 'Reviewer 2 给了 Reject · ' : '';
    var again = byReviewer ? '按空格或点击再改一版' : '按空格或点击再来一次';
    if (s > best) {
      best = s;
      try { localStorage.setItem(BEST_KEY, String(best)); } catch (e) { /* private mode: keep it for this visit */ }
      setMessage(lead + '新纪录 ' + s + '！' + again);
    } else {
      setMessage((lead || '撞到啦 · ') + again);
    }
    updateHud();
  }

  function pickSkin() {
    var ticket = Math.random() * 100;
    var names = Object.keys(SKINS);
    for (var i = 0; i < names.length; i++) {
      ticket -= SKINS[names[i]].weight;
      if (ticket < 0) return names[i];
    }
    return names[0];
  }

  // Shake the die for a moment, flipping faces, then land on a character by weight.
  function rollSkin() {
    if (rolling) return;
    rolling = true;
    skinBtn.classList.add('is-rolling');
    var flips = setInterval(function () { drawDie(1 + Math.floor(Math.random() * 6)); }, 80);
    setTimeout(function () {
      clearInterval(flips);
      rolling = false;
      skinBtn.classList.remove('is-rolling');
      var next = pickSkin();
      var same = next === skin;
      skin = next;
      drawDie(1 + Math.floor(Math.random() * 6));
      try { localStorage.setItem(SKIN_KEY, skin); } catch (e) { /* lasts for this visit */ }
      var look = SKINS[skin];
      var lucky = look.weight <= 1 ? '✦✦ ' : look.weight <= 5 ? '★ ' : '';
      var text = lucky + look.rank + ' · ' + look.name + (same ? '（又是它）' : look.weight <= 1 ? '！！' : lucky ? '！' : '');
      showSkinNotice(text);
    }, ROLL_MS);
  }

  function showSkinNotice(text) {
    if (state === 'running') { toast(text); return; }
    setMessage(text);
    var shown = state;
    setTimeout(function () { if (state === shown && state === 'ready') setMessage(readyText()); }, 2200);
    draw();
  }

  function press() {
    holding = true;
    if (state !== 'running') { start(); return; }
    if (dino.grounded) jump();
    else bufferedAt = performance.now();
  }

  function release() {
    holding = false;
    if (state === 'running' && dino.vy < -CUT_SPEED) dino.vy = -CUT_SPEED;
  }

  function jump() {
    dino.vy = -JUMP_SPEED;
    dino.grounded = false;
    bufferedAt = -1;
  }

  function activateShield() {
    if (state !== 'running' || charges === 0 || shieldLeft > 0) return;
    charges--;
    shieldLeft = SHIELD_SECONDS;
    updateHud();
  }

  function ground(type, x) {
    var size = { crate: [10, 10], crates: [20, 10], cone: [8, 14], stack: [10, 20] }[type];
    return { type: type, x: x, w: size[0], h: size[1], y: GROUND - size[1], vx: 0 };
  }

  function drone(high, x, lv) {
    // A high drone clears a standing dino and hits a jumping one; a low drone has to be jumped.
    var y = high ? GROUND - DINO_H - 10 : GROUND - 12;
    var bob = lv < 3 ? 0 : high ? 3 : Math.min(7, 3 + (lv - 3) * 0.5);
    return { type: 'drone', high: high, x: x, w: 14, h: 7, y: y, baseY: y, bob: bob, phase: Math.random() * 6, vx: 0 };
  }

  // Reviewer 2: a big drone weaving between the high and low lanes. Stay under it while it is high,
  // or clear it with a full jump at any height. Passing earns a rebuttal, smashing it an accept.
  function reviewer(x) {
    return { type: 'r2', high: true, x: x, w: 20, h: 9, y: GROUND - 21, phase: Math.random() * 6, vx: 0 };
  }

  function pick(list) { return list[Math.floor(Math.random() * list.length)]; }

  function spawn() {
    var lv = level();
    var roll = Math.random();
    var x = W + 4;
    var pDrone = lv >= 1 ? Math.min(0.3, 0.14 + 0.03 * lv) : 0;
    var pRow = lv >= 2 ? Math.min(0.25, 0.14 + 0.02 * lv) : 0;
    var pCombo = lv >= 4 ? Math.min(0.25, 0.1 + 0.03 * (lv - 4)) : 0;
    var group = [];
    var i;
    if (lv >= 2 && Math.random() < 0.07 && !obstacles.some(function (o) { return o.type === 'r2'; })) {
      group.push(reviewer(x));
    } else if (roll < pDrone) {
      var d = drone(lv < 2 || Math.random() < 0.5, x, lv);
      // A dashing drone flies at the dino faster than the ground scrolls. Only low ones, and never right behind
      // a high drone, so closing in on the group ahead can't create a "jump and don't jump" overlap.
      if (lv >= 5 && !d.high && !lastGroupHigh && Math.random() < 0.5) d.vx = 40 + Math.random() * 50;
      group.push(d);
    } else if (roll < pDrone + pRow) {
      // A row with short gaps: one long, held jump clears it at low speed; a tap is enough once it is fast.
      var count = 2 + Math.floor(Math.random() * (lv >= 6 ? 3 : lv >= 3 ? 2 : 1));
      for (i = 0; i < count; i++) {
        var prev = group[group.length - 1];
        group.push(ground(Math.random() < 0.7 ? 'crate' : 'cone', prev ? prev.x + prev.w + 10 + Math.floor(Math.random() * 10) : x));
      }
    } else if (roll < pDrone + pRow + pCombo) {
      // Rhythm section: three obstacles a short hop apart, mixing ones to jump and drones to stay under.
      var spacing = speed * 0.55 + 30;
      for (i = 0; i < 3; i++) {
        var kind = pick(i === 0 ? ['crate', 'cone', 'low'] : ['crate', 'cone', 'low', 'high', 'high']);
        group.push(kind === 'low' || kind === 'high' ? drone(kind === 'high', x + i * spacing, lv) : ground(kind, x + i * spacing));
      }
    } else {
      group.push(ground(pick(lv >= 1 ? ['crate', 'crate', 'crates', 'cone', 'stack'] : ['crate', 'crate', 'crates', 'cone']), x));
    }
    group.forEach(function (o) { obstacles.push(o); });
    lastGroupHigh = group.some(function (o) { return o.high; });
    var lastOne = group[group.length - 1];
    var span = lastOne.x + lastOne.w - x;
    var minGap = speed * Math.max(0.45, 0.8 - 0.04 * lv) + 40;
    nextGap = span + minGap + Math.random() * minGap * Math.max(0.35, (lastOne.type === 'drone' ? 1.2 : 0.9) - 0.06 * lv);
    if (lastOne.type === 'r2') nextGap += minGap * 0.5;
    // Now and then a shield floats in the gap, high enough that only a real jump reaches it.
    if (lv >= 1 && !items.length && Math.random() < 0.15) {
      var y = GROUND - 50 - Math.floor(Math.random() * 10);
      items.push({ x: x + span + (nextGap - span) * 0.45, y: y, baseY: y, w: SHIELD[0].length, h: SHIELD.length, phase: Math.random() * 6 });
    }
  }

  function hits(o) {
    var top = GROUND - DINO_H + dino.y;
    var inset = SKINS[skin].hit;
    // Hitbox skips thin parts and the outline so near misses feel fair.
    return DINO_X + inset[0] < o.x + o.w - 1 && DINO_X + DINO_W - inset[1] > o.x + 1 &&
      top + inset[2] < o.y + o.h - 1 && top + DINO_H > o.y + 1;
  }

  function touches(o) {
    var top = GROUND - DINO_H + dino.y;
    return DINO_X < o.x + o.w && DINO_X + DINO_W > o.x && top < o.y + o.h && top + DINO_H > o.y;
  }

  function smash(o) {
    bonus += SMASH_BONUS;
    for (var i = 0; i < 12; i++) {
      particles.push({
        x: o.x + Math.random() * o.w, y: o.y + Math.random() * o.h,
        vx: 40 + Math.random() * 140, vy: -60 - Math.random() * 140, life: 0.5 + Math.random() * 0.4,
        color: i % 3 ? colors.ink : colors.accent
      });
    }
  }

  function update(dt) {
    elapsed += dt;
    var lv = level();
    var target = Math.min(MAX_SPEED, 105 + 20 * (lv + levelProgress(lv)));
    speed += Math.max(-40 * dt, Math.min(40 * dt, target - speed));
    var dx = speed * dt;
    distance += dx;
    if (shieldLeft > 0) shieldLeft = Math.max(0, shieldLeft - dt);
    if (skin === 'capy' && Math.random() < dt * 10) {
      particles.push({
        x: DINO_X + 2 + Math.random() * 16, y: GROUND - DINO_H + dino.y + 6 + Math.random() * 10,
        vx: speed * 0.8, vy: -40 - Math.random() * 40, life: 0.5, color: Math.random() < 0.5 ? GOLD.h : GOLD.y
      });
    }

    if (!dino.grounded) {
      var g = holding && dino.vy < 0 ? HOLD_GRAVITY : GRAVITY;
      dino.vy += g * dt;
      dino.y += dino.vy * dt;
      if (dino.y >= 0) {
        dino.y = 0; dino.vy = 0; dino.grounded = true;
        if (bufferedAt > 0 && performance.now() - bufferedAt < BUFFER_MS) jump();
      }
    }

    for (var i = obstacles.length - 1; i >= 0; i--) {
      var o = obstacles[i];
      o.x -= dx + o.vx * dt;
      if (o.bob) o.y = o.baseY + Math.round(Math.sin(elapsed * 4 + o.phase) * o.bob);
      if (o.type === 'r2') {
        // Two mixed waves keep it hard to predict while staying between the lanes (GROUND-30 to GROUND-12).
        var wave = (Math.sin(elapsed * 3.2 + o.phase) + 0.35 * Math.sin(elapsed * 7.1 + o.phase * 2)) / 1.35;
        o.y = Math.round(GROUND - 21 + wave * 9);
        if (!o.passed && o.x + o.w < DINO_X + 5) {
          o.passed = true;
          bonus += R2_BONUS;
          toast('Rebuttal 成功 +' + R2_BONUS);
        }
      }
      if (o.x + o.w < 0) obstacles.splice(i, 1);
    }
    nextGap -= dx;
    if (nextGap <= 0) spawn();

    for (var k = items.length - 1; k >= 0; k--) {
      var it = items[k];
      it.x -= dx;
      it.y = it.baseY + Math.round(Math.sin(elapsed * 5 + it.phase) * 2);
      if (touches(it)) {
        items.splice(k, 1);
        charges = Math.min(MAX_CHARGES, charges + 1);
        flash(chargesEl);
      } else if (it.x + it.w < 0) {
        items.splice(k, 1);
      }
    }

    for (var p = particles.length - 1; p >= 0; p--) {
      var pt = particles[p];
      pt.vy += GRAVITY * 0.5 * dt;
      pt.x += (pt.vx - speed) * dt;
      pt.y += pt.vy * dt;
      pt.life -= dt;
      if (pt.life <= 0 || pt.y > H) particles.splice(p, 1);
    }

    clouds.forEach(function (c) {
      c.x -= dx * 0.15;
      if (c.x < -20) { c.x = W + Math.random() * 40; c.y = 10 + Math.random() * 40; }
    });

    for (var j = obstacles.length - 1; j >= 0; j--) {
      if (!hits(obstacles[j])) continue;
      if (shieldLeft > 0) {
        var hit = obstacles.splice(j, 1)[0];
        smash(hit);
        if (hit.type === 'r2') {
          bonus += R2_ACCEPT_BONUS - SMASH_BONUS;
          toast('Accepted! +' + R2_ACCEPT_BONUS);
        }
      } else { crash(obstacles[j]); return; }
    }

    var s = score();
    if (s - lastMilestone >= 100) {
      lastMilestone = s - s % 100;
      flash(scoreEl);
    }
    lv = level();
    if (lv > currentLevel) {
      currentLevel = lv;
      flash(levelEl);
      toast('LV ' + (lv + 1) + ' · ' + (LEVEL_TIPS[lv] || '更快了'));
    } else if (tipUntil && elapsed > tipUntil) {
      tipUntil = 0;
      setMessage('');
    }
  }

  function flash(el) {
    el.classList.remove('is-milestone');
    void el.offsetWidth;   // restart the flash animation
    el.classList.add('is-milestone');
  }

  function updateHud() {
    scoreEl.textContent = pad(score());
    levelEl.textContent = level() + 1;
    bestEl.textContent = pad(best);
    chargesEl.textContent = charges;
    shieldBtn.disabled = state !== 'running' || charges === 0 || shieldLeft > 0;
    shieldBtn.classList.toggle('is-active', state === 'running' && shieldLeft > 0);
  }

  function px(x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.fillRect(Math.round(x), Math.round(y), w, h);
  }

  // Pads the silhouette by one pixel and marks every empty pixel touching it as outline (k).
  function trace(rows) {
    var grid = [Array(rows[0].length + 3).join('.')].concat(rows.map(function (r) { return '.' + r + '.'; }));
    return grid.map(function (row, y) {
      return row.replace(/\./g, function (ch, x) {
        var near = [grid[y - 1] && grid[y - 1][x], grid[y + 1] && grid[y + 1][x], row[x - 1], row[x + 1]];
        return near.some(function (n) { return n && n !== '.'; }) ? 'k' : '.';
      });
    });
  }

  function palette() {
    return {
      k: colors.ink, e: colors.ink, '#': colors.accent, b: colors.panel, s: colors.selection, w: colors.paper,
      m: colors.panel, v: colors.ink, o: colors.accent, a: colors.accent, g: colors.muted,
      y: GOLD.y, d: GOLD.d, h: GOLD.h
    };
  }

  function sprite(rows, x, y, target) {
    var map = palette();
    var c2d = target || ctx;
    for (var r = 0; r < rows.length; r++) {
      for (var c = 0; c < rows[r].length; c++) {
        var ch = rows[r][c];
        if (ch === '.') continue;
        c2d.fillStyle = map[ch];
        c2d.fillRect(Math.round(x + c), Math.round(y + r), 1, 1);
      }
    }
  }

  // Pixel die in the corner button: 11 x 11 with an outline and the pips of one face.
  var PIPS = { 1: [[5, 5]], 2: [[3, 3], [7, 7]], 3: [[3, 3], [5, 5], [7, 7]], 4: [[3, 3], [7, 3], [3, 7], [7, 7]],
    5: [[3, 3], [7, 3], [5, 5], [3, 7], [7, 7]], 6: [[3, 3], [7, 3], [3, 5], [7, 5], [3, 7], [7, 7]] };
  var dieFace = 5;
  function drawDie(face) {
    dieFace = face || dieFace;
    var c2d = skinIcon.getContext('2d');
    skinIcon.width = 11;
    skinIcon.height = 11;
    c2d.fillStyle = colors.ink;
    c2d.fillRect(1, 0, 9, 11);
    c2d.fillRect(0, 1, 11, 9);
    c2d.fillStyle = colors.panel;
    c2d.fillRect(1, 1, 9, 9);
    c2d.fillStyle = colors.accent;
    PIPS[dieFace].forEach(function (p) { c2d.fillRect(p[0], p[1], 1, 1); });
  }

  function sleeping() {
    return state === 'ready' && performance.now() - idleSince > SLEEP_MS;
  }

  function drawDino() {
    var look = SKINS[skin];
    var frame;
    if (state === 'over') frame = look.dead;
    else if (sleeping()) frame = look.sleep;
    else if (!dino.grounded) frame = look.frames[2];
    else if (state === 'running') frame = look.frames[Math.floor(distance / 10) % 2];
    else frame = look.frames[0];
    var top = GROUND - DINO_H + Math.round(dino.y);
    sprite(frame, DINO_X, top);
    if (skin === 'robot' && !dino.grounded && state === 'running') {
      var flame = FLAMES[Math.floor(performance.now() / 70) % 2];
      FLAME_X.forEach(function (fx) {
        flame.forEach(function (row, fy) {
          for (var c = 0; c < row.length; c++) {
            if (row[c] !== '.') px(DINO_X + fx + c, top + DINO_H + fy, 1, 1, row[c] === 'w' ? FLAME_CORE : colors.accent);
          }
        });
      });
    }
    if (sleeping()) {
      // Three z's drifting up from the head, each one a little bigger.
      var t = (performance.now() - idleSince) / 1000;
      for (var z = 0; z < 3; z++) {
        var age = (t * 0.6 + z / 3) % 1;
        var size = 2 + z;
        var zx = DINO_X + 18 + z * 4 + Math.round(age * 4);
        var zy = top - 2 - Math.round(age * 16) - z * 3;
        px(zx, zy, size, 1, colors.muted);
        for (var d = 1; d < size - 1; d++) px(zx + size - 1 - d, zy + d, 1, 1, colors.muted);
        px(zx, zy + size - 1, size, 1, colors.muted);
      }
    }
    // Shield: a ring of dots spinning around the dino, blinking in its last second.
    if (shieldLeft > 0 && state !== 'over' && (shieldLeft > 1 || Math.floor(elapsed * 10) % 2)) {
      var cx = DINO_X + DINO_W / 2;
      var cy = top + DINO_H / 2;
      for (var a = 0; a < 20; a++) {
        var angle = a / 20 * Math.PI * 2 + elapsed * 3;
        px(cx + Math.cos(angle) * 15, cy + Math.sin(angle) * 13, 1, 1, a % 2 ? colors.accent : colors.ink);
      }
    }
  }

  var GLYPHS = { R: ['##.', '#.#', '##.', '#.#', '#.#'], 2: ['###', '..#', '###', '#..', '###'] };

  function drawObstacle(o) {
    var x = Math.round(o.x);
    if (o.type === 'r2') {
      var turn = Math.floor(elapsed * 20) % 2;
      px(x + (turn ? 0 : 2), o.y, 6, 1, colors.muted);
      px(x + (turn ? 14 : 12), o.y, 6, 1, colors.muted);
      px(x + 3, o.y + 1, 1, 1, colors.ink);
      px(x + 16, o.y + 1, 1, 1, colors.ink);
      px(x, o.y + 2, o.w, 7, colors.ink);
      px(x + 1, o.y + 3, o.w - 2, 5, colors.accent);
      ['R', 2].forEach(function (ch, n) {
        GLYPHS[ch].forEach(function (row, gy) {
          for (var gx = 0; gx < 3; gx++) if (row[gx] === '#') px(x + 6 + n * 5 + gx, o.y + 3 + gy, 1, 1, colors.paper);
        });
      });
      return;
    }
    if (o.type === 'cone') {
      for (var r = 0; r < o.h - 2; r++) {
        var half = Math.floor(1 + r * 3 / (o.h - 2));
        px(x + 4 - half, o.y + r, half * 2, 1, r === 5 || r === 6 ? colors.panel : colors.accent);
      }
      px(x, o.y + o.h - 2, o.w, 2, colors.ink);
      return;
    }
    if (o.type === 'drone') {
      var spin = Math.floor(elapsed * 20) % 2;
      px(x + (spin ? 0 : 2), o.y, 5, 1, colors.muted);
      px(x + (spin ? 9 : 7), o.y, 5, 1, colors.muted);
      px(x + 3, o.y + 1, 1, 2, colors.ink);
      px(x + 10, o.y + 1, 1, 2, colors.ink);
      px(x + 1, o.y + 3, 12, 3, colors.ink);
      px(x + 6, o.y + 4, 2, 1, colors.accent);
      px(x + 3, o.y + 6, 1, 1, colors.ink);
      px(x + 10, o.y + 6, 1, 1, colors.ink);
      if (o.vx) {   // speed lines behind a dashing drone
        px(x + 15, o.y + 3, 4, 1, colors.muted);
        px(x + 16, o.y + 5, 3, 1, colors.muted);
      }
      return;
    }
    // Crates: tile 10 x 10 boxes over the obstacle's footprint.
    for (var cy = o.y; cy < o.y + o.h; cy += 10) {
      for (var cx = x; cx < x + o.w; cx += 10) {
        px(cx, cy, 10, 10, colors.ink);
        px(cx + 1, cy + 1, 8, 8, colors.panel);
        for (var d = 1; d < 9; d++) {
          px(cx + d, cy + d, 1, 1, colors.muted);
          px(cx + 9 - d, cy + d, 1, 1, colors.muted);
        }
      }
    }
  }

  function drawSky() {
    var t = performance.now() / 1000;
    stars.forEach(function (s) {
      if (Math.sin(t * 2 + s.phase) > -0.6) px(s.x, s.y, 1, 1, Math.sin(t * 2 + s.phase) > 0.7 ? colors.ink : colors.muted);
    });
    // Crescent moon: a disc with a second disc cut out of it.
    var mx = W - 30;
    for (var y = -6; y <= 6; y++) {
      for (var x = -6; x <= 6; x++) {
        if (x * x + y * y <= 36 && (x - 3) * (x - 3) + (y + 2) * (y + 2) > 30) px(mx + x, 14 + y, 1, 1, colors.ink);
      }
    }
  }

  function drawScenery() {
    if (night) drawSky();
    // Dot dunes in the background, echoing the footer field, scrolling slower than the ground.
    var offset = distance * 0.25;
    for (var y = 46; y < GROUND; y += 3) {
      var depth = (y - 46) / (GROUND - 46);
      for (var x = (3 - offset % 3) % 3; x < W; x += 3) {
        var wx = x + offset;
        var v = (Math.sin(wx * 0.05) + Math.sin(wx * 0.017 + y * 0.08) + Math.sin((wx + y) * 0.03)) / 6 + 0.5;
        if (v > 1.05 - depth * 0.55) px(x, y, 1, 1, colors.rule);
      }
    }
    clouds.forEach(function (c) {
      px(c.x + 3, c.y, 8, 1, colors.rule);
      px(c.x, c.y + 1, 16, 2, colors.rule);
    });
    px(0, GROUND, W, 1, colors.ink);
    // Pebbles: a fixed pattern per world column so they scroll with the ground.
    var start = Math.floor(distance / 7);
    for (var i = start; i * 7 - distance < W; i++) {
      var h = (i * 2654435761) >>> 0;
      if (h % 3 === 0) continue;
      px(i * 7 - distance, GROUND + 3 + h % 11, 1 + (h >> 4) % 2, 1, colors.muted);
    }
  }

  function draw() {
    if (!colors.paper) return;
    px(0, 0, W, H, colors.paper);
    drawScenery();
    obstacles.forEach(drawObstacle);
    items.forEach(function (it) { sprite(SHIELD, it.x, it.y); });
    drawDino();
    particles.forEach(function (pt) { px(pt.x, pt.y, 1, 1, pt.color); });
  }

  function loop(now) {
    raf = 0;
    if (state !== 'running') { draw(); return; }
    // The first frame's timestamp can predate the performance.now() taken in start().
    var dt = Math.max(0, Math.min((now - last) / 1000, 1 / 30));
    last = now;
    update(dt);
    updateHud();
    draw();
    if (state === 'running') raf = requestAnimationFrame(loop);
  }

  function isJumpKey(e) {
    return e.code === 'Space' || e.key === 'ArrowUp' || e.key === ' ';
  }

  function stageInView() {
    var r = stage.getBoundingClientRect();
    return r.bottom > 0 && r.top < window.innerHeight;
  }

  document.addEventListener('keydown', function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(t.tagName))) return;
    if (!stageInView()) return;
    idleSince = performance.now();
    if (isJumpKey(e)) {
      e.preventDefault();
      // Only Space starts or resumes a run; the up arrow is for jumping.
      if (!e.repeat && (state === 'running' || e.key !== 'ArrowUp')) press();
    } else if ((e.key === 'x' || e.key === 'X' || e.key === 'ArrowRight') && state === 'running') {
      e.preventDefault();
      activateShield();
    } else if (e.key === 'ArrowDown' && state === 'running') {
      e.preventDefault();
      if (!dino.grounded) dino.vy = Math.max(dino.vy, JUMP_SPEED);   // fast fall
    }
  });
  document.addEventListener('keyup', function (e) {
    if (isJumpKey(e)) release();
  });
  stage.addEventListener('pointerdown', function (e) {
    if (e.button !== 0) return;
    e.preventDefault();
    stage.focus({ preventScroll: true });
    press();
  });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (type) {
    stage.addEventListener(type, release);
  });
  // The shield button must not keep focus, or the next Space would press it instead of jumping.
  shieldBtn.addEventListener('pointerdown', function (e) { e.preventDefault(); });
  shieldBtn.addEventListener('click', function () {
    activateShield();
    stage.focus({ preventScroll: true });
  });
  // The skin button sits on the stage, so stop its press from also counting as a jump.
  skinBtn.addEventListener('pointerdown', function (e) { e.stopPropagation(); e.preventDefault(); });
  skinBtn.addEventListener('click', function () {
    rollSkin();
    stage.focus({ preventScroll: true });
  });
  document.addEventListener('visibilitychange', function () { if (document.hidden) pause(); });
  window.addEventListener('blur', pause);

  night = isNight();
  reset();
  updateHud();
  setMessage(readyText());
  resize();
  readColors();
  // While waiting on the start screen, redraw a few times a second so the dino can doze off.
  setInterval(function () { if (state === 'ready' && !document.hidden) draw(); }, 250);
  if ('ResizeObserver' in window) new ResizeObserver(resize).observe(stage);
  else window.addEventListener('resize', resize);
  new MutationObserver(readColors).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
})();
