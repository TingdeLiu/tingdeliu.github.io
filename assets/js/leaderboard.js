// 排行榜筛选：作用于 #lb-filter-bar 之后、下一个一级标题之前的所有「| 模型 | … |」表格。
// 按范式、输入配置、是否开源筛选，并可隐藏非标准口径（模型格里带 .lb-flag 的灰色行）。
// 每次筛选后按页面说明里的同一条规则重新加粗：同一基准内、可见的非灰色行、至少两个数值时取最优。
(function () {
  'use strict';

  var english = document.documentElement.lang === 'en';
  function label(zh, en) { return english ? en : zh; }

  var FILTERS = [
    { key: 'paradigm', label: label('范式', 'Paradigm'), opts: [['all', label('全部', 'All')], ['trained', label('训练', 'Trained')], ['training-free', label('免训练', 'Training-free')]] },
    { key: 'input', label: label('输入', 'Input'), opts: [['all', label('全部', 'All')], ['mono', label('单目', 'Monocular')], ['multi', label('多目 / 全景', 'Multi-view / panoramic')]] },
    { key: 'open', label: label('开源', 'Open source'), opts: [['all', label('全部', 'All')], ['yes', label('仅开源', 'Open source only')]] },
    { key: 'nonstd', label: label('非标准口径', 'Nonstandard evaluation'), opts: [['show', label('显示', 'Show')], ['hide', label('隐藏', 'Hide')]] }
  ];
  var MULTI_KEYS = ['全景', '多目', '三相机', '四视角', '180°', 'panoramic', 'multi-view', 'three cameras', 'four views'];
  var COLUMNS = { Model: '模型', Paradigm: '范式', Open: '开源', Benchmark: '基准' };
  var METRICS = { SR: 'max', SPL: 'max', NE: 'min', OSR: 'max' };
  var FOLLOWING = 4; // Node.DOCUMENT_POSITION_FOLLOWING

  var state = { paradigm: 'all', input: 'all', open: 'all', nonstd: 'show' };
  var tables = [];
  var countEl = null;

  function text(el) {
    return (el ? el.textContent : '').replace(/\s+/g, ' ').trim();
  }

  function num(s) {
    return /^-?\d+(\.\d+)?$/.test(s) ? parseFloat(s) : null;
  }

  function after(a, b) {
    return (a.compareDocumentPosition(b) & FOLLOWING) !== 0;
  }

  function findTables(bar) {
    var nextH1 = null;
    var h1s = document.querySelectorAll('h1');
    for (var i = 0; i < h1s.length; i++) {
      if (after(bar, h1s[i])) { nextH1 = h1s[i]; break; }
    }
    var out = [];
    document.querySelectorAll('table').forEach(function (t) {
      if (!after(bar, t) || (nextH1 && !after(t, nextH1))) return;
      var first = t.querySelector('thead th');
      if (first && (text(first) === '模型' || text(first) === 'Model')) out.push(t);
    });
    return out;
  }

  function parseTable(tbl) {
    var cols = {};
    tbl.querySelectorAll('thead th').forEach(function (th, i) {
      var key = text(th).split(' ')[0];
      cols[COLUMNS[key] || key] = i;
    });
    var ncol = tbl.querySelectorAll('thead th').length;
    var rows = [];
    tbl.querySelectorAll('tbody tr').forEach(function (tr) {
      var tds = tr.children;
      var modelCell = tds[cols['模型']];
      var flag = modelCell ? modelCell.querySelector('.lb-flag') : null;
      var name = text(modelCell);
      if (flag) name = name.replace(text(flag), '');
      var input = 'unknown';
      if (MULTI_KEYS.some(function (k) { return name.toLowerCase().indexOf(k) !== -1; })) input = 'multi';
      else if (name.indexOf('单目') !== -1 || name.toLowerCase().indexOf('monocular') !== -1) input = 'mono';
      if (flag) tr.classList.add('lb-nonstd');
      var metrics = {};
      Object.keys(METRICS).forEach(function (m) {
        if (cols[m] == null) return;
        var td = tds[cols[m]];
        td.setAttribute('data-v', text(td));
        metrics[m] = td;
      });
      rows.push({
        tr: tr,
        flag: !!flag,
        input: input,
        paradigm: cols['范式'] != null ? ({'训练': 'trained', '免训练': 'training-free', 'Trained': 'trained', 'Training-free': 'training-free'}[text(tds[cols['范式']])] || '') : '',
        open: cols['开源'] != null && /^(是|Yes)/.test(text(tds[cols['开源']])),
        group: cols['基准'] != null ? text(tds[cols['基准']]) : '',
        metrics: metrics
      });
    });
    var empty = document.createElement('tr');
    empty.className = 'lb-empty';
    var cell = document.createElement('td');
    cell.colSpan = ncol;
    cell.textContent = label('没有符合当前筛选条件的行', 'No rows match the selected filters');
    empty.appendChild(cell);
    empty.style.display = 'none';
    tbl.querySelector('tbody').appendChild(empty);
    return { rows: rows, empty: empty };
  }

  function visible(r) {
    if (state.paradigm !== 'all' && r.paradigm !== state.paradigm) return false;
    if (state.input !== 'all' && r.input !== state.input) return false;
    if (state.open === 'yes' && !r.open) return false;
    if (state.nonstd === 'hide' && r.flag) return false;
    return true;
  }

  function rebold(rows) {
    var shown = rows.filter(function (r) { return r.visible; });
    Object.keys(METRICS).forEach(function (m) {
      var groups = {};
      shown.forEach(function (r) {
        if (!r.metrics[m]) return;
        (groups[r.group] = groups[r.group] || []).push(r);
      });
      rows.forEach(function (r) {
        if (r.metrics[m]) r.metrics[m].textContent = r.metrics[m].getAttribute('data-v');
      });
      Object.keys(groups).forEach(function (g) {
        var cand = groups[g].filter(function (r) {
          return !r.flag && num(r.metrics[m].getAttribute('data-v')) !== null;
        });
        if (cand.length < 2) return;
        var vals = cand.map(function (r) { return num(r.metrics[m].getAttribute('data-v')); });
        var best = METRICS[m] === 'min' ? Math.min.apply(null, vals) : Math.max.apply(null, vals);
        cand.forEach(function (r, i) {
          if (vals[i] !== best) return;
          var strong = document.createElement('strong');
          strong.textContent = r.metrics[m].getAttribute('data-v');
          r.metrics[m].textContent = '';
          r.metrics[m].appendChild(strong);
        });
      });
    });
  }

  function update(bar) {
    bar.querySelectorAll('.filter-btn').forEach(function (btn) {
      btn.classList.toggle('active', state[btn.getAttribute('data-key')] === btn.getAttribute('data-val'));
    });
    var shownAll = 0, totalAll = 0;
    tables.forEach(function (t) {
      var shown = 0;
      t.rows.forEach(function (r) {
        r.visible = visible(r);
        r.tr.classList.toggle('lb-hidden', !r.visible);
        if (r.visible) shown++;
      });
      t.empty.style.display = shown ? 'none' : '';
      // 可见行里基准换组处画粗线，筛选隐藏行后分隔线随之移到新的组首行
      var prev = null;
      t.rows.forEach(function (r) {
        var start = r.visible && prev !== null && r.group !== prev;
        r.tr.classList.toggle('lb-group-start', start);
        if (r.visible) prev = r.group;
      });
      shownAll += shown;
      totalAll += t.rows.length;
      rebold(t.rows);
    });
    countEl.textContent = english ? 'Showing ' + shownAll + ' / ' + totalAll + ' rows' : '显示 ' + shownAll + ' / ' + totalAll + ' 行';
  }

  function build(bar) {
    FILTERS.forEach(function (f) {
      var group = document.createElement('span');
      group.className = 'lb-group';
      var label = document.createElement('span');
      label.className = 'filter-label';
      label.textContent = f.label + (english ? ': ' : '：');
      group.appendChild(label);
      f.opts.forEach(function (o) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'filter-btn';
        btn.setAttribute('data-key', f.key);
        btn.setAttribute('data-val', o[0]);
        btn.textContent = o[1];
        btn.addEventListener('click', function () {
          state[f.key] = o[0];
          update(bar);
        });
        group.appendChild(btn);
      });
      bar.appendChild(group);
    });
    countEl = document.createElement('span');
    countEl.className = 'filter-count';
    bar.appendChild(countEl);
  }

  function init() {
    var bar = document.getElementById('lb-filter-bar');
    if (!bar || bar.getAttribute('data-ready')) return;
    bar.setAttribute('data-ready', '1');
    tables = findTables(bar).map(parseTable);
    if (!tables.length) return;
    build(bar);
    update(bar);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
