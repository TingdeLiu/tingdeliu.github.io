// 排行榜筛选：作用于 #lb-filter-bar 之后、下一个一级标题之前的所有「| 模型 | … |」表格。
// 按范式、推理方式（模型格里带 .lb-api 角标即调用闭源 API）、输入配置、是否开源筛选，并可隐藏非标准口径（模型格里带 .lb-flag 的灰色行）。
// 每次筛选后按页面说明里的同一条规则重新加粗：同一基准内、可见的非灰色行、至少两个数值时取最优。
(function () {
  'use strict';

  var english = document.documentElement.lang === 'en';
  function label(zh, en) { return english ? en : zh; }

  var FILTERS = [
    { key: 'paradigm', label: label('范式', 'Paradigm'), opts: [['all', label('全部', 'All')], ['trained', label('训练', 'Trained')], ['training-free', label('免训练', 'Training-free')]] },
    { key: 'api', label: label('推理', 'Inference'), opts: [['all', label('全部', 'All')], ['local', label('仅本地模型', 'Local models only')], ['api', label('调用闭源 API', 'Calls closed API')]] },
    { key: 'input', label: label('输入', 'Input'), opts: [['all', label('全部', 'All')], ['mono', label('单目', 'Monocular')], ['multi', label('多目 / 全景', 'Multi-view / panoramic')]] },
    { key: 'open', label: label('开源', 'Open source'), opts: [['all', label('全部', 'All')], ['yes', label('仅开源', 'Open source only')]] },
    { key: 'nonstd', label: label('非标准口径', 'Nonstandard evaluation'), opts: [['show', label('显示', 'Show')], ['hide', label('隐藏', 'Hide')]] }
  ];
  var MULTI_KEYS = ['全景', '多目', '三相机', '四视角', '180°', 'panoramic', 'multi-view', 'three cameras', 'four views'];
  // 英文版把「基准」列译成了 Baseline，一并识别
  var COLUMNS = { Model: '模型', Year: '年份', Paradigm: '范式', Base: '基模', Open: '开源', Benchmark: '基准', Baseline: '基准', Evaluation: '评测设置' };
  var CHART_METRICS = ['SR', 'SPL', 'NE', 'ISR', 'CSR', 'CGT'];
  var CHART_TOP = 20;
  // 分组表（带「基准」或「评测设置」列）只给行数不少于此值的组画图，且至少要有两个这样的组
  var GROUP_MIN = 5;
  var METRICS = { SR: 'max', SPL: 'max', NE: 'min', OSR: 'max', ISR: 'max', CSR: 'max', CGT: 'max' };
  var FOLLOWING = 4; // Node.DOCUMENT_POSITION_FOLLOWING

  var state = { paradigm: 'all', api: 'all', input: 'all', open: 'all', nonstd: 'show' };
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
      var api = modelCell ? modelCell.querySelector('.lb-api') : null;
      var name = text(modelCell);
      if (flag) name = name.replace(text(flag), '');
      if (api) name = name.replace(text(api), '');
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
      var link = modelCell ? modelCell.querySelector('a') : null;
      var openLink = cols['开源'] != null ? tds[cols['开源']].querySelector('a') : null;
      rows.push({
        tr: tr,
        name: name.replace(/\s+/g, ' ').trim(),
        api: !!api,
        href: link ? link.getAttribute('href') : '',
        flagText: text(flag),
        year: cols['年份'] != null ? text(tds[cols['年份']]) : '',
        paradigmText: cols['范式'] != null ? text(tds[cols['范式']]) : '',
        base: cols['基模'] != null ? text(tds[cols['基模']]) : '',
        openText: cols['开源'] != null ? text(tds[cols['开源']]) : '',
        openHref: openLink ? openLink.getAttribute('href') : '',
        flag: !!flag,
        input: input,
        paradigm: cols['范式'] != null ? ({'训练': 'trained', '免训练': 'training-free', 'Trained': 'trained', 'Training-free': 'training-free'}[text(tds[cols['范式']])] || '') : '',
        open: cols['开源'] != null && /^(是|Yes)/.test(text(tds[cols['开源']])),
        group: cols['评测设置'] != null ? text(tds[cols['评测设置']]) : (cols['基准'] != null ? text(tds[cols['基准']]) : ''),
        evaluationSetting: cols['评测设置'] != null,
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
    var t = { tbl: tbl, rows: rows, empty: empty, chart: null };
    // 单一基准的表直接画图；带「基准」或「评测设置」列的表混了多种口径，按组切换着画，每次只比同一组
    if (cols.SR != null) {
      if (cols['基准'] == null && cols['评测设置'] == null) buildChart(t, null);
      else {
        var groups = chartGroups(rows);
        if (groups.length >= 2) buildChart(t, groups);
      }
    }
    return t;
  }

  function chartGroups(rows) {
    var count = {}, order = [];
    rows.forEach(function (r) {
      if (!(r.group in count)) { count[r.group] = 0; order.push(r.group); }
      count[r.group]++;
    });
    return order.filter(function (g) { return count[g] >= GROUP_MIN; });
  }

  // ---- 条形图视图：数据直接取自上面的表格行，表格仍是唯一数据源 ----

  function el(tag, cls, txt) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  }

  function tableBox(t) {
    return t.tbl.closest('.article-table-scroll') || t.tbl;
  }

  function buildChart(t, groups) {
    var c = { view: 'chart', metric: 'SR', expanded: false, open: null, groups: groups, group: groups ? groups[0] : null };
    var box = el('div', 'lb-chart');
    var tools = el('div', 'lb-chart-tools');
    var views = el('span', 'lb-chart-seg');
    [['chart', label('图', 'Chart')], ['table', label('表', 'Table')]].forEach(function (v) {
      var b = el('button', 'filter-btn', v[1]);
      b.type = 'button';
      b.setAttribute('data-view', v[0]);
      b.addEventListener('click', function () { setView(t, v[0]); });
      views.appendChild(b);
    });
    tools.appendChild(views);
    if (groups) {
      var gseg = el('span', 'lb-chart-seg lb-chart-groups');
      gseg.appendChild(el('span', 'filter-label', t.rows[0].evaluationSetting ? label('评测设置：', 'Setting: ') : label('基准：', 'Benchmark: ')));
      groups.forEach(function (g) {
        var b = el('button', 'filter-btn', g);
        b.type = 'button';
        b.setAttribute('data-group', g);
        b.addEventListener('click', function () { c.group = g; c.open = null; renderChart(t); });
        gseg.appendChild(b);
      });
      tools.appendChild(gseg);
    }
    var sorts = el('span', 'lb-chart-seg lb-chart-sort');
    sorts.appendChild(el('span', 'filter-label', label('排序：', 'Sort: ')));
    CHART_METRICS.forEach(function (m) {
      if (!t.rows.some(function (r) { return r.metrics[m]; })) return;
      var b = el('button', 'filter-btn', m + (METRICS[m] === 'min' ? ' ↓' : ' ↑'));
      b.type = 'button';
      b.setAttribute('data-metric', m);
      b.addEventListener('click', function () { c.metric = m; renderChart(t); });
      sorts.appendChild(b);
    });
    tools.appendChild(sorts);
    var legend = el('span', 'lb-chart-legend');
    [['local', label('本地模型', 'Local model')], ['api', label('调用闭源 API', 'Calls closed API')], ['nonstd', label('非标准口径', 'Nonstandard')]].forEach(function (l) {
      var s = el('span', 'lb-legend-item');
      s.appendChild(el('span', 'lb-swatch lb-c-' + l[0]));
      s.appendChild(document.createTextNode(l[1]));
      legend.appendChild(s);
    });
    tools.appendChild(legend);
    box.appendChild(tools);
    c.list = el('div', 'lb-chart-list');
    box.appendChild(c.list);
    c.foot = el('div', 'lb-chart-foot');
    box.appendChild(c.foot);
    c.box = box;
    t.chart = c;
    var anchor = tableBox(t);
    anchor.parentNode.insertBefore(box, anchor);
  }

  function setView(t, view) {
    t.chart.view = view;
    renderChart(t);
  }

  function renderChart(t) {
    var c = t.chart;
    if (!c) return;
    // 图视图下由 CSS 的相邻选择器隐藏紧随其后的表格；article-ui.js 之后才套的滚动外壳也同样命中
    c.box.classList.toggle('is-chart', c.view === 'chart');
    c.box.querySelectorAll('[data-view]').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-view') === c.view);
    });
    c.box.querySelectorAll('[data-metric]').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-metric') === c.metric);
    });
    c.box.querySelectorAll('[data-group]').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-group') === c.group);
    });
    c.box.querySelectorAll('.lb-chart-sort, .lb-chart-groups').forEach(function (s) {
      s.style.display = c.view === 'chart' ? '' : 'none';
    });
    c.box.querySelector('.lb-chart-legend').style.display = c.view === 'chart' ? '' : 'none';
    c.list.style.display = c.foot.style.display = c.view === 'chart' ? '' : 'none';
    if (c.view !== 'chart') return;

    var m = c.metric, asc = METRICS[m] === 'min';
    var shown = t.rows.filter(function (r) { return r.visible && (!c.groups || r.group === c.group); });
    var data = [], missing = 0;
    shown.forEach(function (r) {
      var v = r.metrics[m] ? num(r.metrics[m].getAttribute('data-v')) : null;
      if (v === null) missing++;
      else data.push({ r: r, v: v });
    });
    // 稳定排序：同分保持表格原顺序
    data.forEach(function (d, i) { d.i = i; });
    data.sort(function (a, b) { return (asc ? a.v - b.v : b.v - a.v) || a.i - b.i; });
    var max = Math.max.apply(null, data.map(function (d) { return d.v; }).concat([0]));
    var limit = c.expanded ? data.length : Math.min(CHART_TOP, data.length);

    c.list.innerHTML = '';
    // 同分同名次（如 LHPR 里一排 0），下一个不同的分数跳过并列的名次
    var rank = 0, seen = 0, prev = null;
    data.forEach(function (d, i) {
      if (!d.r.flag) {
        seen++;
        if (d.v !== prev) { rank = seen; prev = d.v; }
      }
      if (i >= limit) return;
      var r = d.r;
      var row = el('button', 'lb-bar-row' + (c.open === r ? ' is-open' : ''));
      row.type = 'button';
      row.setAttribute('aria-expanded', c.open === r ? 'true' : 'false');
      row.appendChild(el('span', 'lb-bar-rank', r.flag ? '·' : String(rank)));
      var nm = el('span', 'lb-bar-name', r.name);
      nm.title = r.name + (r.flagText ? ' · ' + r.flagText : '');
      // 颜色之外再给调用 API 的行一个文字标记，不只靠颜色区分
      if (r.api) nm.appendChild(el('span', 'lb-bar-api', 'API'));
      row.appendChild(nm);
      var track = el('span', 'lb-bar-track');
      var bar = el('span', 'lb-bar lb-c-' + (r.flag ? 'nonstd' : (r.api ? 'api' : 'local')));
      bar.style.width = (max ? d.v / max * 100 : 0).toFixed(1) + '%';
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(el('span', 'lb-bar-value', r.metrics[m].getAttribute('data-v')));
      row.addEventListener('click', function () {
        c.open = c.open === r ? null : r;
        renderChart(t);
      });
      c.list.appendChild(row);
      if (c.open === r) c.list.appendChild(detail(t, r));
    });
    if (!data.length) c.list.appendChild(el('div', 'lb-chart-empty', label('没有符合当前筛选条件的行', 'No rows match the selected filters')));

    c.foot.innerHTML = '';
    if (data.length > CHART_TOP) {
      var more = el('button', 'filter-btn', c.expanded
        ? label('收起', 'Show top ' + CHART_TOP)
        : label('展开全部 ' + data.length + ' 行', 'Show all ' + data.length + ' rows'));
      more.type = 'button';
      more.addEventListener('click', function () { c.expanded = !c.expanded; renderChart(t); });
      c.foot.appendChild(more);
    }
    var notes = [];
    if (asc) notes.push(label(m + ' 越小越好，条越短越好', 'Lower ' + m + ' is better; shorter bars are better'));
    if (missing) notes.push(label(missing + ' 行未报告 ' + m, missing + ' rows do not report ' + m));
    notes.push(label('灰色条纹为非标准口径，不参与名次', 'Hatched bars use nonstandard evaluation and are not ranked'));
    c.foot.appendChild(el('span', 'lb-chart-note', notes.join(label('；', '; '))));
  }

  function detail(t, r) {
    var card = el('div', 'lb-detail');
    var grid = el('div', 'lb-detail-metrics');
    Object.keys(METRICS).forEach(function (m) {
      if (!r.metrics[m]) return;
      var cell = el('div', 'lb-detail-metric');
      cell.appendChild(el('span', '', m + (METRICS[m] === 'min' ? ' ↓' : ' ↑')));
      cell.appendChild(el('b', '', r.metrics[m].getAttribute('data-v')));
      grid.appendChild(cell);
    });
    card.appendChild(grid);
    var meta = [r.paradigmText, r.api ? label('推理时调用闭源 API', 'Calls a closed API at inference') : label('本地模型推理', 'Local-model inference'), r.base && r.base !== '–' ? label('基模 ', 'Base model ') + r.base : '', r.year].filter(Boolean);
    var info = el('div', 'lb-detail-info', meta.join(' · '));
    if (r.openText) {
      info.appendChild(document.createTextNode(' · ' + label('开源 ', 'Open source ')));
      if (r.openHref) {
        var a = el('a', '', r.openText);
        a.href = r.openHref;
        info.appendChild(a);
      } else {
        info.appendChild(document.createTextNode(r.openText));
      }
    }
    card.appendChild(info);
    if (r.flag) card.appendChild(el('div', 'lb-detail-flag', r.flagText + label('：非标准口径，不参与加粗与名次，与全量行的高低只作参考', ': nonstandard evaluation, excluded from bolding and ranking; compare with full-set rows only as a reference')));
    var links = el('div', 'lb-detail-links');
    if (r.href) {
      var go = el('a', '', label('精读章节 →', 'Paper notes →'));
      go.href = r.href;
      links.appendChild(go);
    }
    var locate = el('button', 'lb-link-btn', label('在表格中定位', 'Find in table'));
    locate.type = 'button';
    locate.addEventListener('click', function () {
      setView(t, 'table');
      r.tr.scrollIntoView({ block: 'center' });
      r.tr.classList.remove('lb-flash');
      void r.tr.offsetWidth;
      r.tr.classList.add('lb-flash');
    });
    links.appendChild(locate);
    card.appendChild(links);
    return card;
  }

  function visible(r) {
    if (state.paradigm !== 'all' && r.paradigm !== state.paradigm) return false;
    if (state.api !== 'all' && r.api !== (state.api === 'api')) return false;
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
        if (cand[0].evaluationSetting && METRICS[m] === 'max' && best === 0) return;
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
      renderChart(t);
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
