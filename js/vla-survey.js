(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    var article = document.querySelector('.vla-survey');
    if (!article) return;
    // Preserve bookmarks to paper sections moved from the survey.
    if (/\/VLA-Survey\/?$/.test(window.location.pathname) && /^#5(?:-\d+|\d{1,2}-)/.test(window.location.hash)) {
      window.location.replace('/VLA-Papers/' + window.location.hash);
      return;
    }
    var index = article.querySelector('.vla-paper-index');
    if (!index) return;
    var query = index.querySelector('input');
    var year = index.querySelector('select');
    var results = index.querySelector('.vla-paper-results');
    var count = index.querySelector('.vla-result-count');
    var empty = index.querySelector('.vla-empty');
    var reset = index.querySelector('.vla-reset');

    function normalize(value) {
      return value.normalize('NFKC').toLowerCase().replace(/π/g, 'pi').replace(/[\s_‐‑–—-]+/g, '');
    }

    // Derive the index from the article so adding a paper needs no separate registry.
    var papers = Array.prototype.filter.call(article.querySelectorAll('.entry > h2[id]'), function (heading) {
      return /^5\.\d+\s/.test(heading.textContent);
    }).map(function (heading) {
      var title = heading.textContent.replace(/\s*#\s*$/, '').trim();
      var body = '';
      var sibling = heading.nextElementSibling;
      while (sibling && !/^H[12]$/.test(sibling.tagName)) {
        body += ' ' + sibling.textContent;
        sibling = sibling.nextElementSibling;
      }
      var years = title.match(/20\d{2}/g) || [];
      var item = document.createElement('li');
      var link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = title;
      item.appendChild(link);
      results.appendChild(item);
      return { item: item, search: normalize(title + ' ' + body), years: years };
    });
    if (!papers.length) return;

    var years = [];
    papers.forEach(function (paper) {
      paper.years.forEach(function (value) {
        if (years.indexOf(value) === -1) years.push(value);
      });
    });
    years.sort().reverse().forEach(function (value) {
      var option = document.createElement('option');
      option.value = option.textContent = value;
      year.appendChild(option);
    });

    function update() {
      var terms = query.value.trim().split(/\s+/).map(normalize).filter(Boolean);
      var shown = 0;
      papers.forEach(function (paper) {
        var matches = (!year.value || paper.years.indexOf(year.value) !== -1) && terms.every(function (term) {
          return paper.search.indexOf(term) !== -1;
        });
        paper.item.hidden = !matches;
        if (matches) shown++;
      });
      count.textContent = '显示 ' + shown + ' / ' + papers.length + ' 篇 · 点击标题跳转';
      empty.hidden = shown > 0;
      reset.hidden = !query.value && !year.value;
      results.scrollTop = 0;
    }
    query.addEventListener('input', update);
    year.addEventListener('change', update);
    reset.addEventListener('click', function () {
      query.value = '';
      year.value = '';
      update();
      query.focus();
    });
    update();
    index.hidden = false;
  });
})();
