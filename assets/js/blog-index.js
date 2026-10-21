(function () {
  'use strict';
  var library = document.querySelector('[data-blog-library]');
  if (!library) return;
  var cards = Array.from(library.querySelectorAll('[data-blog-card]'));
  var buttons = Array.from(library.querySelectorAll('[data-blog-filter]'));
  var search = library.querySelector('[data-blog-search]');
  var count = library.querySelector('[data-blog-count]');
  var empty = library.querySelector('[data-blog-empty]');
  var english = document.documentElement.lang === 'en';
  var kind = 'all';
  function normalize(text) { return text.normalize('NFKC').toLocaleLowerCase(); }
  var entries = cards.map(function (card) {
    return { card: card, text: normalize(card.dataset.search) };
  });
  function render() {
    var query = normalize(search.value).trim();
    var terms = query ? query.split(/\s+/) : [];
    var visible = 0;
    entries.forEach(function (entry) {
      var matches = (kind === 'all' || entry.card.dataset.kind === kind) &&
        terms.every(function (term) { return entry.text.includes(term); });
      entry.card.hidden = !matches;
      if (matches) visible++;
    });
    buttons.forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.blogFilter === kind));
    });
    count.textContent = english ? visible + (visible === 1 ? ' article' : ' articles') : visible + ' 篇文章';
    empty.hidden = visible > 0;
  }
  function readHash() {
    var candidate = window.location.hash.slice(1);
    kind = buttons.some(function (button) { return button.dataset.blogFilter === candidate; }) ? candidate : 'all';
    render();
  }
  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      kind = button.dataset.blogFilter;
      var url = new URL(window.location.href);
      url.hash = kind === 'all' ? '' : kind;
      window.history.replaceState(null, '', url);
      render();
    });
  });
  search.addEventListener('input', render);
  library.querySelector('[data-blog-reset]').addEventListener('click', function () {
    search.value = '';
    buttons[0].click();
    search.focus();
  });
  window.addEventListener('hashchange', readHash);
  library.querySelector('[data-blog-controls]').hidden = false;
  readHash();
})();
