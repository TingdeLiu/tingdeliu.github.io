(function () {
  'use strict';
  var library = document.querySelector('[data-research-library]');
  if (!library) return;
  var buttons = Array.from(library.querySelectorAll('[data-research-filter]'));
  var rows = Array.from(library.querySelectorAll('[data-research-row]'));
  var count = library.querySelector('[data-research-count]');
  var english = document.documentElement.lang === 'en';

  function render(candidate) {
    var selected = buttons.some(function (button) {
      return button.dataset.researchFilter === candidate;
    }) ? candidate : 'all';
    var visible = 0;
    rows.forEach(function (row) {
      row.hidden = selected !== 'all' && row.dataset.researchGroup !== selected;
      if (!row.hidden) visible++;
    });
    buttons.forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.dataset.researchFilter === selected));
    });
    count.textContent = english ? visible + (visible === 1 ? ' article' : ' articles') : visible + ' 篇文章';
  }

  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      var selected = button.dataset.researchFilter;
      var url = new URL(window.location.href);
      url.hash = selected === 'all' ? '' : selected;
      window.history.replaceState(null, '', url);
      render(selected);
    });
  });
  window.addEventListener('hashchange', function () { render(window.location.hash.slice(1)); });
  render(window.location.hash.slice(1));
  library.querySelector('[data-research-controls]').hidden = false;
})();
