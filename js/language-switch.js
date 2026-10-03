(function () {
  'use strict';
  // Heading IDs are shared across a translated article and its source.
  // Only carry fragments known to exist on the current page.
  function updateLinks() {
    var hash = window.location.hash;
    var id = '';
    try { id = decodeURIComponent(hash.slice(1)); } catch (_) { hash = ''; }
    if (!id || !document.getElementById(id)) hash = '';
    document.querySelectorAll('[data-language-switch]').forEach(function (link) {
      var target = new URL(link.href);
      target.hash = link.getAttribute('data-language-switch') === 'article' ? hash : '';
      link.href = target.href;
    });
  }
  document.addEventListener('DOMContentLoaded', updateLinks);
  window.addEventListener('hashchange', updateLinks);
  document.addEventListener('click', function (event) {
    if (event.target.closest('[data-language-switch]')) updateLinks();
  });
})();
