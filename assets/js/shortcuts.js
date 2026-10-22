(function () {
  'use strict';
  // Single-key navigation: H/P/R/B/A follow the masthead links, "/" focuses the list search.
  function isTyping(target) {
    if (!target) return false;
    var tag = target.tagName;
    return target.isContentEditable || tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  }
  document.addEventListener('keydown', function (event) {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || isTyping(event.target)) return;
    if (document.querySelector('dialog[open]') || document.body.classList.contains('toc-drawer-open')) return;
    var key = event.key.toLowerCase();
    if (key === '/') {
      var search = document.querySelector('[data-blog-search]');
      if (search && search.getClientRects().length) {
        event.preventDefault();
        search.focus();
      }
      return;
    }
    if (event.shiftKey) return;
    var link = document.querySelector('.article-masthead [data-shortcut="' + key + '"]');
    if (!link) return;
    event.preventDefault();
    if (link.getAttribute('aria-current') === 'page' && link.pathname === window.location.pathname) return;
    window.location.href = link.href;
  });
})();
