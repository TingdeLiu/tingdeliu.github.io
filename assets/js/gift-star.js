(function () {
  'use strict';
  // Gift-box Star button: plays a short animation, opens a panel, and loads the repo's star count on first open.
  var gifts = Array.from(document.querySelectorAll('[data-gift]'));
  if (!gifts.length) return;
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var CACHE_KEY = 'tingde-star-count';
  var countRequested = false;

  function showCount(count) {
    document.querySelectorAll('[data-star-count]').forEach(function (el) {
      el.textContent = '★ ' + count;
      el.hidden = false;
    });
  }

  function loadCount(link) {
    if (countRequested) return;
    countRequested = true;
    var cached = null;
    try { cached = sessionStorage.getItem(CACHE_KEY); } catch (error) { /* Storage may be unavailable. */ }
    if (cached) { showCount(cached); return; }
    var repo = link.pathname.replace(/^\/|\/$/g, '');
    fetch('https://api.github.com/repos/' + repo, { headers: { Accept: 'application/vnd.github+json' } })
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (data) {
        if (!data || typeof data.stargazers_count !== 'number') return;
        try { sessionStorage.setItem(CACHE_KEY, String(data.stargazers_count)); } catch (error) { /* Count still shows. */ }
        showCount(data.stargazers_count);
      })
      .catch(function () { /* Without a count the button still works. */ });
  }

  function close(gift, restoreFocus) {
    var panel = gift.querySelector('[data-gift-panel]');
    if (panel.hidden) return;
    panel.hidden = true;
    gift.querySelector('[data-gift-trigger]').setAttribute('aria-expanded', 'false');
    if (restoreFocus) gift.querySelector('[data-gift-trigger]').focus();
  }

  function open(gift) {
    var trigger = gift.querySelector('[data-gift-trigger]');
    var panel = gift.querySelector('[data-gift-panel]');
    gifts.forEach(function (other) { if (other !== gift) close(other, false); });
    if (!reduce.matches) {
      gift.classList.remove('is-opening');
      void gift.offsetWidth;   // restart the animation on repeated clicks
      gift.classList.add('is-opening');
    }
    panel.hidden = false;
    trigger.setAttribute('aria-expanded', 'true');
    var link = panel.querySelector('.gift-star-link');
    link.focus({ preventScroll: true });
    loadCount(link);
  }

  gifts.forEach(function (gift) {
    var panel = gift.querySelector('[data-gift-panel]');
    gift.querySelector('[data-gift-trigger]').addEventListener('click', function () {
      if (panel.hidden) open(gift); else close(gift, false);
    });
    gift.addEventListener('animationend', function (event) {
      if (event.target.classList.contains('gift-star')) gift.classList.remove('is-opening');
    });
  });

  document.addEventListener('keydown', function (event) {
    if (event.key !== 'Escape') return;
    gifts.forEach(function (gift) {
      if (gift.contains(document.activeElement)) close(gift, true); else close(gift, false);
    });
  });
  document.addEventListener('click', function (event) {
    gifts.forEach(function (gift) { if (!gift.contains(event.target)) close(gift, false); });
  });
})();
