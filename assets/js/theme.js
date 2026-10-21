(function () {
  'use strict';

  var root = document.documentElement;
  var key = 'tingde-color-theme';
  var system = window.matchMedia('(prefers-color-scheme: dark)');
  var preference = null;
  try { preference = localStorage.getItem(key); } catch (error) { /* Storage may be unavailable. */ }
  if (preference !== 'light' && preference !== 'dark') preference = null;

  function apply() {
    var theme = preference || (system.matches ? 'dark' : 'light');
    root.setAttribute('data-theme', theme);
    root.classList.add('theme-ready');
    document.querySelectorAll('[data-theme-toggle]').forEach(function (button) {
      var english = button.getAttribute('data-lang') === 'en';
      var dark = theme === 'dark';
      var label = english ? (dark ? 'Switch to light mode' : 'Switch to dark mode') : (dark ? '切换至浅色模式' : '切换至深色模式');
      button.setAttribute('aria-label', label);
      button.setAttribute('title', label);
      button.setAttribute('aria-pressed', String(dark));
      button.querySelector('[data-theme-label]').textContent = english ? (dark ? 'Light' : 'Dark') : (dark ? '浅色' : '深色');
    });
  }

  // Run in the head so a saved preference is applied before the first paint.
  apply();
  document.addEventListener('DOMContentLoaded', function () {
    apply();
    document.querySelectorAll('[data-theme-toggle]').forEach(function (button) {
      button.addEventListener('click', function () {
        preference = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        try { localStorage.setItem(key, preference); } catch (error) { /* Switching still works for this page. */ }
        apply();
      });
    });
  });
  system.addEventListener('change', function () { if (!preference) apply(); });
  window.addEventListener('storage', function (event) {
    if (event.key !== key && event.key !== null) return;
    preference = event.newValue === 'light' || event.newValue === 'dark' ? event.newValue : null;
    apply();
  });
})();
