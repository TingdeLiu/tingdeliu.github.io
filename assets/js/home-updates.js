// 首页「最近更新」：标出读者上次来访之后修订过的文章。
// 只在本机浏览器里记一个来访日期，不上传任何数据；存储不可用时静默退化为普通列表。
(function () {
  var root = document.querySelector('[data-home-updates]');
  if (!root) return;

  var VISIT_KEY = 'home-updates:last-visit';
  var BASELINE_KEY = 'home-updates:baseline';
  var now = new Date();
  var today = now.getFullYear() + '-' +
    String(now.getMonth() + 1).padStart(2, '0') + '-' +
    String(now.getDate()).padStart(2, '0');

  // 同一次浏览期间（刷新、来回跳转）沿用进入时的基准，标记不会一刷新就消失。
  var baseline;
  try {
    baseline = sessionStorage.getItem(BASELINE_KEY);
    if (baseline === null) {
      baseline = localStorage.getItem(VISIT_KEY) || '';
      sessionStorage.setItem(BASELINE_KEY, baseline);
    }
    localStorage.setItem(VISIT_KEY, today);
  } catch (error) {
    return;
  }
  if (!baseline) return; // 第一次来访，没有可比较的基准

  // 文章日期只精确到天：与上次来访同一天的修订也算作“可能没看过”，宁可多标不漏标。
  var unseen = 0;
  root.querySelectorAll('[data-updated]').forEach(function (row) {
    if (row.getAttribute('data-updated') >= baseline) {
      row.classList.add('is-unseen');
      unseen += 1;
    }
  });
  var counter = root.querySelector('[data-unseen-count]');
  if (unseen && counter) {
    counter.textContent = counter.getAttribute('data-template').replace('{n}', unseen);
    counter.hidden = false;
  }
})();
