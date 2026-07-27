(function () {
  'use strict';

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    var textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
    return Promise.resolve();
  }

  function directChildList(item) {
    return Array.prototype.find.call(item.children, function (child) {
      return child.tagName === 'UL' || child.tagName === 'OL';
    });
  }

  function enhanceTables() {
    document.querySelectorAll('.entry table').forEach(function (table) {
      if (table.closest('.article-table-scroll')) return;

      var wrapper = document.createElement('div');
      wrapper.className = 'article-table-scroll article-wide';
      wrapper.tabIndex = 0;
      wrapper.setAttribute('role', 'region');
      wrapper.setAttribute('aria-label', '可横向滚动的数据表格');
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  }

  function enhanceCodeBlocks() {
    document.querySelectorAll('.entry pre').forEach(function (pre) {
      if (pre.closest('.article-code-shell') || pre.closest('.mermaid')) return;

      var target = pre.closest('.highlighter-rouge') || pre;
      if (target.closest('.article-code-shell')) return;

      var shell = document.createElement('div');
      shell.className = 'article-code-shell article-wide';
      target.parentNode.insertBefore(shell, target);
      shell.appendChild(target);

      var button = document.createElement('button');
      button.className = 'article-code-copy';
      button.type = 'button';
      button.textContent = '复制';
      button.setAttribute('aria-label', '复制代码');
      shell.appendChild(button);

      button.addEventListener('click', function () {
        var code = pre.querySelector('code');
        copyText(code ? code.innerText : pre.innerText).then(function () {
          button.textContent = '已复制';
          button.classList.add('is-copied');
          window.setTimeout(function () {
            button.textContent = '复制';
            button.classList.remove('is-copied');
          }, 1600);
        });
      });
    });
  }

  function enhanceHeadingAnchors() {
    document.querySelectorAll('.entry h1[id], .entry h2[id], .entry h3[id]').forEach(function (heading) {
      if (heading.querySelector('.heading-anchor')) return;

      var anchor = document.createElement('a');
      anchor.className = 'heading-anchor';
      anchor.href = '#' + heading.id;
      anchor.textContent = '#';
      anchor.setAttribute('aria-label', '复制本节链接');
      anchor.title = '复制本节链接';
      heading.appendChild(anchor);

      anchor.addEventListener('click', function (event) {
        event.preventDefault();
        var url = window.location.origin + window.location.pathname + window.location.search + '#' + heading.id;
        window.history.pushState(null, '', '#' + heading.id);
        heading.scrollIntoView({ behavior: 'smooth', block: 'start' });
        copyText(url).then(function () {
          anchor.classList.add('is-copied');
          anchor.setAttribute('aria-label', '链接已复制');
          window.setTimeout(function () {
            anchor.classList.remove('is-copied');
            anchor.setAttribute('aria-label', '复制本节链接');
          }, 1600);
        });
      });
    });
  }

  function setupLightbox() {
    var dialog = document.getElementById('article-lightbox');
    if (!dialog) return;

    var preview = dialog.querySelector('img');
    var caption = dialog.querySelector('figcaption');
    var closeButton = dialog.querySelector('.article-lightbox-close');

    function closeLightbox() {
      if (typeof dialog.close === 'function' && dialog.open) {
        dialog.close();
      } else {
        dialog.removeAttribute('open');
      }
    }

    function openLightbox(source) {
      preview.src = source.currentSrc || source.src;
      preview.alt = source.alt || '';

      var container = source.closest('figure, div');
      var sourceCaption = container ? container.querySelector('figcaption') : null;
      caption.textContent = sourceCaption ? sourceCaption.textContent.trim() : (source.alt || '');
      caption.hidden = !caption.textContent;

      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
      } else {
        dialog.setAttribute('open', '');
      }
    }

    document.querySelectorAll('.entry img').forEach(function (image) {
      if (image.closest('a')) return;
      image.classList.add('is-zoomable');
      image.tabIndex = 0;
      image.setAttribute('role', 'button');
      image.setAttribute('aria-label', image.alt ? '放大图片：' + image.alt : '放大图片');

      image.addEventListener('click', function () {
        openLightbox(image);
      });
      image.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openLightbox(image);
        }
      });
    });

    closeButton.addEventListener('click', closeLightbox);
    dialog.addEventListener('click', function (event) {
      if (event.target === dialog) closeLightbox();
    });
  }

  function setupProgress() {
    var bar = document.querySelector('.article-progress-bar');
    var article = document.querySelector('.posts');
    if (!bar || !article) return function () {};

    function update() {
      var articleTop = article.getBoundingClientRect().top + window.scrollY;
      var articleEnd = articleTop + article.offsetHeight - window.innerHeight;
      var distance = Math.max(articleEnd - articleTop, 1);
      var progress = Math.min(Math.max((window.scrollY - articleTop) / distance, 0), 1);
      bar.style.transform = 'scaleX(' + progress + ')';
    }

    update();
    return update;
  }

  function setupArticleToc() {
    var toc = document.getElementById('article-toc');
    if (!toc) return function () {};

    var trigger = document.querySelector('.article-toc-trigger');
    var overlay = document.querySelector('.article-toc-overlay');
    var closeButton = toc.querySelector('.article-toc-close');
    var scrollArea = toc.querySelector('.article-toc-scroll');
    var links = Array.prototype.slice.call(toc.querySelectorAll('.article-toc-link'));
    var branches = Array.prototype.slice.call(toc.querySelectorAll('.toc-level-2'));
    var drawerQuery = window.matchMedia('(max-width: 1359px)');
    var linkById = new Map();
    var activeId = '';

    function linkTarget(link) {
      var href = link.getAttribute('href') || '';
      if (href.charAt(0) !== '#') return '';
      try {
        return decodeURIComponent(href.slice(1));
      } catch (error) {
        return href.slice(1);
      }
    }

    links.forEach(function (link) {
      linkById.set(linkTarget(link), link);
    });

    branches.forEach(function (branch, index) {
      var nestedList = directChildList(branch);
      if (!nestedList || !nestedList.querySelector('.toc-level-3')) return;

      nestedList.id = nestedList.id || 'toc-branch-' + index;
      nestedList.hidden = true;

      var toggle = document.createElement('button');
      toggle.className = 'toc-branch-toggle';
      toggle.type = 'button';
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('aria-controls', nestedList.id);
      toggle.setAttribute('aria-label', '展开子目录');
      toggle.innerHTML = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="m7.5 4.5 5.5 5.5-5.5 5.5-1.4-1.4 4.1-4.1-4.1-4.1 1.4-1.4Z"/></svg>';
      branch.insertBefore(toggle, branch.firstChild);

      toggle.addEventListener('click', function () {
        if (branch.classList.contains('is-open')) {
          closeBranch(branch);
        } else {
          openBranch(branch);
        }
      });
    });

    function closeBranch(branch) {
      var nestedList = directChildList(branch);
      var toggle = branch.querySelector(':scope > .toc-branch-toggle');
      branch.classList.remove('is-open');
      if (nestedList) nestedList.hidden = true;
      if (toggle) {
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-label', '展开子目录');
      }
    }

    function openBranch(branch) {
      branches.forEach(function (other) {
        if (other !== branch) closeBranch(other);
      });

      var nestedList = directChildList(branch);
      var toggle = branch.querySelector(':scope > .toc-branch-toggle');
      if (!nestedList || !toggle) return;
      branch.classList.add('is-open');
      nestedList.hidden = false;
      toggle.setAttribute('aria-expanded', 'true');
      toggle.setAttribute('aria-label', '收起子目录');
    }

    function keepActiveVisible(link) {
      if (!scrollArea || !link) return;
      var areaRect = scrollArea.getBoundingClientRect();
      var linkRect = link.getBoundingClientRect();
      var topBoundary = areaRect.top + 12;
      var bottomBoundary = areaRect.bottom - 12;

      if (linkRect.top < topBoundary) {
        scrollArea.scrollBy({ top: linkRect.top - topBoundary, behavior: 'smooth' });
      } else if (linkRect.bottom > bottomBoundary) {
        scrollArea.scrollBy({ top: linkRect.bottom - bottomBoundary, behavior: 'smooth' });
      }
    }

    function activate(id) {
      if (!id || id === activeId) return;
      var link = linkById.get(id);
      if (!link) return;
      activeId = id;

      links.forEach(function (item) {
        item.classList.remove('is-active');
        item.removeAttribute('aria-current');
      });
      link.classList.add('is-active');
      link.setAttribute('aria-current', 'location');

      var item = link.closest('li');
      var branch = null;
      if (item && item.classList.contains('toc-level-2')) {
        branch = item;
      } else if (item) {
        branch = item.closest('.toc-level-2');
      }

      if (branch) {
        openBranch(branch);
      } else {
        branches.forEach(closeBranch);
      }
      keepActiveVisible(link);
    }

    function openDrawer() {
      if (!drawerQuery.matches) return;
      document.body.classList.add('toc-drawer-open');
      trigger.setAttribute('aria-expanded', 'true');
      toc.setAttribute('aria-hidden', 'false');
      closeButton.focus({ preventScroll: true });
    }

    function closeDrawer(options) {
      document.body.classList.remove('toc-drawer-open');
      trigger.setAttribute('aria-expanded', 'false');
      toc.setAttribute('aria-hidden', drawerQuery.matches ? 'true' : 'false');
      if (options && options.restoreFocus) trigger.focus({ preventScroll: true });
    }

    trigger.addEventListener('click', openDrawer);
    overlay.addEventListener('click', function () {
      closeDrawer({ restoreFocus: true });
    });
    closeButton.addEventListener('click', function () {
      closeDrawer({ restoreFocus: true });
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && document.body.classList.contains('toc-drawer-open')) {
        closeDrawer({ restoreFocus: true });
      }
    });
    drawerQuery.addEventListener('change', function () {
      closeDrawer();
    });

    links.forEach(function (link) {
      link.addEventListener('click', function () {
        activate(linkTarget(link));
        closeDrawer();
      });
    });

    var headings = Array.prototype.slice.call(
      document.querySelectorAll('.entry h1[id], .entry h2[id], .entry h3[id]')
    ).filter(function (heading) {
      return linkById.has(heading.id);
    });

    function updateActiveHeading() {
      if (!headings.length) return;
      var threshold = Math.min(window.innerHeight * 0.22, 180);
      var current = headings[0];

      headings.forEach(function (heading) {
        if (heading.getBoundingClientRect().top <= threshold) current = heading;
      });

      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) {
        current = headings[headings.length - 1];
      }
      activate(current.id);
    }

    closeDrawer();
    updateActiveHeading();
    return updateActiveHeading;
  }

  document.addEventListener('DOMContentLoaded', function () {
    enhanceTables();
    enhanceCodeBlocks();
    enhanceHeadingAnchors();
    setupLightbox();

    var updateProgress = setupProgress();
    var updateToc = setupArticleToc();
    var frameRequested = false;

    function updateOnFrame() {
      if (frameRequested) return;
      frameRequested = true;
      window.requestAnimationFrame(function () {
        frameRequested = false;
        updateProgress();
        updateToc();
      });
    }

    window.addEventListener('scroll', updateOnFrame, { passive: true });
    window.addEventListener('resize', updateOnFrame);
    window.addEventListener('load', updateOnFrame);
  });
})();
