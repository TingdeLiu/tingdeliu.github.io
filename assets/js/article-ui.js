(function () {
  'use strict';

  var english = document.documentElement.lang === 'en';
  var labels = {
    "可横向滚动的数据表格": "Horizontally scrollable data table",
    "复制": "Copy",
    "复制代码": "Copy code",
    "已复制": "Copied",
    "复制本节链接": "Copy link to this section",
    "链接已复制": "Link copied",
    "放大图片：": "Enlarge image: ",
    "放大图片": "Enlarge image",
    "展开子目录": "Expand subsections",
    "收起子目录": "Collapse subsections"
  };
  function label(value) { return english ? (labels[value] || value) : value; }

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

  function groupComponentMatrix(table) {
    if (table.classList.contains('is-grouped')) return;

    var dataColumns = [3, 4, 10, 11, 12];
    Array.prototype.forEach.call(table.rows, function (row) {
      Array.prototype.forEach.call(row.cells, function (cell, index) {
        if (index < 3) return;
        cell.setAttribute('data-component-group', dataColumns.indexOf(index) !== -1 ? 'data' : 'policy');
      });
    });

    var controls = document.createElement('div');
    controls.className = 'vln-component-switch';
    controls.setAttribute('role', 'group');
    controls.setAttribute('aria-label', english ? 'Compare component groups' : '选择要素分组');
    if (!table.id) table.id = 'vln-component-matrix-' + document.querySelectorAll('.vln-component-switch').length;

    [
      { value: 'policy', name: english ? 'Policy & control' : '决策与控制' },
      { value: 'data', name: english ? 'Data & engineering' : '数据与工程' }
    ].forEach(function (group) {
      var button = document.createElement('button');
      button.type = 'button';
      button.textContent = group.name;
      button.setAttribute('aria-controls', table.id);
      button.setAttribute('aria-pressed', group.value === 'policy' ? 'true' : 'false');
      button.addEventListener('click', function () {
        table.setAttribute('data-component-view', group.value);
        controls.querySelectorAll('button').forEach(function (other) {
          other.setAttribute('aria-pressed', other === button ? 'true' : 'false');
        });
      });
      controls.appendChild(button);
    });

    var hint = document.createElement('span');
    hint.textContent = english ? '5 components per view · rank, model and SR stay visible' : '每组 5 个要素 · 排名、模型和 SR 始终保留';
    controls.appendChild(hint);
    table.parentNode.insertBefore(controls, table);
    table.setAttribute('data-component-view', 'policy');
    table.classList.add('is-grouped');
  }

  function enhanceTables() {
    document.querySelectorAll('.entry table').forEach(function (table) {
      if (table.matches('.vln-component-criteria, .vln-component-matrix')) {
        // These tables fit the reading column instead of using a scroll shell.
        var headings = Array.prototype.map.call(table.querySelectorAll('thead th'), function (th) {
          return th.textContent.trim();
        });
        table.querySelectorAll('tbody tr').forEach(function (row) {
          Array.prototype.forEach.call(row.cells, function (cell, index) {
            cell.setAttribute('data-label', headings[index]);
          });
        });
        table.classList.add('is-labelled');
        if (table.classList.contains('vln-component-matrix')) groupComponentMatrix(table);
        return;
      }
      if (table.closest('.article-table-scroll')) return;

      var fitTable = table.closest('.prose-reading') && table.rows[0] && table.rows[0].cells.length <= 5;
      var wrapper = document.createElement('div');
      wrapper.className = 'article-table-scroll article-wide';
      if (fitTable) {
        wrapper.classList.add('article-table-fit');
      } else {
        wrapper.tabIndex = 0;
      }
      wrapper.setAttribute('role', 'region');
      wrapper.setAttribute('aria-label', fitTable ? (english ? 'Comparison table' : '对比表格') : label('可横向滚动的数据表格'));
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  }

  function enhanceProseLayout() {
    document.querySelectorAll('.prose-reading .entry p').forEach(function (paragraph) {
      // Keep standalone equations on the wide canvas alongside figures and tables.
      var text = paragraph.textContent.trim();
      var mathOnly = /^\\\([\s\S]*\\\)$/.test(text) || /^\\\[[\s\S]*\\\]$/.test(text);
      // A paragraph with prose around one inline formula still reads as prose.
      var renderedMathOnly = paragraph.children.length === 1 &&
        paragraph.firstElementChild.tagName === 'MJX-CONTAINER' &&
        Array.prototype.every.call(paragraph.childNodes, function (node) {
          return node.nodeType !== 3 || !node.textContent.trim();
        });
      if (mathOnly || renderedMathOnly) {
        paragraph.classList.add('article-wide');
      }
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
      button.textContent = label('复制');
      button.setAttribute('aria-label', label('复制代码'));
      shell.appendChild(button);

      button.addEventListener('click', function () {
        var code = pre.querySelector('code');
        copyText(code ? code.innerText : pre.innerText).then(function () {
          button.textContent = label('已复制');
          button.classList.add('is-copied');
          window.setTimeout(function () {
            button.textContent = label('复制');
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
      anchor.setAttribute('aria-label', label('复制本节链接'));
      anchor.title = label('复制本节链接');
      heading.appendChild(anchor);

      anchor.addEventListener('click', function (event) {
        event.preventDefault();
        var url = window.location.origin + window.location.pathname + window.location.search + '#' + heading.id;
        window.history.pushState(null, '', '#' + heading.id);
        heading.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
        copyText(url).then(function () {
          anchor.classList.add('is-copied');
          anchor.setAttribute('aria-label', label('链接已复制'));
          window.setTimeout(function () {
            anchor.classList.remove('is-copied');
            anchor.setAttribute('aria-label', label('复制本节链接'));
          }, 1600);
        });
      });
    });
  }

  function enhancePaperFigures() {
    document.querySelectorAll('.paper-figures .entry img').forEach(function (image) {
      // Keep intentionally small figures below the article's default limit.
      var inlineWidth = image.style.width;
      var requestedWidth = inlineWidth || image.getAttribute('width') || '';
      if (/^\d+(?:\.\d+)?%$/.test(requestedWidth) && parseFloat(requestedWidth) < 80) {
        image.style.setProperty('--paper-figure-width', requestedWidth);
      }
      if (/^\d+(?:\.\d+)?%$/.test(inlineWidth)) {
        image.style.removeProperty('width');
      }

      var filename = (image.getAttribute('src') || '').split('/').pop();
      function setDimensions(width, height) {
        if (!(width > 0 && height > 0)) return;
        var ratio = width / height;
        image.style.setProperty('--paper-figure-natural-width', width + 'px');
        // Fit the height by reducing width too, keeping the image's proportions.
        image.style.setProperty('--paper-figure-height-fit-width', 'calc(75vh * ' + ratio + ')');
        if (ratio < 1.6) return;

        if (/architecture|framework|pipeline/i.test(filename)) {
          image.classList.add('paper-figure-wide');
        } else if (/comparison|ablation|statistics|performance|training|system-overview/i.test(filename)) {
          image.classList.add('paper-figure-large');
        }
      }

      // These are the stored pixel dimensions, reserving space before lazy loading.
      var dimensions = image.style.aspectRatio.split('/').map(Number);
      if (dimensions.length === 2 && dimensions[0] > 16 && dimensions[1] > 16) {
        setDimensions(dimensions[0], dimensions[1]);
      } else if (Number(image.getAttribute('width')) > 0 && Number(image.getAttribute('height')) > 0) {
        setDimensions(Number(image.getAttribute('width')), Number(image.getAttribute('height')));
      } else if (image.complete && image.naturalHeight) {
        setDimensions(image.naturalWidth, image.naturalHeight);
      } else {
        image.addEventListener('load', function () {
          setDimensions(image.naturalWidth, image.naturalHeight);
        }, { once: true });
      }
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
      image.setAttribute('aria-label', image.alt ? label('放大图片：') + image.alt : label('放大图片'));

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

  // Demo videos use data-autoplay + preload="none" instead of autoplay, so a page with
  // dozens of clips downloads only the ones near the viewport, and pauses them off screen.
  function setupViewportVideos() {
    var videos = document.querySelectorAll('video[data-autoplay]');
    if (!videos.length) return;
    function play(video) {
      var started = video.play();
      if (started && started.catch) started.catch(function () {});
    }
    if (!('IntersectionObserver' in window)) {
      Array.prototype.forEach.call(videos, play);
      return;
    }
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) play(entry.target);
        else entry.target.pause();
      });
    }, { rootMargin: '200px 0px' });
    Array.prototype.forEach.call(videos, function (video) { observer.observe(video); });
  }

  function setupProgress() {
    var bar = document.querySelector('.article-progress-bar');
    var article = document.querySelector('.posts');
    var meter = document.querySelector('.article-reading-meter span');
    var percent = document.querySelector('.article-reading-percent');
    if (!bar || !article) return function () {};

    function update() {
      var articleTop = article.getBoundingClientRect().top + window.scrollY;
      var articleEnd = articleTop + article.offsetHeight - window.innerHeight;
      var distance = Math.max(articleEnd - articleTop, 1);
      var progress = Math.min(Math.max((window.scrollY - articleTop) / distance, 0), 1);
      bar.style.transform = 'scaleX(' + progress + ')';
      if (meter) meter.style.transform = 'scaleX(' + progress + ')';
      if (percent) percent.textContent = Math.round(progress * 100) + '%';
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
    var tocList = toc.querySelector('.article-toc-list');
    var items = Array.prototype.slice.call(toc.querySelectorAll('li[class*="toc-level-"]'));
    var drawerQuery = window.matchMedia('(max-width: 1099px)');
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

    function tocLevel(item) {
      var match = /toc-level-(\d)/.exec(item.className);
      return match ? Number(match[1]) : 0;
    }

    // 默认常显前两级；前两级条目太多时只常显第一级。文章可用 toc_expand 固定常显层数
    var topLevel = items.reduce(function (min, item) { return Math.min(min, tocLevel(item)); }, 6);
    var outlineCount = items.filter(function (item) { return tocLevel(item) <= topLevel + 1; }).length;
    var expandSetting = Number(toc.getAttribute('data-toc-expand'));
    var expandDepth = expandSetting >= 1 ? expandSetting : (outlineCount > 15 ? 1 : 2);
    var foldLevel = topLevel + expandDepth - 1;
    if (tocList && foldLevel === topLevel) tocList.classList.add('is-fold-top');

    var branches = items.filter(function (item) {
      var nestedList = directChildList(item);
      return tocLevel(item) >= foldLevel && nestedList && nestedList.querySelector('li');
    });

    branches.forEach(function (branch, index) {
      var nestedList = directChildList(branch);

      nestedList.id = nestedList.id || 'toc-branch-' + index;
      nestedList.hidden = true;

      var toggle = document.createElement('button');
      toggle.className = 'toc-branch-toggle';
      toggle.type = 'button';
      toggle.setAttribute('aria-expanded', 'false');
      toggle.setAttribute('aria-controls', nestedList.id);
      toggle.setAttribute('aria-label', label('展开子目录'));
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
      branches.forEach(function (inner) {
        if (inner !== branch && branch.contains(inner)) closeBranch(inner);
      });
      var nestedList = directChildList(branch);
      var toggle = branch.querySelector(':scope > .toc-branch-toggle');
      branch.classList.remove('is-open');
      if (nestedList) nestedList.hidden = true;
      if (toggle) {
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-label', label('展开子目录'));
      }
    }

    function openBranch(branch) {
      branches.forEach(function (other) {
        if (!other.contains(branch) && other.classList.contains('is-open')) closeBranch(other);
      });

      var nestedList = directChildList(branch);
      var toggle = branch.querySelector(':scope > .toc-branch-toggle');
      if (!nestedList || !toggle) return;
      branch.classList.add('is-open');
      nestedList.hidden = false;
      toggle.setAttribute('aria-expanded', 'true');
      toggle.setAttribute('aria-label', label('收起子目录'));
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

      var chain = branches.filter(function (branch) { return branch.contains(link); });
      branches.forEach(function (branch) {
        if (chain.indexOf(branch) === -1 && branch.classList.contains('is-open')) closeBranch(branch);
      });
      chain.forEach(openBranch);
      keepActiveVisible(link);
    }

    function openDrawer() {
      if (!drawerQuery.matches) return;
      document.body.classList.add('toc-drawer-open');
      trigger.setAttribute('aria-expanded', 'true');
      toc.setAttribute('aria-hidden', 'false');
      toc.inert = false;
      closeButton.focus({ preventScroll: true });
    }

    function closeDrawer(options) {
      document.body.classList.remove('toc-drawer-open');
      trigger.setAttribute('aria-expanded', 'false');
      toc.setAttribute('aria-hidden', drawerQuery.matches ? 'true' : 'false');
      toc.inert = drawerQuery.matches;
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
      if (event.key === 'Tab' && document.body.classList.contains('toc-drawer-open')) {
        var focusable = Array.prototype.filter.call(toc.querySelectorAll('a, button'), function (el) { return el.getClientRects().length > 0; });
        var first = focusable[0];
        var last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
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
    document.body.classList.add('article-ui-ready');
    var entry = document.querySelector('.entry');
    var readingTime = document.querySelector('[data-reading-time]');
    if (entry && readingTime) {
      var prose = entry.textContent;
      var cjk = (prose.match(/[\u3400-\u9fff]/g) || []).length;
      var words = (prose.replace(/[\u3400-\u9fff]/g, ' ').match(/\S+/g) || []).length;
      var minutes = Math.max(1, Math.ceil(cjk / 350 + words / 220));
      readingTime.textContent = minutes + (english ? ' min' : ' 分钟');
    }
    enhanceProseLayout();
    enhanceTables();
    enhanceCodeBlocks();
    enhanceHeadingAnchors();
    enhancePaperFigures();
    setupViewportVideos();
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
