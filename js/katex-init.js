// 用 KaTeX 渲染公式。分隔符沿用原 MathJax 配置：$..$、\(..\) 为行内，$$..$$、\[..\] 为独立公式。
// head.html 以 defer 方式依次加载 katex.min.js、auto-render.min.js，再加载本文件；
// defer 脚本在 DOM 解析完成后、DOMContentLoaded 之前按顺序执行，所以这里可以直接渲染，
// 而且先于 article-ui.js 等后加载的脚本。库没加载成功时公式保持原始 TeX 文本。
(function () {
  if (typeof renderMathInElement !== 'function') return;

  renderMathInElement(document.body, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '\\[', right: '\\]', display: true },
      { left: '\\(', right: '\\)', display: false },
      { left: '$', right: '$', display: false }
    ],
    // 解析失败的公式显示为红色源码而不是整页中断；strict 关掉后，公式里出现中文等
    // 非 LaTeX 字符时不再往控制台刷警告（渲染结果不变）。
    throwOnError: false,
    strict: false
  });
})();
