// Verify that the accuracy range stays together at desktop and phone widths.
// Usage: node scripts/tests/check_label_smoothing.cjs [site base URL]
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage();
    await page.route(/google-analytics|googletagmanager|events.vercount|api.github.com|buttons.github.io|use.typekit.net/, r => r.abort());
    const base = process.argv[2] || 'http://127.0.0.1:4173';
    for (const route of ['/en/Deep-Learning-Survey/', '/Deep-Learning-Survey/']) {
      await page.goto(base + route, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise);
      await page.evaluate(() => MathJax.startup.promise);
      const row = page.locator('.entry li').filter({ hasText: route.startsWith('/en/') ? 'Slightly improved generalization' : '轻微提升泛化' });
      assert.equal(await row.count(), 1);
      assert.equal(await row.locator('mjx-container').count(), 0, 'Accuracy range must not split into separate formulas');
      const range = row.locator('span').filter({ hasText: '0.2–0.5%' });
      assert.equal(await range.count(), 1);
      for (const width of [1440, 390, 320]) {
        await page.setViewportSize({ width, height: 844 });
        await row.scrollIntoViewIfNeeded();
        const metrics = await range.evaluate(n => {
          const text = document.createRange();
          text.selectNodeContents(n);
          const lines = [...text.getClientRects()].map(r => r.top);
          return { lines: new Set(lines).size, right: n.getBoundingClientRect().right };
        });
        assert.equal(metrics.lines, 1, `${route} ${width}px: percentage range wraps`);
        assert(metrics.right <= width + 1, `${route} ${width}px: percentage range overflows`);
        console.log(`${route} ${width}px: 0.2–0.5% stays on one line`);
      }
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
