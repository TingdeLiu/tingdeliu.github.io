import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_english_site import Document, research_pair_errors


class StaleResearchTranslationTests(unittest.TestCase):
    def document(self, directory, name, slug, heading, mode, stale=False):
        path = Path(directory) / name
        notice = '<aside class="translation-stale">Source changed</aside>' if stale else ''
        path.write_text(
            '<html><head>'
            f'<link rel="alternate" hreflang="en" href="https://tingdeliu.github.io/en/{slug}/">'
            f'<link rel="alternate" hreflang="zh-CN" href="https://tingdeliu.github.io/{slug}/">'
            '</head><body>'
            f'{notice}<a data-language-switch="{mode}" href="/">Switch</a>'
            f'<div class="entry"><h2 id="{heading}">Heading</h2><table><tr><td>Value</td></tr></table></div>'
            '</body></html>', encoding='utf-8')
        return Document(path)

    def test_stale_pair_keeps_safe_switch_without_requiring_content_parity(self):
        slug = 'Embodied-Agent-Papers'
        with tempfile.TemporaryDirectory() as directory:
            english = self.document(directory, 'en.html', slug, 'old', 'index', stale=True)
            chinese = self.document(directory, 'zh.html', slug, 'new', 'index')
            errors = research_pair_errors(slug, {'stale': True}, english, chinese,
                                          '## New\n| 2 |\n', '## Old\n| 1 |\n')
        self.assertEqual(errors, [])

    def test_stale_pair_still_requires_notice_and_safe_switch(self):
        slug = 'Embodied-Agent-Papers'
        with tempfile.TemporaryDirectory() as directory:
            english = self.document(directory, 'en.html', slug, 'old', 'article')
            chinese = self.document(directory, 'zh.html', slug, 'new', 'article')
            errors = research_pair_errors(slug, {'stale': True}, english, chinese, '', '')
        self.assertIn(f'{slug}: stale translation notice missing', errors)
        self.assertEqual(errors.count(f'{slug}: unsafe section switching'), 2)

    def test_synchronized_pair_still_requires_content_parity(self):
        slug = 'Embodied-Agent-Papers'
        with tempfile.TemporaryDirectory() as directory:
            english = self.document(directory, 'en.html', slug, 'old', 'article')
            chinese = self.document(directory, 'zh.html', slug, 'new', 'article')
            errors = research_pair_errors(slug, {'stale': False}, english, chinese, '', '')
        self.assertIn(f'{slug}: heading structure/IDs differ', errors)


if __name__ == '__main__':
    unittest.main()
