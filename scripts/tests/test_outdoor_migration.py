"""Guard CausalNav placement, legacy links, and cross-page filter indexes."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CN = ROOT / '_posts/research'
EN = ROOT / '_translations/en/research'


class OutdoorMigrationTests(unittest.TestCase):
    def test_single_reading_and_contiguous_numbering(self):
        for path, count in [
            (CN / '2026-01-06-VLN-Papers-ObjectNav.md', 26),
            (EN / 'VLN-Papers-Extended.md', 26),
            (CN / '2026-01-06-VLN-Papers-Outdoor.md', 3),
        ]:
            text = path.read_text(encoding='utf-8')
            numbers = [int(n) for n in re.findall(r'^## (\d+)\. ', text, re.M)]
            self.assertEqual(numbers, list(range(1, count + 1)))
        outdoor = (CN / '2026-01-06-VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        self.assertEqual(outdoor.count('## 2. CausalNav (2026)'), 1)
        for anchor in ['abot-n1', 'canonnav', 'nomad', 'sparsevideonav', 'vl-nav']:
            self.assertIn('#' + anchor + ')', outdoor)

    def test_remote_indexes_follow_destination_headings(self):
        files = [CN / name for name in [
            '2026-01-05-VLN-Papers-InstructionFollowing.md',
            '2026-01-06-VLN-Papers-ObjectNav.md',
            '2026-11-07-VLN-Papers-AgenticNav.md',
        ]] + [EN / name for name in ['VLN-Papers.md', 'VLN-Papers-Extended.md', 'Agentic-Navigation-Papers.md']]
        outdoor = (CN / '2026-01-06-VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        for path in files:
            text = path.read_text(encoding='utf-8')
            entries = re.findall(r'\{ n: "([^"]+)", a: "([^"]+)", url: "([^"]+)"', text)
            causal = [entry for entry in entries if entry[1] == 'causalnav']
            self.assertEqual(causal, [('2. CausalNav (2026)', 'causalnav', '/VLN-Papers-Outdoor/')])
            da_nav = [entry for entry in entries if entry[1] == 'da-nav']
            self.assertEqual(da_nav, [('3. DA-Nav (2026)', 'da-nav', '/VLN-Papers-Outdoor/')])
            target = EN / 'VLN-Papers-Extended.md' if path.parent == EN else CN / '2026-01-06-VLN-Papers-ObjectNav.md'
            goals = target.read_text(encoding='utf-8')
            for title, anchor, url in entries:
                if url.endswith('/VLN-Papers-Extended/'):
                    self.assertIn('## ' + title + '\n{: id="' + anchor + '"}', goals)
            if 'ObjectNav' in path.name or path.name == 'VLN-Papers-Extended.md':
                self.assertNotRegex(text, r'(?m)^## \d+\. CausalNav')
                for anchor in ['causalnav', 'causalnav-精华-12', '1-研究背景问题-11', '2-主要方法创新点-9', '3-核心结果发现-10', '4-局限性-10']:
                    self.assertIn('id="' + anchor + '"', text)
                    self.assertIn('{: id="' + anchor + '"}', outdoor)

    def test_english_reading_is_retained_as_unpublished_draft(self):
        text = (EN / 'VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        self.assertIn('published: false', text)
        self.assertIn('translation_scope: partial', text)
        self.assertIn('## 2. CausalNav (2026)', text)


if __name__ == '__main__':
    unittest.main()
