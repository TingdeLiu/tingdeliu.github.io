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
            (CN / '2026-01-06-VLN-Papers-Outdoor.md', 8),
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
            for title, anchor in [('4. CityWalker (2025)', 'citywalker'), ('5. UrbanNav (2026)', 'urbannav'), ('6. MOSU2 (2026)', 'mosu2'), ('7. LM-Nav (2022)', 'lm-nav')]:
                self.assertEqual([entry for entry in entries if entry[1] == anchor], [(title, anchor, '/VLN-Papers-Outdoor/')])
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

    def test_outdoor_batch_assets_and_no_duplicate_body_toc(self):
        text = (CN / '2026-01-06-VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        self.assertIn('toc: true', text)
        self.assertNotIn('{:toc}', text)
        for anchor in ['citywalker', 'urbannav', 'mosu2', 'lm-nav']:
            for suffix in ['', '-takeaways', '-background', '-method', '-results', '-limitations']:
                self.assertEqual(text.count('{: id="' + anchor + suffix + '"}'), 1)
        images = re.findall(r'<img src="(/images/(?:vln|robotics_navigation)/(?:CityWalker|UrbanNav|MOSU2|LM-Nav)-[^"]+)"', text)
        self.assertEqual(len(images), 10)
        for image in images:
            self.assertTrue((ROOT / image.lstrip('/')).is_file(), image)
        for block in re.findall(r'<div align="center">.*?</div>', text, re.S):
            self.assertNotIn('\n\n', block)
        self.assertIn('非加权平均', text)
        self.assertIn('两组 Net Success 均为 0.80', text)

    def test_compass_reading_assets_and_evaluation_scope(self):
        text = (CN / '2026-01-06-VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        self.assertEqual(text.count('## 8. Compass 1 (2026)'), 1)
        self.assertIn('收录 8 篇精读', text)
        for suffix in ['', '-takeaways', '-background', '-method', '-results', '-limitations']:
            self.assertEqual(text.count('{: id="compass-1' + suffix + '"}'), 1)
        images = re.findall(r'<img src="(/images/vln/Compass-1-[^"]+)"', text)
        self.assertEqual(len(images), 4)
        for image in images:
            self.assertTrue((ROOT / image.lstrip('/')).is_file(), image)
        self.assertIn('总 SR 79.4% 混合了室内与户外结果', text)
        self.assertIn('8. **Compass 1** (2026).', text)

    def test_complete_english_collection_preserves_source_features(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from check_translation_drafts import features, paper_sections
        text = (EN / 'VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        source = (CN / '2026-01-06-VLN-Papers-Outdoor.md').read_text(encoding='utf-8')
        self.assertNotIn('published: false', text)
        self.assertIn('translation_scope: complete', text)
        self.assertNotIn('{:toc}', text)
        self.assertEqual(re.findall(r'^## (\d+)\. ', text, re.M), list(map(str, range(1, 9))))
        originals, translated = paper_sections(source), paper_sections(text)
        self.assertEqual(originals.keys(), translated.keys())
        for anchor in originals:
            self.assertEqual(features(originals[anchor]), features(translated[anchor]), anchor)
        self.assertEqual(features(source), features(text))
        # Chinese text is retained only in stable heading IDs for section switching.
        prose = re.sub(r'^\{: id="[^"]+"\}\s*$', '', text, flags=re.M)
        self.assertNotRegex(prose, r'[\u4e00-\u9fff]')


if __name__ == '__main__':
    unittest.main()
