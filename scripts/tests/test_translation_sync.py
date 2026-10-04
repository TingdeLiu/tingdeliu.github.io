import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_translations import compare_sections, section_hashes


class TranslationSyncTests(unittest.TestCase):
    def snapshot(self, body, metadata='title: Example\ndate: 2026-09-29'):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.md'
            path.write_text('---\n' + metadata + '\n---\n' + body, encoding='utf-8')
            return section_hashes(path)

    def test_only_changed_section_is_reported(self):
        before = self.snapshot('# First\nOne\n## Second\nTwo\n')
        after = self.snapshot('# First\nOne\n## Second\nThree\n')
        self.assertEqual(compare_sections(after, before), {'changed': ['Second'], 'added': [], 'removed': []})

    def test_additions_and_deletions_are_reported(self):
        before = self.snapshot('# Removed\nOld\n')
        after = self.snapshot('# Added\nNew\n')
        self.assertEqual(compare_sections(after, before)['added'], ['Added'])
        self.assertEqual(compare_sections(after, before)['removed'], ['Removed'])

    def test_code_headings_do_not_create_sections(self):
        result = self.snapshot('# First\n```python\n# Not a heading\n```\n## Second\nTwo\n')
        self.assertEqual(list(result), ['metadata', 'introduction', 'First', 'Second'])

    def test_semantic_metadata_changes_but_pairing_does_not(self):
        before = self.snapshot('# First\nText\n')
        paired = self.snapshot('# First\nText\n', 'title: Example\ndate: 2026-09-29\nlang: zh-CN\ntranslation_id: example')
        updated = self.snapshot('# First\nText\n', 'title: Changed\ndate: 2026-09-29')
        self.assertEqual(before, paired)
        self.assertEqual(compare_sections(updated, before)['changed'], ['metadata'])

    def test_repeated_labels_are_scoped_to_stable_paper_anchors(self):
        body = '# Papers\n## 1. A {#a}\n### 精华\nOne\n## 2. B {#b}\n### 精华\nTwo\n# References\nEnd\n'
        before = self.snapshot(body)
        after = self.snapshot(body.replace('Two', 'Changed'))
        self.assertEqual(compare_sections(after, before)['changed'], ['paper:b / 精华'])
        self.assertIn('References', before)
        inserted = self.snapshot(body.replace('## 1.', '## 0. New {#new}\n### 精华\nNew\n## 1.'))
        self.assertEqual(compare_sections(inserted, before)['changed'], [])

    def test_repeated_survey_labels_track_their_enclosing_section(self):
        body = '# Models\n## Differential\n### Equations\nOne\n## Ackermann\n### Equations\nTwo\n'
        before = self.snapshot(body)
        after = self.snapshot(body.replace('Two', 'Changed'))
        self.assertEqual(compare_sections(after, before)['changed'],
                         ['section:Models / Ackermann / Equations'])
        inserted = self.snapshot(body.replace('## Ackermann', '## Extra\nText\n## Ackermann'))
        self.assertEqual(compare_sections(inserted, before)['changed'], [])


if __name__ == '__main__':
    unittest.main()
