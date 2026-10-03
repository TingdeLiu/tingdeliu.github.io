"""Regression checks for content that must survive paper translation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_translation_drafts import features, paper_sections


class TranslationFeaturesTests(unittest.TestCase):
    def test_accessible_caption_does_not_duplicate_equation(self):
        source = '<img src="/images/a.webp" />\n<figcaption>$x$</figcaption>'
        translation = '<img src="/images/a.webp" alt="Value $x$" />\n<figcaption>$x$</figcaption>'
        self.assertEqual(features(source), features(translation))

    def test_table_value_and_formula_changes_are_detected(self):
        source = '| Model | SR |\n|---|---|\n| A | 69.9 |\n\n$$x=1$$'
        self.assertNotEqual(features(source), features(source.replace('69.9', '96.9')))
        self.assertNotEqual(features(source), features(source.replace('x=1', 'x=2')))

    def test_omitted_textual_table_column_is_detected(self):
        source = '| Model | Open source |\n|---|---|\n| A | No |'
        self.assertNotEqual(features(source), features(source.replace('| A | No |', '| A |')))

    def test_translated_mask_labels_preserve_mathematics(self):
        source = r'$$j \text{ 在公共 Trunk} \lor i,j \text{ 在同一 Branch}$$'
        target = r'$$j \text{ is in the shared trunk} \lor i,j \text{ are in the same branch}$$'
        self.assertEqual(features(source), features(target))

    def test_last_paper_excludes_references(self):
        body = '## 1. Paper\n{: id="paper"}\nText\n# References\nOther material'
        self.assertEqual(paper_sections(body), {'paper': '## 1. Paper\n{: id="paper"}\nText'})

    def test_reward_labels_translate_without_hiding_reward_changes(self):
        source = r'$$1.0 \text{若执行 done() 但未成功（误报）}$$'
        target = r'$$1.0 \text{done() called without success (false positive)}$$'
        self.assertEqual(features(source), features(target))
        self.assertNotEqual(features(source), features(target.replace('1.0', '-1.0')))
        self.assertNotEqual(features(source), features(target.replace('without success', 'with success')))


if __name__ == '__main__':
    unittest.main()
