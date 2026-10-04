"""Regression checks for content that must survive paper translation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_translation_drafts import features, paper_sections, normalize_agent_table_labels, normalize_ml_labels


class TranslationFeaturesTests(unittest.TestCase):
    def test_ml_magnitudes_detect_hundred_million_mistranslation(self):
        source = '| 128 万 | 1.17 亿 | 15 亿 | 1750 亿 |'
        target = '| 1.28 million | 117 million | 1.5 billion | 175 billion |'
        normalize = lambda body: features(normalize_ml_labels(body))['table_numbers']
        self.assertEqual(normalize(source), normalize(target))
        self.assertNotEqual(normalize(source), normalize(target.replace('175 billion', '1750 billion')))

    def test_ml_descriptive_math_labels_preserve_gate_equations(self):
        source = r'$$r_t = \sigma(W_r x_t) \quad \text{（重置门）}$$'
        target = r'$$r_t = \sigma(W_r x_t) \quad \text{(reset gate)}$$'
        normalize = lambda body: features(normalize_ml_labels(body))['math']
        self.assertEqual(normalize(source), normalize(target))
        self.assertNotEqual(normalize(source), normalize(target.replace('W_r', 'W_z')))

    def test_accessible_caption_does_not_duplicate_equation(self):
        source = '<img src="/images/a.webp" />\n<figcaption>$x$</figcaption>'
        translation = '<img src="/images/a.webp" alt="Value $x$" />\n<figcaption>$x$</figcaption>'
        self.assertEqual(features(source), features(translation))

    def test_table_value_and_formula_changes_are_detected(self):
        source = '| Model | SR |\n|---|---|\n| A | 69.9 |\n\n$$x=1$$'
        self.assertNotEqual(features(source), features(source.replace('69.9', '96.9')))
        self.assertNotEqual(features(source), features(source.replace('x=1', 'x=2')))

    def test_currency_does_not_swallow_prose_or_formulas(self):
        source = '$5/月 VPS\n\n# Heading\n$0.66 / M tokens | $0.435\n\n$x=1$ and $$y=2$$'
        target = '$5/month VPS\n\n# Heading\n$0.66 / M tokens | $0.435\n\n$x=1$ and $$y=2$$'
        self.assertEqual(features(source)['math'], features(target)['math'])
        self.assertEqual(set(features(source)['math']), {'$x=1$', '$$y=2$$'})
        self.assertNotEqual(features(source)['math'], features(target.replace('x=1', 'x=2'))['math'])

    def test_agent_table_dates_and_unit_conversions(self):
        source = '| 2025 年 11 月 | 1,000 万 token | 11.5 万 token |'
        target = '| November 2025 | 10 million tokens | 115,000 tokens |'
        normalize = lambda body: features(normalize_agent_table_labels(body))['table_numbers']
        self.assertEqual(normalize(source), normalize(target))
        self.assertNotEqual(normalize(source), normalize(target.replace('115,000', '11,500')))

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
