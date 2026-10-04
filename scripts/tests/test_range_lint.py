"""Catch the range patterns that caused fragmented math on phone screens."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lint_posts import lint


class RangeLintTests(unittest.TestCase):
    def rules(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'post.md'
            path.write_text('---\nlayout: post\ntitle: Example\ndate: 2026-10-04\n---\n' + body, encoding='utf-8')
            _, warnings = lint(path)
        return [rule for _, rule, _ in warnings]

    def test_split_ranges_include_percent_exponents_and_double_dollars(self):
        for body in [r'$\sim 0.2$–$0.5\%$', r'$10^3$–$10^4$',
                     r'$$k \approx 1.5$$–$$2.0$$', r'$7\times$ 至 $500\times$']:
            self.assertIn('split-math-range', self.rules(body))

    def test_currency_range_is_not_a_formula(self):
        self.assertIn('currency-range', self.rules('$3.80 ~ $4.20'))

    def test_complete_ranges_pass(self):
        self.assertEqual(self.rules(r'$10^3 \text{–} 10^4$'), [])
        self.assertEqual(self.rules('<span style="white-space: nowrap;">USD 3.80–4.20</span>'), [])

    def test_code_and_separate_list_items_are_not_ranges(self):
        self.assertEqual(self.rules('```text\n$1$–$2$\n```\n`$1$–$2$`'), [])
        self.assertEqual(self.rules('$f_x=f_y=f$\n- $(c_x,c_y)$'), [])


if __name__ == '__main__':
    unittest.main()
