"""Catch anchor collisions and bilingual mismatches before the Jekyll build."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIRS = [
    ('2026-01-05-VLN-Papers-InstructionFollowing.md', 'VLN-Papers'),
    ('2026-01-06-VLN-Papers-ObjectNav.md', 'VLN-Papers-Extended'),
    ('2026-11-07-VLN-Papers-AgenticNav.md', 'Agentic-Navigation-Papers'),
]


def heading_ids(path):
    result, fence = [], None
    lines = path.read_text(encoding='utf-8').splitlines()
    for index, line in enumerate(lines):
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        heading = re.match(r'^(#{1,6}) (.+)', line)
        if not heading:
            continue
        inline = re.search(r'\{#([^}]+)\}$', heading[2])
        following = re.fullmatch(r'\{: id="([^"]+)"\}', lines[index + 1]) if index + 1 < len(lines) else None
        anchor = inline[1] if inline else following[1] if following else None
        result.append((len(heading[1]), anchor))
    return result


class NavigationHeadingTests(unittest.TestCase):
    def test_navigation_headings_use_block_attribute_lists(self):
        for filename, slug in PAIRS:
            for path in [ROOT / '_posts/research' / filename,
                         ROOT / '_translations/en/research' / (slug + '.md')]:
                with self.subTest(path=path.name):
                    body = path.read_text(encoding='utf-8')
                    self.assertNotRegex(body, r'(?m)^#{1,6} .+ \{#[^}]+\}$')
                    blocks = re.findall(r'(?m)^#{1,6} .+\n\{: id="[^"]+"\}$', body)
                    self.assertEqual(len(blocks), len(heading_ids(path)))

    def test_all_heading_anchors_are_explicit_unique_and_bilingual(self):
        for filename, slug in PAIRS:
            with self.subTest(collection=slug):
                cn = ROOT / '_posts/research' / filename
                en = ROOT / '_translations/en/research' / (slug + '.md')
                source = heading_ids(cn)
                target = heading_ids(en)
                self.assertGreater(len(source), 0)
                self.assertEqual(source, target)
                for path, headings in [(cn, source), (en, target)]:
                    ids = [anchor for _, anchor in headings]
                    self.assertNotIn(None, ids, str(path))
                    html_ids = re.findall(r'<(?:div|span|a|section|nav|table|h[1-6])\b[^<>]*\bid="([^"]+)"',
                                          path.read_text(encoding='utf-8'))
                    all_ids = ids + html_ids
                    self.assertEqual(len(all_ids), len(set(all_ids)), str(path))

    def test_agent_survey_links_all_three_navigation_collections(self):
        for path in [ROOT / '_posts/research/2026-03-05-AI-Agent-Survey.md',
                     ROOT / '_translations/en/research/AI-Agent-Survey.md']:
            with self.subTest(path=path.name):
                body = path.read_text(encoding='utf-8')
                for _, slug in PAIRS:
                    url = 'https://tingdeliu.github.io/' + slug + '/'
                    self.assertEqual(body.count('](' + url + ')'), 1)


if __name__ == '__main__':
    unittest.main()
