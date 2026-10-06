#!/usr/bin/env python3
"""Validate the built English release and its Chinese source using stdlib only."""
import re
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from check_translations import read_document

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / '_site'
VOID = set('area base br col embed hr img input link meta param source track wbr'.split())


class Document(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.raw = path.read_text(encoding='utf-8')
        self.stack, self.nodes, self.visible = [], [], []
        self.feed(self.raw)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        inside = any('entry' in n['attrs'].get('class', '').split() for n in self.stack)
        node = dict(tag=tag, attrs=attrs, entry=inside, text='')
        self.nodes.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]['tag'] == tag:
                del self.stack[i:]
                break

    def handle_data(self, text):
        if not any(n['tag'] in ('script', 'style') for n in self.stack):
            self.visible.append(text)

    def select(self, tag, entry=False):
        return [n for n in self.nodes if n['tag'] == tag and (not entry or n['entry'])]

    @property
    def ids(self):
        return [n['attrs']['id'] for n in self.nodes if n['attrs'].get('id')]


def main():
    errors, cache = [], {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    def load(path):
        if path not in cache:
            cache[path] = Document(path)
        return cache[path]

    english_files = sorted((SITE / 'en').rglob('*.html'))
    for draft in (ROOT / '_translations').rglob('*.md'):
        fields, _, _ = read_document(draft)
        if fields.get('published', '').lower() == 'false':
            route = fields['permalink'].strip('/')
            check(not (SITE / route / 'index.html').exists(), f'Unpublished translation leaked into build: {route}')
    check(len(english_files) >= 3, 'Missing English entry pages or survey')
    for route, language, other in [('', 'zh-CN', '/en/'), ('en', 'en', '/')]:
        doc = load(SITE / route / 'index.html')
        check(doc.select('html')[0]['attrs'].get('lang') == language, f'/{route}: wrong homepage language')
        switches = [n for n in doc.select('a') if 'data-language-switch' in n['attrs']]
        check(len(switches) == 1 and switches[0]['attrs'].get('href') == other,
              f'/{route}: missing homepage language switch')
        alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href')
                      for n in doc.select('link') if n['attrs'].get('hreflang')}
        check(alternates == {'en': 'https://tingdeliu.github.io/en/', 'zh-CN': 'https://tingdeliu.github.io/'},
              f'/{route}: incorrect homepage language alternates')
        cards = [n['attrs'].get('href', '') for n in doc.select('a')
                 if 'rc-card' in n['attrs'].get('class', '').split()]
        check(bool(cards) and len(cards) == len(set(cards)), f'/{route}: empty or duplicate homepage cards')
        check(all(url.startswith('/en/') == (language == 'en') for url in cards),
              f'/{route}: homepage cards use the wrong language')
    for route in ('', 'en', 'research', 'blog', 'en/research', 'en/blog', 'home', 'about', 'archive', 'tags'):
        doc = load(SITE / route / 'index.html')
        heading = re.search(r'<header class="page-heading">(.*?)</header>', doc.raw, re.S)
        check(bool(heading) and 'site-language-switch' in heading[1],
              f'/{route}/: language switch missing from content heading')
        switches = [n['attrs'].get('href') for n in doc.select('a') if 'data-language-switch' in n['attrs']]
        other = '/' + (route.removeprefix('en/') if route.startswith('en/') else 'en/' + route) + '/'
        if route in ('home', 'about', 'archive', 'tags'):
            other = '/en/'
        elif route in ('', 'en'):
            other = '/en/' if route == '' else '/'
        check(switches == [other], f'/{route}/: missing or duplicate navigation language switch')
    for path in english_files:
        doc = load(path)
        route = '/' + path.relative_to(SITE).as_posix().removesuffix('index.html')
        check(doc.select('html')[0]['attrs'].get('lang') == 'en', f'{route}: wrong HTML language')
        check(not re.search(r'[\u4e00-\u9fff]', ''.join(doc.visible).replace('中文', '')), f'{route}: untranslated visible Chinese')
        check(len(doc.ids) == len(set(doc.ids)), f'{route}: duplicate IDs')
        canonical = [n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('rel') == 'canonical']
        check(canonical == ['https://tingdeliu.github.io' + route], f'{route}: canonical is not self-referencing')
        for node in doc.select('a') + doc.select('img') + doc.select('script') + doc.select('link'):
            ref = node['attrs'].get('href', node['attrs'].get('src', ''))
            if not ref:
                continue
            url = urlsplit(urljoin('https://tingdeliu.github.io' + route, ref))
            if url.scheme not in ('http', 'https') or url.netloc != 'tingdeliu.github.io':
                continue
            target = SITE / unquote(url.path).lstrip('/')
            if target.is_dir():
                target /= 'index.html'
            elif not target.suffix and target.with_suffix('.html').exists():
                target = target.with_suffix('.html')
            check(target.exists(), f'{route}: missing local target {ref}')
            if target.exists() and target.suffix == '.html' and url.fragment:
                check(unquote(url.fragment) in load(target).ids, f'{route}: missing fragment {ref}')

    en = load(SITE / 'en/VLN-Survey/index.html')
    zh = load(SITE / 'VLN-Survey/index.html')
    en_headings = [n['attrs'].get('id') for n in en.nodes if n['entry'] and re.fullmatch('h[1-6]', n['tag'])]
    zh_headings = [n['attrs'].get('id') for n in zh.nodes if n['entry'] and re.fullmatch('h[1-6]', n['tag'])]
    math_pattern = r'\\\[(.*?)\\\]|\\\((.*?)\\\)'
    def external(doc):
        return Counter(n['attrs'].get('href') for n in doc.select('a', True)
                       if n['attrs'].get('href', '').startswith(('http://', 'https://')))
    status = json.loads((ROOT / '_data/translation_status.json').read_text(encoding='utf-8'))['vln-survey']
    if not status['stale']:
        check(en_headings == zh_headings, 'Translated headings do not preserve source structure/IDs')
        for tag in ('table', 'img', 'pre'):
            check(len(en.select(tag, True)) == len(zh.select(tag, True)), f'{tag}: source/translation structural count differs')
        check(re.findall(math_pattern, en.raw, re.S) == re.findall(math_pattern, zh.raw, re.S), 'Rendered mathematical expressions differ')
        check(external(en) == external(zh), 'External citations differ between source and translation')
    else:
        print('Source changed: skipping source/translation parity until the English edition is synchronized.')
    for slug, identity, source_path in [('VLN-Papers', 'vln-papers', '_posts/research/2026-01-05-VLN-Papers.md'), ('VLN-Papers-Extended', 'vln-papers-extended', '_posts/research/2026-01-06-VLN-Papers-Extended.md')]:
        papers_path = SITE / f'en/{slug}/index.html'
        if papers_path.exists():
            papers, original = load(papers_path), load(SITE / f'{slug}/index.html')
            papers_status = json.loads((ROOT / '_data/translation_status.json').read_text(encoding='utf-8'))[identity]
            if not papers_status['stale']:
                def headings(doc):
                    return [n['attrs'].get('id') for n in doc.nodes if n['entry'] and re.fullmatch('h[1-6]', n['tag'])]
                check(headings(papers) == headings(original), f'{slug}: heading structure/IDs differ')
                for tag in ('table', 'img', 'pre'):
                    check(len(papers.select(tag, True)) == len(original.select(tag, True)), f'{slug}: {tag} counts differ')
                def equations(doc):
                    from check_translation_drafts import normalize_math_labels
                    raw = normalize_math_labels(re.sub(r'<img\b[^>]*>', '', doc.raw))
                    return Counter(re.findall(math_pattern, raw, re.S))
                check(equations(papers) == equations(original), f'{slug}: rendered equations differ')
                check(external(papers) == external(original), f'{slug}: external citations differ')
                from check_translation_drafts import features, paper_sections
                _, _, source_body = read_document(ROOT / source_path)
                _, _, english_body = read_document(ROOT / f'_translations/en/research/{slug}.md')
                originals, readings = paper_sections(source_body), paper_sections(english_body)
                check(set(originals) == set(readings), f'{slug}: incomplete paper set')
                for anchor in originals.keys() & readings.keys():
                    check(features(originals[anchor]) == features(readings[anchor]), f'{slug}: {anchor} source features differ')
                # Include the opening leaderboards and comparison matrix in numeric checks.
                check(features(source_body)['table_numbers'] == features(english_body)['table_numbers'], f'{slug}: table numbers differ')
                check(features(source_body)['table_row_widths'] == features(english_body)['table_row_widths'], f'{slug}: table columns differ')
            for doc in (papers, original):
                switches = [n['attrs']['data-language-switch'] for n in doc.select('a') if 'data-language-switch' in n['attrs']]
                check(switches == ['index' if papers_status['stale'] else 'article'], f'{slug}: unsafe language switching')
                alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('hreflang')}
                check(alternates == {'en': f'https://tingdeliu.github.io/en/{slug}/', 'zh-CN': f'https://tingdeliu.github.io/{slug}/'}, f'{slug}: incorrect language alternates')
            print(f'{slug}: {len(readings) if not papers_status["stale"] else "stale"} readings; {len(papers.select("table", True))} tables; {len(papers.select("img", True))} figures')
    for slug in ('AI-Agent-Survey', 'Embodied-Agent-Harness-Survey', 'Embodied-Agent-Papers', 'Machine-Learning-Survey', 'Deep-Learning-Survey', 'Reinforcement-Learning-Survey', 'LLM-Training-Survey', 'VLM-Survey', 'Spatial-Intelligence-Survey', 'VLA-Survey', 'VLA-Papers', 'World-Models-Survey', 'Python-Engineering-Survey', 'ROS2-Survey'):
        target = ROOT / f'_translations/en/research/{slug}.md'
        check(target.exists(), f'{slug}: English article missing')
        if not target.exists():
            continue
        fields, _, english_body = read_document(target)
        _, _, source_body = read_document(ROOT / fields['source_path'])
        translated = load(SITE / f'en/{slug}/index.html')
        original = load(SITE / f'{slug}/index.html')
        heading_ids = lambda doc: [n['attrs'].get('id') for n in doc.nodes if n['entry'] and re.fullmatch('h[1-6]', n['tag'])]
        check(heading_ids(translated) == heading_ids(original), f'{slug}: heading structure/IDs differ')
        for tag in ('table', 'img', 'pre'):
            check(len(translated.select(tag, True)) == len(original.select(tag, True)), f'{slug}: {tag} counts differ')
        check(external(translated) == external(original), f'{slug}: external citations differ')
        from check_translation_drafts import features, normalize_agent_table_labels, normalize_ml_labels, normalize_large_model_labels, normalize_vla_labels
        normalize = normalize_vla_labels if slug in ('VLA-Survey', 'VLA-Papers', 'World-Models-Survey') else normalize_large_model_labels if slug in ('LLM-Training-Survey', 'VLM-Survey', 'Spatial-Intelligence-Survey') else normalize_ml_labels if 'Learning-Survey' in slug else normalize_agent_table_labels
        before, after = features(normalize(source_body)), features(normalize(english_body))
        for feature in ('math', 'links', 'table_numbers', 'table_row_widths'):
            check(before[feature] == after[feature], f'{slug}: {feature} differ')
        for doc in (translated, original):
            switches = [n['attrs']['data-language-switch'] for n in doc.select('a') if 'data-language-switch' in n['attrs']]
            check(switches == ['article'], f'{slug}: section switching missing')
            alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('hreflang')}
            check(alternates == {'en': f'https://tingdeliu.github.io/en/{slug}/', 'zh-CN': f'https://tingdeliu.github.io/{slug}/'}, f'{slug}: incorrect language alternates')
        check(f'/en/{slug}/' not in (SITE / 'feed.xml').read_text(encoding='utf-8'), f'{slug}: translation leaked into Chinese feed')
        cards = [n['attrs'].get('href') for n in load(SITE / 'en/research/index.html').select('a') if 'rc-card' in n['attrs'].get('class', '').split()]
        check(cards.count(f'/en/{slug}/') == 1, f'{slug}: missing or duplicate Research card')
        print(f'{slug}: {len(heading_ids(translated))} headings; {len(translated.select("table", True))} tables; {len(translated.select("img", True))} images')
    navigation_path = SITE / 'en/Robot-Navigation-Survey/index.html'
    if navigation_path.exists():
        navigation = load(navigation_path)
        original = load(SITE / 'Robot-Navigation-Survey/index.html')
        navigation_status = json.loads((ROOT / '_data/translation_status.json').read_text(encoding='utf-8'))['robot-navigation-survey']
        if not navigation_status['stale']:
            heading_ids = lambda doc: [n['attrs'].get('id') for n in doc.nodes if n['entry'] and re.fullmatch('h[1-6]', n['tag'])]
            check(heading_ids(navigation) == heading_ids(original), 'Robot navigation: heading structure/IDs differ')
            for tag in ('table', 'img', 'video', 'pre'):
                check(len(navigation.select(tag, True)) == len(original.select(tag, True)), f'Robot navigation: {tag} counts differ')
            check(external(navigation) == external(original), 'Robot navigation: external citations differ')
            from check_translation_drafts import features
            _, _, source_body = read_document(ROOT / '_posts/research/2026-02-27-Robot-Navigation-Survey.md')
            _, _, english_body = read_document(ROOT / '_translations/en/research/Robot-Navigation-Survey.md')
            for zh_label, en_label in {
                r'\text{（障碍物格本身）}': r'\text{ (obstacle cell)}',
                r'\text{（内切圆内，必碰撞）}': r'\text{ (inside inscribed radius: collision)}',
                r'\text{（膨胀梯度区）}': r'\text{ (inflation gradient)}',
                'T_{左}': r'T_{left}', 'T_{右}': r'T_{right}',
                'T_{上}': r'T_{up}', 'T_{下}': r'T_{down}',
                r'\text{若 }': r'\text{if }',
                r'\text{否则（退化为单侧更新）}': r'\text{otherwise (one-sided update)}',
            }.items():
                source_body = source_body.replace(zh_label, en_label)
            before, after = features(source_body), features(english_body)
            for feature in ('math', 'links', 'table_numbers', 'table_row_widths'):
                check(before[feature] == after[feature], f'Robot navigation: {feature} differ')
            videos = lambda doc: [n['attrs'].get('src') for n in doc.select('video', True)]
            check(videos(navigation) == videos(original), 'Robot navigation: demonstration videos differ')
        for doc in (navigation, original):
            switches = [n['attrs']['data-language-switch'] for n in doc.select('a') if 'data-language-switch' in n['attrs']]
            check(switches == ['index' if navigation_status['stale'] else 'article'], 'Robot navigation: unsafe section switching')
            alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('hreflang')}
            check(alternates == {'en': 'https://tingdeliu.github.io/en/Robot-Navigation-Survey/', 'zh-CN': 'https://tingdeliu.github.io/Robot-Navigation-Survey/'}, 'Robot navigation: incorrect language alternates')
        check(all('alt' in n['attrs'] for n in navigation.select('img', True)), 'Robot navigation: image alternative text missing')
        check('/en/Robot-Navigation-Survey/' not in (SITE / 'feed.xml').read_text(encoding='utf-8'), 'Robot navigation: translation leaked into Chinese feed')
        cards = [n['attrs'].get('href') for n in load(SITE / 'en/research/index.html').select('a') if 'rc-card' in n['attrs'].get('class', '').split()]
        check(cards.count('/en/Robot-Navigation-Survey/') == 1, 'Robot navigation: missing or duplicate Research card')
        print(f'Robot navigation: {len(navigation.select("table", True))} tables; {len(navigation.select("img", True))} images; {len(navigation.select("video", True))} videos')
    check(('translation-stale' in en.raw) == status['stale'], 'Stale translation notice does not match source status')
    for doc, name in [(en, 'English'), (zh, 'Chinese')]:
        switches = [n['attrs']['data-language-switch'] for n in doc.select('a') if 'data-language-switch' in n['attrs']]
        check(switches == ['index' if status['stale'] else 'article'], f'{name}: unsafe section switching for synchronization status')
        alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('hreflang')}
        check(alternates == {'en': 'https://tingdeliu.github.io/en/VLN-Survey/', 'zh-CN': 'https://tingdeliu.github.io/VLN-Survey/'}, f'{name}: incorrect language alternates')
    check('/en/VLN-Survey/' not in (SITE / 'feed.xml').read_text(encoding='utf-8'), 'Translation leaked into Chinese feed')
    for slug in ('VLN-Papers', 'VLN-Papers-Extended'):
        check(f'/en/{slug}/' not in (SITE / 'feed.xml').read_text(encoding='utf-8'), f'{slug}: translation leaked into Chinese feed')
    weekly_targets = [p for p in sorted((ROOT / '_translations/en/blog').glob('vln-weekly-*.md'))
                      if read_document(p)[0].get('published', '').lower() != 'false']
    if weekly_targets:
        from check_translation_drafts import features
        weekly_urls = []
        def prose_numbers(body):
            from decimal import Decimal
            body = re.sub(r'\{: id="[^"]+"\}', '', body)
            body = re.sub(r'\]\([^)]*\)|https?://\S+', '', body)
            body = re.sub(r'^## .+$', '', body, flags=re.M)
            body = re.sub(r'\bSection \d+\b|第[一二三四五六七八九十]+节|3D|三维', '', body, flags=re.I)
            def scale(m):
                factors = {'万':10000, '亿':100000000, 'thousand':1000, 'million':1000000, 'billion':1000000000}
                return str(Decimal(m[1].replace(',', '')) * factors[m[2].lower()])
            body = re.sub(r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(万|亿|thousand\b|million\b|billion\b)', scale, body, flags=re.I)
            return Counter(Decimal(n.replace(',', '')) for n in re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', body))
        for target in weekly_targets:
            fields, _, english_body = read_document(target)
            _, _, source_body = read_document(ROOT / fields['source_path'])
            route, original_route = fields['permalink'], fields['source_url']
            weekly_urls.append(route)
            translated = load(SITE / route.strip('/') / 'index.html')
            original = load(SITE / original_route.strip('/') / 'index.html')
            check('**' not in ''.join(translated.visible), f'{route}: unrendered emphasis markup')
            def heading_ids(doc):
                return [n['attrs'].get('id') for n in doc.nodes if n['entry'] and re.fullmatch('h[1-6]', n['tag'])]
            check(heading_ids(translated) == heading_ids(original), f'{route}: weekly heading structure/IDs differ')
            check(external(translated) == external(original), f'{route}: weekly source links differ')
            check(features(source_body) == features(english_body), f'{route}: weekly protected content differs')
            check(prose_numbers(source_body) == prose_numbers(english_body), f'{route}: weekly numerical values differ')
            for doc in (translated, original):
                switches = [n['attrs']['data-language-switch'] for n in doc.select('a') if 'data-language-switch' in n['attrs']]
                check(switches == ['article'], f'{route}: weekly language switch does not preserve sections')
                alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('hreflang')}
                check(alternates == {'en': 'https://tingdeliu.github.io' + route, 'zh-CN': 'https://tingdeliu.github.io' + original_route}, f'{route}: weekly language alternates differ')
            check(route not in (SITE / 'feed.xml').read_text(encoding='utf-8'), f'{route}: weekly translation leaked into Chinese feed')
        english_blog = load(SITE / 'en/blog/index.html')
        cards = [n['attrs'].get('href') for n in english_blog.select('a') if 'rc-card' in n['attrs'].get('class', '').split()]
        check(cards == list(reversed(weekly_urls)), 'English Blog issue order or cards differ')
        chinese_blog = load(SITE / 'blog/index.html')
        chinese_cards = [n['attrs'].get('href') for n in chinese_blog.select('a') if 'rc-card' in n['attrs'].get('class', '').split()]
        check(not any(url in chinese_cards for url in weekly_urls), 'Weekly translations duplicated in Chinese Blog')
        research_cards = [n['attrs'].get('href') for n in load(SITE / 'en/research/index.html').select('a') if 'rc-card' in n['attrs'].get('class', '').split()]
        check(not any(url in research_cards for url in weekly_urls), 'Weekly digests mixed into English Research')
        print(f'Blog: {len(weekly_targets)} complete weekly digest translations')
    zh_research = load(SITE / 'research/index.html')
    check(not any('/en/VLN-Survey/' == n['attrs'].get('href') for n in zh_research.select('a')), 'Translation duplicated in Chinese research cards')
    for route in ('about', 'archive', 'blog', 'home', 'research', 'tags',
                  'en', 'en/blog', 'en/research'):
        check((SITE / route / 'index.html').is_file(), f'Entry page missing: /{route}/')
    for asset in ('style.css', 'assets/css/vla-survey.css',
                  'assets/js/article-ui.js', 'assets/js/language-switch.js',
                  'assets/js/leaderboard.js', 'assets/js/vla-survey.js'):
        check((SITE / asset).is_file(), f'Frontend asset missing: /{asset}')
    for directory in ('docs', 'scripts', '.cache', 'tmp', 'paper_summary', 'pages', '_site-drafts'):
        check(not (SITE / directory).exists(), f'Internal directory published: {directory}')
    check(not (SITE / 'AGENTS.md').exists(), 'Local agent instructions published')
    for error in errors:
        print('ERROR:', error)
    print(f'{len(english_files)} English pages; {len(en_headings)} English headings; '
          f'{len(en.select("table", True))} tables; {len(errors)} errors')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
