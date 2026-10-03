#!/usr/bin/env python3
"""Validate the built English release and its Chinese source using stdlib only."""
import re
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

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
    check(len(english_files) >= 3, 'Missing English entry pages or survey')
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
    status = json.loads((ROOT / '_data/translation_status.json').read_text(encoding='utf-8'))['vln-survey']
    if not status['stale']:
        check(en_headings == zh_headings, 'Translated headings do not preserve source structure/IDs')
        for tag in ('table', 'img', 'pre'):
            check(len(en.select(tag, True)) == len(zh.select(tag, True)), f'{tag}: source/translation structural count differs')
        math_pattern = r'\\\[(.*?)\\\]|\\\((.*?)\\\)'
        check(re.findall(math_pattern, en.raw, re.S) == re.findall(math_pattern, zh.raw, re.S), 'Rendered mathematical expressions differ')
        for tag, attr in [('a', 'href')]:
            def external(doc):
                return Counter(n['attrs'].get(attr) for n in doc.select(tag, True)
                               if n['attrs'].get(attr, '').startswith(('http://', 'https://')))
            check(external(en) == external(zh), 'External citations differ between source and translation')
    else:
        print('Source changed: skipping source/translation parity until the English edition is synchronized.')
    check(('translation-stale' in en.raw) == status['stale'], 'Stale translation notice does not match source status')
    for doc, name in [(en, 'English'), (zh, 'Chinese')]:
        switches = [n['attrs']['data-language-switch'] for n in doc.select('a') if 'data-language-switch' in n['attrs']]
        check(switches == ['index' if status['stale'] else 'article'], f'{name}: unsafe section switching for synchronization status')
        alternates = {n['attrs'].get('hreflang'): n['attrs'].get('href') for n in doc.select('link') if n['attrs'].get('hreflang')}
        check(alternates == {'en': 'https://tingdeliu.github.io/en/VLN-Survey/', 'zh-CN': 'https://tingdeliu.github.io/VLN-Survey/'}, f'{name}: incorrect language alternates')
    check('/en/VLN-Survey/' not in (SITE / 'feed.xml').read_text(encoding='utf-8'), 'Translation leaked into Chinese feed')
    zh_research = load(SITE / 'research/index.html')
    check(not any('/en/VLN-Survey/' == n['attrs'].get('href') for n in zh_research.select('a')), 'Translation duplicated in Chinese research cards')
    for directory in ('translations', 'docs', 'scripts', 'tmp'):
        check(not (SITE / directory).exists(), f'Internal directory published: {directory}')
    for error in errors:
        print('ERROR:', error)
    print(f'{len(english_files)} English pages; {len(en_headings)} English headings; '
          f'{len(en.select("table", True))} tables; {len(errors)} errors')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
