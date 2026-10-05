#!/usr/bin/env python3
"""Reject a source map overwriting the site's compiled primary stylesheet."""
import argparse
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def validate_stylesheet(css):
    try:
        data = json.loads(css)
    except json.JSONDecodeError:
        data = None
    if data is not None:
        if isinstance(data, dict) and 'mappings' in data and 'sources' in data:
            raise ValueError('Sass source map JSON overwrote the compiled stylesheet')
        raise ValueError('Stylesheet contains JSON instead of CSS')
    for selector in ('body', r'\.container'):
        if not re.search(r'(?:^|\})\s*' + selector + r'\s*\{', css):
            raise ValueError(f'Compiled stylesheet is missing the {selector} rule')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stylesheet', nargs='?', default=str(ROOT / '_site/style.css'),
                        help='Built stylesheet path or deployed HTTP(S) URL')
    args = parser.parse_args()
    try:
        if args.stylesheet.startswith(('http://', 'https://')):
            request = Request(args.stylesheet, headers={'Cache-Control': 'no-cache'})
            with urlopen(request, timeout=20) as response:
                css = response.read().decode('utf-8-sig')
        else:
            css = Path(args.stylesheet).read_text(encoding='utf-8-sig')
        validate_stylesheet(css)
    except (OSError, ValueError) as error:
        print(f'ERROR: {args.stylesheet}: {error}')
        return 1
    print(f'PASS: {args.stylesheet}: compiled body and container styles are present')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
