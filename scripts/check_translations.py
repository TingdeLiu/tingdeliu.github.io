#!/usr/bin/env python3
"""Track reviewed source versions without overwriting translations.

Run before Jekyll to generate _data/translation_status.json. Stale translations
warn but do not block Chinese publishing. --strict makes stale records fail.
Only use --record ID --date YYYY-MM-DD after reviewing the translation.
"""
import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_document(path):
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n')
    match = re.match(r'\A---\n(.*?)\n---\n(.*)', text, re.S)
    if not match:
        raise ValueError(f'{path}: missing front matter')
    fields = {}
    for line in match[1].splitlines():
        if ':' in line and not line.startswith((' ', '#')):
            key, value = line.split(':', 1)
            fields[key] = value.strip().strip('\"\'')
    return fields, match[1], match[2]


def section_hashes(path):
    _, metadata, body = read_document(path)
    # Language pairing does not alter the translated meaning of the source.
    metadata = '\n'.join(line for line in metadata.splitlines()
                         if not re.match(r'^(lang|translation_id):', line))
    sections = {'metadata': digest(metadata.strip())}
    key, lines, fence, paper = 'introduction', [], None, None
    for line in body.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})', line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        heading = re.match(r'^#{1,6}\s+(.+)', line) if not fence else None
        if heading:
            sections[key] = digest('\n'.join(lines).strip())
            if re.match(r'^#\s', line):
                paper = None
            paper_heading = re.match(r'^##\s+\d+\.\s.*\{#([^}]+)\}', line)
            if paper_heading:
                paper = paper_heading[1]
            # Repeated labels such as 精华 belong to a specific paper. A paper's
            # stable anchor prevents unrelated additions from renumbering keys.
            key = f'paper:{paper} / {heading[1]}' if paper else heading[1]
            if key in sections:
                raise ValueError(f'{path}: duplicate source heading {key}')
            lines = []
        lines.append(line)
    sections[key] = digest('\n'.join(lines).strip())
    return sections


def compare_sections(current, recorded):
    return {
        'changed': [key for key in current if key in recorded and current[key] != recorded[key]],
        'added': [key for key in current if key not in recorded],
        'removed': [key for key in recorded if key not in current],
    }


def within_root(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside repository: {relative}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', metavar='ID', help='Record source after translation review')
    parser.add_argument('--date', help='Reviewed translation date, YYYY-MM-DD')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--output', default='_data/translation_status.json')
    args = parser.parse_args()
    if args.record and not args.date:
        parser.error('--record requires --date')
    if args.date:
        date.fromisoformat(args.date)
    statuses, seen = {}, set()
    for target in sorted((ROOT / '_translations').rglob('*.md')):
        fields, _, _ = read_document(target)
        if fields.get('published', '').lower() == 'false':
            print(f'{target.relative_to(ROOT).as_posix()}: unpublished draft (not synchronized)')
            continue
        identity = fields.get('translation_id')
        if not identity or identity in seen:
            raise ValueError(f'{target}: missing or duplicate translation_id')
        if fields.get('lang') != 'en':
            raise ValueError(f'{target}: this registry currently supports English translations')
        seen.add(identity)
        source = within_root(fields['source_path'])
        source_fields, _, _ = read_document(source)
        if source_fields.get('translation_id') != identity:
            raise ValueError(f'{target}: source translation_id does not match')
        current = section_hashes(source)
        record_path = within_root(f'translations/{identity}.en.json')
        if args.record == identity:
            if fields.get('translation_updated') != args.date:
                raise ValueError('Set translation_updated in the translated document to --date first')
            record = {
                'source': source.relative_to(ROOT).as_posix(),
                'translation': target.relative_to(ROOT).as_posix(),
                'reviewed_on': args.date,
                'source_revision_date': fields['source_revision_date'],
                'sections': current,
            }
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record['source'] != fields['source_path'] or record['translation'] != target.relative_to(ROOT).as_posix():
            raise ValueError(f'{record_path}: source/translation mapping mismatch')
        if record['reviewed_on'] != fields['translation_updated'] or record['source_revision_date'] != fields['source_revision_date']:
            raise ValueError(f'{record_path}: version metadata mismatch; review before recording a new snapshot')
        changes = compare_sections(current, record['sections'])
        stale = any(changes.values())
        statuses[identity] = dict(stale=stale, reviewed_on=record['reviewed_on'], **changes)
        print(f'{identity}: {"NEEDS SYNC" if stale else "synchronized"}')
        for kind, headings in changes.items():
            for heading in headings:
                print(f'  {kind}: {heading}')
    if args.record and args.record not in seen:
        raise ValueError(f'Unknown translation: {args.record}')
    output = within_root(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return int(args.strict and any(item['stale'] for item in statuses.values()))


if __name__ == '__main__':
    raise SystemExit(main())
