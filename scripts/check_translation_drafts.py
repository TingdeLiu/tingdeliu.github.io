#!/usr/bin/env python3
"""Check complete paper sections within unpublished, partial translation drafts."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from check_translations import ROOT, digest, read_document, within_root


def paper_sections(body):
    starts = list(re.finditer(r'^## \d+\. .+(?:\n\{: id="[^"]+"\})?', body, re.M))
    result = {}
    for i, start in enumerate(starts):
        heading = start.group()
        anchor = re.search(r'\{#([^}]+)\}|\{: id="([^"]+)"\}', heading)
        if not anchor:
            raise ValueError(f'Paper heading has no explicit anchor: {heading}')
        identity = anchor[1] or anchor[2]
        end = starts[i + 1].start() if i + 1 < len(starts) else len(body)
        section = body[start.start():end]
        # The source has reference sections and scripts after the last paper.
        boundary = re.search(r'^# ', section, re.M)
        if boundary:
            section = section[:boundary.start()]
        result[identity] = section.strip()
    return result


def normalize_math_labels(prose):
    """Normalize only reviewed descriptive labels; preserve equations and values."""
    labels = {
        r'\text{ 在公共 Trunk}': r'\text{ is in the shared trunk}',
        r'\text{ 在同一 Branch}': r'\text{ are in the same branch}',
        r'\text{ 米}': r'\text{ m}',
        r'\text{"红色工牌"}': r'\text{"red badge"}',
        r'\text{"当前时间-5分钟"}': r'\text{"current time minus 5 minutes"}',
        'Observable (可见)': 'Observable (visible)',
        'Unobservable (被遮挡/位于表面后方)': 'Unobservable (occluded / behind the surface)',
        'Disappeared (已消失/位于表面前方)': 'Disappeared (absent / in front of the surface)',
        '若执行 done() 且机器人距离目标物体符合成功阈值': 'done() called and robot-to-target distance meets the success threshold',
        '若执行 done() 但未成功（误报）': 'done() called without success (false positive)',
        '执行其他动作': 'other actions',
        '当目标物体已被收入场景图且机器人向其靠近': 'target is in the scene graph and the robot moves closer',
        '其他情况': 'otherwise',
        '若目标 G 已在场景图中': 'goal G is already in the scene graph',
        '当目标 G 首次被收入场景图': 'goal G first enters the scene graph',
        '发现新拓扑节点的归一化增量': 'normalized increase in newly discovered topological nodes',
    }
    for source, translated in labels.items():
        prose = prose.replace(source, translated)
    return prose


def features(section):
    prose = normalize_math_labels(re.sub(r'<img\b[^>]*>', '', section))
    return {
        'images': re.findall(r'<img[^>]+src="([^"]+)"', section),
        'math': Counter(re.findall(r'\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)', prose, re.S)),
        'links': Counter(re.findall(r'\]\((https?://[^)]+)\)', section)),
        'tables': len(re.findall(r'^\|\s*:?-', section, re.M)),
        'table_row_widths': [line.count('|') for line in section.splitlines() if line.startswith('|')],
        'table_numbers': Counter(re.findall(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?',
                                            '\n'.join(line for line in section.splitlines() if line.startswith('|')))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='Fail when reviewed source sections change')
    args = parser.parse_args()
    errors, stale = [], False
    for path in sorted((ROOT / 'translations').glob('*.en.progress.json')):
        progress = json.loads(path.read_text(encoding='utf-8'))
        target = within_root(progress['translation'])
        source = within_root(progress['source'])
        fields, _, translated_body = read_document(target)
        _, _, source_body = read_document(source)
        if progress.get('stage') == 'complete' and fields.get('published', '').lower() != 'false':
            print(f'{target.relative_to(ROOT)}: complete; checked by the published source snapshot')
            continue
        if fields.get('published', '').lower() != 'false' or fields.get('translation_scope') != 'partial':
            errors.append(f'{target}: partial draft must remain unpublished')
        if fields.get('source_path') != progress['source']:
            errors.append(f'{target}: incorrect source mapping')
        originals, translated = paper_sections(source_body), paper_sections(translated_body)
        expected = progress['completed_papers']
        if set(translated) != set(expected):
            errors.append(f'{target}: translated paper set differs from progress record')
        for identity, record in expected.items():
            if identity not in originals or identity not in translated:
                errors.append(f'{identity}: missing source or translated section')
                continue
            changed = digest(originals[identity]) != record['source_hash']
            if changed:
                stale = True
                print(f'{identity}: SOURCE CHANGED; review this draft section')
            else:
                before, after = features(originals[identity]), features(translated[identity])
                for feature in before:
                    if before[feature] != after[feature]:
                        errors.append(f'{identity}: {feature} differ between source and draft')
            visible_text = re.sub(r'\{: id="[^"]+"\}', '', translated[identity])
            if re.search(r'[\u4e00-\u9fff]', visible_text):
                errors.append(f'{identity}: untranslated Chinese remains')
        print(f'{target.relative_to(ROOT)}: {len(translated)} / {len(originals)} papers translated; unpublished')
    for error in errors:
        print('ERROR:', error)
    return int(bool(errors) or (args.strict and stale))


if __name__ == '__main__':
    raise SystemExit(main())
