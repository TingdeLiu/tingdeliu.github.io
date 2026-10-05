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
    # Inline TeX ends on the same line. Currency such as $0.66 or $5/month
    # must not consume tables, headings, and prose up to a later dollar sign.
    display_math = re.findall(r'\$\$.*?\$\$', prose, re.S)
    inline_prose = re.sub(r'\$\$.*?\$\$|```.*?```', '', prose, flags=re.S)
    inline_math = re.findall(r'(?<![\\$])\$(?!\$)[^\n$]+?(?<!\s)\$(?![\d$])', inline_prose)
    return {
        'images': re.findall(r'<img[^>]+src="([^"]+)"', section),
        'math': Counter(display_math + inline_math),
        'links': Counter(re.findall(r'\]\((https?://[^)]+)\)', section)),
        'tables': len(re.findall(r'^\|\s*:?-', section, re.M)),
        'table_row_widths': [line.count('|') for line in section.splitlines() if line.startswith('|')],
        'table_numbers': Counter(re.findall(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?',
                                            '\n'.join(line for line in section.splitlines() if line.startswith('|')))),
    }


def normalize_agent_table_labels(body):
    """Canonicalize reviewed date/unit conversions in agent-survey tables only."""
    months = 'January February March April May June July August September October November December'.split()
    replacements = {
        '百毫秒级': 'Hundreds of milliseconds',
        '发布一周年': 'First anniversary',
        '466 道三级难度题目': '466 questions at three difficulty levels',
    }
    lines = []
    for line in body.splitlines():
        if line.startswith('|'):
            for source, english in replacements.items():
                line = line.replace(source, english)
            for number, name in enumerate(months, 1):
                line = re.sub(r'\b' + name + r'\b', str(number), line)
            line = re.sub(r'(\d[\d,]*(?:\.\d+)?)\s*百万',
                          lambda m: format(float(m[1].replace(',', '')) * 1000000, '.12g'), line)
            line = re.sub(r'(\d[\d,]*(?:\.\d+)?)\s*万',
                          lambda m: format(float(m[1].replace(',', '')) * 10000, '.12g'), line)
            line = re.sub(r'(\d[\d,]*(?:\.\d+)?)\s*million',
                          lambda m: format(float(m[1].replace(',', '')) * 1000000, '.12g'), line)
            line = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', line)
        lines.append(line)
    return '\n'.join(lines)


def normalize_vla_labels(body):
    """Canonicalize reviewed descriptive labels without changing VLA equations."""
    labels = {
        '世界模型:': 'World model:', '联合预测:': 'Joint prediction:',
        '从观察和语言预测动作': 'predict actions from observations and language',
        '从当前观察和动作预测未来观察': 'predict future observations from current observations and actions',
        '从观察序列推断动作': 'infer actions from observation sequences',
        '从观察和语言生成未来视频': 'generate future video from observations and language',
        '同时生成视频和动作': 'generate video and actions jointly',
    }
    def localize(match):
        text = match[0]
        for zh, en in labels.items():
            text = text.replace(zh, en)
        return text
    body = re.sub(r'\$\$.*?\$\$|(?<!\$)\$(?!\$)[^\n$]+?(?<!\s)\$(?![\d$])', localize, body, flags=re.S)
    return normalize_large_model_labels(body)


def normalize_large_model_labels(body):
    """Canonicalize translated descriptions inside otherwise identical equations."""
    labels = {
        '静态总显存': 'Static GPU memory', ' 字节': ' bytes',
        '总GPU数': 'Total GPUs', 'DP度': 'DP degree', 'TP度': 'TP degree', 'PP度': 'PP degree',
        '修正因子': 'Correction factor', '旧': 'old', '新': 'new',
        '当前块贡献': 'Current block contribution', '量化': 'Quantization',
        'KV Cache大小': 'KV cache size',
        '压缩为低维潜在向量，维度 ': 'Compressed latent vector, dimension ',
        '推理时上投影还原': 'Up-projection at inference',
        'GPU 小时': 'GPU hours', '单卡峰值 FLOPS': 'Peak FLOPS per GPU',
        '总成本': 'Total cost', '单价': 'Hourly price',
        '存储、人力等其他成本': 'Other costs: storage, labor, etc.',
        '词表大小': 'Vocabulary size',
        '（线性缩放规则，适用于 SGD）': '(linear scaling rule for SGD)',
        '（平方根缩放规则，适用于 Adam/AdamW）': '(square-root scaling rule for Adam/AdamW)',
    }
    # These replacements apply only to math; prose uses independently reviewed English.
    def localize(match):
        text = match[0]
        for zh, en in labels.items():
            text = text.replace(zh, en)
        return text
    body = re.sub(r'\$\$.*?\$\$|(?<!\$)\$(?!\$)[^\n$]+?(?<!\s)\$(?![\d$])', localize, body, flags=re.S)
    body = normalize_agent_table_labels(body)
    lines = []
    for line in body.splitlines():
        if line.startswith('|'):
            line = line.replace('三维高斯 (3D Gaussian)', '3D Gaussian')
            line = line.replace('三维', '3D')
            line = re.sub(r'(?i)three[- ]dimensional', '3D', line)
            for n, word in [('一', '1'), ('二', '2'), ('三', '3')]:
                line = line.replace('第' + n + '阶段', 'Stage ' + word)
                line = line.replace('阶段' + n, 'Stage ' + word)
            for word, n in [('One', '1'), ('Two', '2'), ('Three', '3'), ('first', '1'), ('second', '2'), ('third', '3')]:
                line = re.sub(r'(?i)(stage|phase):?\s+' + word + r'\b', r'\1 ' + n, line)
                line = re.sub(r'(?i)\b(?:the )?' + word + r' stage\b', 'Stage ' + n, line)
            from decimal import Decimal
            scales = {'千万': 10000000, '亿': 100000000, 'billion': 1000000000}
            def scale(match):
                value = format(Decimal(match[1]) * scales[match[2].lower()], 'f')
                return value.rstrip('0').rstrip('.') if '.' in value else value
            line = re.sub(r'(\d+(?:\.\d+)?)\s*(千万|亿|billion\b)', scale, line, flags=re.I)
        lines.append(line)
    return '\n'.join(lines)


def normalize_ml_labels(body):
    """Canonicalize reviewed ML units and descriptive TeX labels for parity."""
    labels = {
        '（重置门）': '(reset gate)', '（更新门）': '(update gate)',
        '（速度累积）': '(velocity accumulation)',
        '一阶矩': 'first moment', '二阶矩': 'second moment',
        '步骤一/三': 'Step 1/3', '步骤三': 'Step 3', '步骤二': 'Step 2',
    }
    for source, english in labels.items():
        body = body.replace(source, english)
    # Express both editions in base units, including Chinese ten-thousands
    # and hundred-millions. This catches mistranslated magnitudes.
    units = {'万': 10000, '亿': 100000000, 'million': 1000000,
             'billion': 1000000000}
    body = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', body)
    body = re.sub(r'(\d+(?:\.\d+)?)\s*(万|亿|million|billion)',
                  lambda m: format(float(m[1]) * units[m[2]], '.12g'), body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='Fail when reviewed source sections change')
    args = parser.parse_args()
    errors, stale = [], False
    for path in sorted((ROOT / 'docs/translations').glob('*.en.progress.json')):
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
