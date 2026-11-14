"""Generate the explanatory SVG figures for the reinforcement-learning survey.

Every figure is emitted twice, once with Chinese labels (``rl-*.svg``) and once
with English labels (``rl-*-en.svg``), into ``images/ML/``.  The numbers drawn
in the figures (MDP values, grid-world values, PPO clip curves, GRPO
advantages) are computed here rather than typed by hand, so the figures and the
worked examples in the post stay consistent.

Usage::

    python -I scripts/figures/rl_survey_figures.py           # write SVGs
    python -I scripts/figures/rl_survey_figures.py --print   # also print numbers
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "images" / "ML"

# ---------------------------------------------------------------- palette ---
BG = "#fbfaf7"
FRAME = "#e4e0d8"
INK = "#1f2328"
MUTED = "#6b6f76"
FAINT = "#b9bcc2"
BLUE = "#2f6fab"
BLUE_SOFT = "#dbe7f3"
ORANGE = "#d97757"
ORANGE_SOFT = "#f6e1d8"
GREEN = "#3a8a5c"
GREEN_SOFT = "#dcefe3"
RED = "#b8483e"
RED_SOFT = "#f3dcd9"

SANS = "'PingFang SC','Microsoft YaHei','Noto Sans SC','Helvetica Neue',Arial,sans-serif"
SERIF = "'Times New Roman','Noto Serif',Georgia,serif"

STYLE = f"""
<style>
  text {{ font-family: {SANS}; fill: {INK}; }}
  .t {{ font-size: 15px; }}
  .s {{ font-size: 13px; fill: {MUTED}; }}
  .xs {{ font-size: 11.5px; fill: {MUTED}; }}
  .b {{ font-weight: 600; }}
  .m {{ font-family: {SERIF}; font-style: italic; font-size: 16px; }}
  .ms {{ font-family: {SERIF}; font-style: italic; font-size: 14px; }}
  .h {{ font-size: 15px; font-weight: 600; }}
</style>"""


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w: int, h: int, body: list[str], title: str) -> str:
    marker = "".join(
        f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        for name, col in (("ink", INK), ("muted", MUTED), ("blue", BLUE), ("orange", ORANGE),
                          ("green", GREEN), ("red", RED), ("faint", FAINT))
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}">\n'
        f"<title>{esc(title)}</title>\n<defs>{marker}{STYLE}</defs>\n"
        f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="14" fill="{BG}" stroke="{FRAME}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def text(x, y, s, cls="t", anchor="middle", fill=None, extra=""):
    f = f' style="fill:{fill}"' if fill else ""
    # escape bare comparison signs while keeping the <tspan> markup used for subscripts
    s = s.replace(" < ", " &lt; ").replace(" > ", " &gt; ")
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{f} {extra}>{s}</text>'


def line(x1, y1, x2, y2, col=INK, w=1.6, arrow=None, dash=None):
    a = f' marker-end="url(#ah-{arrow})"' if arrow else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="{w}"{d}{a}/>'


def circle(x, y, r, fill="none", stroke=INK, w=1.6):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>'


def rect(x, y, w, h, fill="none", stroke=INK, sw=1.4, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def unit(dx, dy):
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def curve(p0, p1, bend, r0=0.0, r1=0.0, col=INK, w=1.6, arrow="ink", dash=None):
    """Quadratic edge from p0 to p1, bowed sideways by ``bend`` px, trimmed by radii.

    Returns (svg, midpoint, normal) so labels can be placed beside the curve.
    """
    (x0, y0), (x1, y1) = p0, p1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    nx, ny = unit(-(y1 - y0), x1 - x0)
    cx, cy = mx + nx * bend, my + ny * bend
    ux, uy = unit(cx - x0, cy - y0)
    sx, sy = x0 + ux * r0, y0 + uy * r0
    vx, vy = unit(cx - x1, cy - y1)
    ex, ey = x1 + vx * r1, y1 + vy * r1
    a = f' marker-end="url(#ah-{arrow})"' if arrow else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    path = f'<path d="M{sx:.1f},{sy:.1f} Q{cx:.1f},{cy:.1f} {ex:.1f},{ey:.1f}" fill="none" stroke="{col}" stroke-width="{w}"{d}{a}/>'
    # point on the curve at t = 0.5
    px = 0.25 * sx + 0.5 * cx + 0.25 * ex
    py = 0.25 * sy + 0.5 * cy + 0.25 * ey
    return path, (px, py), (nx, ny)


def sub(base: str, s: str, cls="m") -> str:
    """Italic math symbol with a subscript, e.g. sub('s', 't+1')."""
    return f'{base}<tspan dy="4" font-size="70%">{s}</tspan><tspan dy="-4"> </tspan>'


# =============================================================== 1. MDP ======
MDP = {
    # (state, action): [(next_state, prob, reward), ...]
    ("H", "clean"): [("H", 0.7, 2.0), ("L", 0.3, 2.0)],
    ("H", "wait"): [("H", 1.0, 0.5)],
    ("L", "clean"): [("L", 0.6, 2.0), ("H", 0.4, -3.0)],
    ("L", "wait"): [("L", 1.0, 0.5)],
    ("L", "charge"): [("H", 1.0, 0.0)],
}
GAMMA_MDP = 0.9


def mdp_policy_value(policy: dict[str, str], gamma=GAMMA_MDP):
    """Exact V_pi for a deterministic policy by iterating the Bellman equation."""
    v = {"H": 0.0, "L": 0.0}
    for _ in range(5000):
        v = {s: sum(p * (r + gamma * v[s2]) for s2, p, r in MDP[(s, policy[s])]) for s in v}
    return v


def mdp_optimal(gamma=GAMMA_MDP):
    v = {"H": 0.0, "L": 0.0}
    for _ in range(5000):
        q = {k: sum(p * (r + gamma * v[s2]) for s2, p, r in out) for k, out in MDP.items()}
        v = {s: max(val for (ss, _), val in q.items() if ss == s) for s in v}
    return v, q


def fig_mdp(lang: str) -> str:
    zh = lang == "zh"
    L = {
        "H": "高电量" if zh else "High",
        "L": "低电量" if zh else "Low",
        "clean": "清扫" if zh else "clean",
        "wait": "待机" if zh else "wait",
        "charge": "充电" if zh else "recharge",
        "legend_s": "状态 s" if zh else "state s",
        "legend_a": "动作 a（选定后由环境掷骰子）" if zh else "action a (then the environment rolls the dice)",
        "legend_e": "箭头标注：转移概率 P(s′|s,a)，即时奖励 R" if zh else "edge label: transition prob. P(s′|s,a), reward R",
        "title": "扫地机器人的两状态 MDP" if zh else "A two-state MDP for a cleaning robot",
    }
    H, Lp = (190, 205), (570, 205)
    R = 50
    b = []
    b.append(text(380, 34, L["title"], "h"))
    # states
    for (x, y), key in ((H, "H"), (Lp, "L")):
        b.append(circle(x, y, R, fill=BLUE_SOFT, stroke=BLUE, w=2))
        b.append(text(x, y - 2, L[key], "t b"))
        b.append(text(x, y + 20, f"s = {key}", "ms", fill=MUTED))

    def action(dot, name, src, outs, label_side):
        dx, dy = dot
        p, _, _ = curve(src, dot, 0, R, 7, col=MUTED, w=1.4, arrow=None)
        b.append(p)
        b.append(circle(dx, dy, 7, fill=INK, stroke=INK))
        lx, ly = label_side
        b.append(text(dx + lx, dy + ly, L[name], "t b", fill=INK))
        for dst, bend, prob, rew, (ox, oy) in outs:
            col = GREEN if rew > 0 else (RED if rew < 0 else MUTED)
            arrow = {GREEN: "green", RED: "red", MUTED: "muted"}[col]
            path, (mx, my), _ = curve(dot, dst, bend, 8, R + 3, col=col, w=1.8, arrow=arrow)
            b.append(path)
            sign = "+" if rew > 0 else ("−" if rew < 0 else "")
            lab = f"{prob:g}, {sign}{abs(rew):g}"
            b.append(text(mx + ox, my + oy, lab, "ms", fill=col))

    # H: clean (top middle) -> H 0.7 / L 0.3
    action((380, 92), "clean", H, [(H, 40, 0.7, 2, (-4, -10)), (Lp, -40, 0.3, 2, (6, -10))], (0, -16))
    # H: wait (top-left self loop)
    action((70, 92), "wait", H, [(H, -55, 1.0, 0.5, (-28, 18))], (-2, -16))
    # L: clean (bottom middle) -> L 0.6 / H 0.4 (-3)
    action((380, 322), "clean", Lp, [(Lp, 40, 0.6, 2, (6, 20)), (H, -40, 0.4, -3, (-6, 20))], (0, 28))
    # L: wait (top-right self loop)
    action((690, 92), "wait", Lp, [(Lp, 55, 1.0, 0.5, (30, 18))], (2, -16))
    # L: charge (middle) -> H
    action((380, 205), "charge", Lp, [(H, 0, 1.0, 0, (0, -9))], (0, 26))

    # legend
    y = 372
    b.append(circle(52, y - 4, 8, fill=BLUE_SOFT, stroke=BLUE, w=1.6))
    b.append(text(66, y, L["legend_s"], "s", "start"))
    lx = 150 if zh else 150
    b.append(circle(lx, y - 4, 6, fill=INK, stroke=INK))
    b.append(text(lx + 12, y, L["legend_a"], "s", "start"))
    b.append(text(52 - 8, y + 22, L["legend_e"], "s", "start"))
    b.append(line(470, y + 18, 500, y + 18, GREEN, 1.8, "green"))
    b.append(text(506, y + 22, "+" if True else "", "s", "start", GREEN))
    b.append(line(530, y + 18, 560, y + 18, RED, 1.8, "red"))
    b.append(text(566, y + 22, "−", "s", "start", RED))
    return svg(760, 410, b, L["title"])


# ===================================================== 2. trajectory/γ =====
def fig_return(lang: str) -> str:
    zh = lang == "zh"
    title = "轨迹、马尔可夫性与折扣权重" if zh else "Trajectory, Markov property and discount weights"
    b = [text(380, 32, title, "h")]
    # top: trajectory chain
    xs = [70, 185, 300, 415, 530, 645]
    labels = ["t−2", "t−1", "t", "t+1", "t+2", "t+3"]
    y = 110
    for i, (x, lab) in enumerate(zip(xs, labels)):
        past = i < 2
        cur = i == 2
        fill = "#eeeeec" if past else (ORANGE_SOFT if cur else BLUE_SOFT)
        stroke = FAINT if past else (ORANGE if cur else BLUE)
        b.append(circle(x, y, 24, fill=fill, stroke=stroke, w=2))
        b.append(f'<text x="{x}" y="{y+6}" class="m" text-anchor="middle" style="fill:{MUTED if past else INK}">s<tspan dy="5" font-size="70%">{lab}</tspan></text>')
        if i < len(xs) - 1:
            col = FAINT if i < 2 else INK
            arr = "faint" if i < 2 else "ink"
            b.append(line(x + 26, y, xs[i + 1] - 28, y, col, 1.6, arr))
            mx = (x + xs[i + 1]) / 2
            nxt = labels[i + 1]
            b.append(f'<text x="{mx}" y="{y-12}" class="ms" text-anchor="middle" style="fill:{MUTED if i<2 else INK}">a<tspan dy="4" font-size="70%">{lab}</tspan></text>')
            if i >= 2:
                b.append(f'<text x="{mx}" y="{y+30}" class="ms" text-anchor="middle" style="fill:{GREEN}">r<tspan dy="4" font-size="70%">{nxt}</tspan></text>')
    b.append(text(720, y + 6, "⋯", "t", fill=MUTED))
    # markov brace
    b.append(f'<path d="M48,{y-42} L48,{y-50} L322,{y-50} L322,{y-42}" fill="none" stroke="{ORANGE}" stroke-width="1.4"/>')
    mk = "马尔可夫性：预测未来只需 s_t，灰色历史可以丢掉" if zh else "Markov property: the future depends only on s_t; the grey history can be dropped"
    mk = mk.replace("s_t", "s<tspan dy='4' font-size='70%'>t</tspan><tspan dy='-4'> </tspan>")
    b.append(f'<text x="48" y="{y-58}" class="s" text-anchor="start" style="fill:{ORANGE}">{mk}</text>')
    g = "回报 G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ⋯" if zh else "Return G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + ⋯"
    g = (g.replace("G_t", "G<tspan dy='4' font-size='70%'>t</tspan><tspan dy='-4'> </tspan>")
          .replace("r_{t+1}", "r<tspan dy='4' font-size='70%'>t+1</tspan><tspan dy='-4'> </tspan>")
          .replace("r_{t+2}", "r<tspan dy='4' font-size='70%'>t+2</tspan><tspan dy='-4'> </tspan>")
          .replace("r_{t+3}", "r<tspan dy='4' font-size='70%'>t+3</tspan><tspan dy='-4'> </tspan>"))
    b.append(f'<text x="415" y="{y+62}" class="t" text-anchor="middle" style="fill:{GREEN}">{g}</text>')

    # bottom: discount bars
    x0, y0, wbar, hmax = 80, 340, 30, 130
    K = 16
    b.append(line(x0 - 6, y0, x0 + K * wbar + 10, y0, MUTED, 1.2))
    b.append(line(x0 - 6, y0, x0 - 6, y0 - hmax - 12, MUTED, 1.2))
    b.append(text(x0 - 12, y0 - hmax + 4, "1", "xs", "end"))
    b.append(text(x0 - 12, y0 + 4, "0", "xs", "end"))
    for k in range(K):
        for gamma, col, off in ((0.9, BLUE, 2), (0.5, ORANGE, 15)):
            h = hmax * gamma ** k
            b.append(f'<rect x="{x0+k*wbar+off:.1f}" y="{y0-h:.1f}" width="12" height="{h:.1f}" fill="{col}" opacity="0.85"/>')
        if k % 2 == 0:
            b.append(text(x0 + k * wbar + 15, y0 + 16, str(k), "xs"))
    axis = "未来第 k 步奖励的权重 γᵏ" if zh else "weight γᵏ on the reward k steps ahead"
    b.append(text(x0 + K * wbar / 2, y0 + 36, axis, "s"))
    lx = x0 + K * wbar + 30
    b.append(f'<rect x="{lx}" y="{y0-hmax}" width="12" height="12" fill="{BLUE}"/>')
    b.append(text(lx + 18, y0 - hmax + 11, "γ = 0.9", "s", "start"))
    b.append(text(lx + 18, y0 - hmax + 29, ("有效视野 ≈ 10 步" if zh else "horizon ≈ 10 steps"), "xs", "start"))
    b.append(f'<rect x="{lx}" y="{y0-hmax+48}" width="12" height="12" fill="{ORANGE}"/>')
    b.append(text(lx + 18, y0 - hmax + 59, "γ = 0.5", "s", "start"))
    b.append(text(lx + 18, y0 - hmax + 77, ("有效视野 ≈ 2 步" if zh else "horizon ≈ 2 steps"), "xs", "start"))
    b.append(text(lx, y0 - hmax + 108, "≈ 1 / (1 − γ)", "ms", "start", MUTED))
    return svg(760, 400, b, title)


# ===================================================== 3. grid world ======
GRID_W, GRID_H = 5, 4
WALLS = {(1, 1), (3, 2)}
TERMINAL = {(4, 0): 1.0, (4, 1): -1.0}
STEP_R = -0.04
GAMMA_GRID = 0.95
SLIP = 0.1
ACTIONS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
PERP = {"U": ("L", "R"), "D": ("L", "R"), "L": ("U", "D"), "R": ("U", "D")}


def grid_move(s, a):
    x, y = s
    dx, dy = ACTIONS[a]
    n = (x + dx, y + dy)
    if not (0 <= n[0] < GRID_W and 0 <= n[1] < GRID_H) or n in WALLS:
        return s
    return n


def grid_value_iteration():
    states = [(x, y) for y in range(GRID_H) for x in range(GRID_W) if (x, y) not in WALLS]
    v = {s: 0.0 for s in states}
    for _ in range(2000):
        nv = {}
        for s in states:
            if s in TERMINAL:
                nv[s] = TERMINAL[s]
                continue
            best = -1e9
            for a in ACTIONS:
                outs = [(grid_move(s, a), 1 - 2 * SLIP)] + [(grid_move(s, p), SLIP) for p in PERP[a]]
                q = sum(p * (STEP_R + GAMMA_GRID * v[s2]) for s2, p in outs)
                best = max(best, q)
            nv[s] = best
        v = nv
    pi = {}
    for s in states:
        if s in TERMINAL:
            continue
        qs = {}
        for a in ACTIONS:
            outs = [(grid_move(s, a), 1 - 2 * SLIP)] + [(grid_move(s, p), SLIP) for p in PERP[a]]
            qs[a] = sum(p * (STEP_R + GAMMA_GRID * v[s2]) for s2, p in outs)
        pi[s] = max(qs, key=qs.get)
    return v, pi


def heat(val):
    """Diverging colour: red (−1) → paper (0) → blue (+1)."""
    t = max(-1.0, min(1.0, val))
    base = (0xfb, 0xfa, 0xf7)
    tgt = (0x2f, 0x6f, 0xab) if t >= 0 else (0xb8, 0x48, 0x3e)
    a = abs(t) ** 0.8 * 0.85
    c = tuple(round(b0 + (t0 - b0) * a) for b0, t0 in zip(base, tgt))
    return "#%02x%02x%02x" % c, a > 0.5


def fig_grid(lang: str) -> str:
    zh = lang == "zh"
    title = "网格世界中的最优价值与最优策略" if zh else "Optimal values and optimal policy in a grid world"
    v, pi = grid_value_iteration()
    C = 62
    b = [text(380, 30, title, "h")]
    heads = (("最优状态价值 V*(s)", "贪心策略 π*(s) = argmaxₐ Q*(s, a)") if zh
             else ("Optimal state value V*(s)", "Greedy policy π*(s) = argmaxₐ Q*(s, a)"))
    for gi, gx in enumerate((50, 405)):
        gy = 70
        b.append(text(gx + GRID_W * C / 2, gy - 12, heads[gi], "t b"))
        for y in range(GRID_H):
            for x in range(GRID_W):
                X, Y = gx + x * C, gy + y * C
                s = (x, y)
                if s in WALLS:
                    b.append(f'<rect x="{X}" y="{Y}" width="{C}" height="{C}" fill="#5b5f66"/>')
                    continue
                if gi == 0:
                    col, dark = heat(v[s])
                    b.append(f'<rect x="{X}" y="{Y}" width="{C}" height="{C}" fill="{col}"/>')
                    txt = f"{v[s]:+.2f}" if s not in TERMINAL else f"{TERMINAL[s]:+g}"
                    b.append(text(X + C / 2, Y + C / 2 + 5, txt.replace("-", "−"), "t", fill="#ffffff" if dark else INK))
                else:
                    if s in TERMINAL:
                        good = TERMINAL[s] > 0
                        b.append(f'<rect x="{X}" y="{Y}" width="{C}" height="{C}" fill="{GREEN_SOFT if good else RED_SOFT}"/>')
                        lab = ("终点 +1" if good else "陷阱 −1") if zh else ("goal +1" if good else "pit −1")
                        b.append(text(X + C / 2, Y + C / 2 + 5, lab, "s b", fill=GREEN if good else RED))
                    else:
                        b.append(f'<rect x="{X}" y="{Y}" width="{C}" height="{C}" fill="{BG}"/>')
                        dx, dy = ACTIONS[pi[s]]
                        cx, cy = X + C / 2, Y + C / 2
                        b.append(line(cx - dx * 16, cy - dy * 16, cx + dx * 16, cy + dy * 16, BLUE, 2.6, "blue"))
        for i in range(GRID_W + 1):
            b.append(line(gx + i * C, gy, gx + i * C, gy + GRID_H * C, FRAME, 1.2))
        for j in range(GRID_H + 1):
            b.append(line(gx, gy + j * C, gx + GRID_W * C, gy + j * C, FRAME, 1.2))
        # start marker
        b.append(text(gx + 10, gy + 3 * C + 16, "S", "xs b", "start", MUTED))
    note = (f"每走一步奖励 {STEP_R:g}，动作以 {1-2*SLIP:g} 的概率按预期执行、各 {SLIP:g} 的概率向两侧滑开，γ = {GAMMA_GRID:g}；深灰格为墙"
            if zh else
            f"Step reward {STEP_R:g}; an action succeeds with prob. {1-2*SLIP:g} and slips sideways with {SLIP:g} each; γ = {GAMMA_GRID:g}; dark cells are walls")
    b.append(text(380, 345, note.replace("-", "−"), "s"))
    return svg(760, 365, b, title)


# ===================================================== 4. backup diagram ===
def fig_backup(lang: str) -> str:
    zh = lang == "zh"
    title = "Bellman 期望方程的备份图" if zh else "Backup diagram of the Bellman expectation equation"
    b = [text(380, 30, title, "h")]
    # the full equation on one line, each factor coloured like the tree level it comes from
    b.append(
        f'<text x="380" y="66" class="m" text-anchor="middle">V<tspan dy="4" font-size="70%">π</tspan>'
        f'<tspan dy="-4">(s) = </tspan><tspan style="fill:{ORANGE}">Σ<tspan dy="4" font-size="70%">a</tspan>'
        f'<tspan dy="-4"> π(a|s) </tspan></tspan><tspan style="fill:{GREEN}">Σ<tspan dy="4" font-size="70%">s′</tspan>'
        f'<tspan dy="-4"> P(s′|s,a) [ R(s,a) </tspan></tspan><tspan style="fill:{BLUE}">+ γ V'
        f'<tspan dy="4" font-size="70%">π</tspan><tspan dy="-4">(s′)</tspan></tspan> ]</text>'
    )
    root = (280, 110)
    acts = [(160, 195), (280, 195), (400, 195)]
    leaves = []
    b.append(circle(*root, 18, fill=BLUE_SOFT, stroke=BLUE, w=2))
    b.append(text(root[0], root[1] + 6, "s", "m"))
    for i, a in enumerate(acts):
        b.append(line(root[0], root[1] + 18, a[0], a[1] - 8, INK, 1.4))
        b.append(circle(*a, 8, fill=INK, stroke=INK))
        for dx in (-34, 34):
            leaf = (a[0] + dx, 290)
            leaves.append(leaf)
            b.append(line(a[0], a[1] + 8, leaf[0], leaf[1] - 16, INK, 1.4))
            b.append(circle(*leaf, 16, fill=BLUE_SOFT, stroke=BLUE, w=1.6))
            b.append(text(leaf[0], leaf[1] + 5, "s′", "ms"))
    # level annotations on the right of the tree
    lx = 495
    rows = (
        (110, BLUE, "根：要求的 V" if zh else "root: the V we want", "π"),
        (152, ORANGE, "① 策略选动作，权重 π(a|s)" if zh else "① policy picks a: π(a|s)", None),
        (240, GREEN, "② 环境掷骰子，权重 P(s′|s,a)，得 R" if zh else "② dynamics pick s′: P(s′|s,a), reward R", None),
        (295, BLUE, "③ 叶子：R + γ · 下一状态的价值" if zh else "③ leaf: R + γ · value of the next state", None),
    )
    for y, col, s, _ in rows:
        b.append(line(lx - 30, y - 5, lx - 8, y - 5, col, 1.2, dash="2 3"))
        b.append(text(lx, y, s, "s", "start", col))
    # backup arrow on the left
    b.append(f'<path d="M100,300 C40,250 70,130 258,112" fill="none" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="5 4" marker-end="url(#ah-muted)"/>')
    b.append(text(40, 205, "加权求和" if zh else "weighted", "xs", "middle"))
    b.append(text(40, 221, "回根节点" if zh else "backup", "xs", "middle"))
    return svg(760, 330, b, title)


# ===================================================== 5. PPO clip =========
def fig_ppo(lang: str) -> str:
    zh = lang == "zh"
    title = "PPO-Clip 目标随概率比率 r 的变化" if zh else "The PPO-Clip objective as a function of the ratio r"
    eps = 0.2
    b = [text(380, 30, title, "h")]
    panels = [(60, 1.0), (420, -1.0)]
    W, Hh = 290, 200
    for px, A in panels:
        py = 70
        xmin, xmax = 0.0, 2.0
        ymin, ymax = (0.0, 2.0) if A > 0 else (-2.0, 0.0)
        X = lambda r: px + (r - xmin) / (xmax - xmin) * W
        Y = lambda v: py + (ymax - v) / (ymax - ymin) * Hh
        # zero-gradient shaded zone
        if A > 0:
            zx0, zx1 = X(1 + eps), X(xmax)
        else:
            zx0, zx1 = X(xmin), X(1 - eps)
        b.append(f'<rect x="{zx0:.1f}" y="{py}" width="{zx1-zx0:.1f}" height="{Hh}" fill="{ORANGE_SOFT}" opacity="0.8"/>')
        # axes
        b.append(line(px, Y(0), px + W + 8, Y(0), MUTED, 1.2, "muted"))
        b.append(line(px, py + Hh, px, py - 8, MUTED, 1.2, "muted"))
        for rv, lab in ((1 - eps, "1−ε"), (1.0, "1"), (1 + eps, "1+ε")):
            b.append(line(X(rv), py, X(rv), py + Hh, FAINT, 1, dash="3 3"))
            b.append(text(X(rv), py + Hh + 18, lab, "ms"))
        b.append(text(px + W + 14, Y(0) + 5, "r", "m", "start"))
        # unclipped r*A (dashed)
        b.append(line(X(0), Y(0), X(2), Y(2 * A), FAINT, 1.6, dash="6 4"))
        # clipped objective
        pts = []
        for i in range(201):
            r = xmin + (xmax - xmin) * i / 200
            val = min(r * A, max(min(r, 1 + eps), 1 - eps) * A)
            pts.append(f"{X(r):.1f},{Y(val):.1f}")
        b.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{BLUE}" stroke-width="3"/>')
        b.append(circle(X(1.0), Y(A), 5, fill=BLUE, stroke=BLUE))
        head = ("优势 A > 0（好动作）" if A > 0 else "优势 A < 0（坏动作）") if zh else ("Advantage A > 0 (good action)" if A > 0 else "Advantage A < 0 (bad action)")
        b.append(text(px + W / 2, py - 18, head, "t b", fill=GREEN if A > 0 else RED))
        if A > 0:
            note = "阴影区 r > 1+ε：目标变平、梯度为 0，不再继续推高" if zh else "shaded r > 1+ε: flat, zero gradient, no further push up"
        else:
            note = "阴影区 r < 1−ε：目标变平、梯度为 0，不再继续压低" if zh else "shaded r < 1−ε: flat, zero gradient, no further push down"
        b.append(text(px + W / 2, py + Hh + 42, note, "xs", fill=ORANGE))
    # legend
    ly = 362
    b.append(line(150, ly - 4, 185, ly - 4, BLUE, 3))
    b.append(text(192, ly, "L^CLIP（实际优化的目标）" if zh else "L^CLIP (the objective PPO optimises)", "s", "start"))
    b.append(line(450, ly - 4, 485, ly - 4, FAINT, 1.6, dash="6 4"))
    b.append(text(492, ly, "未裁剪的 r·A" if zh else "unclipped r·A", "s", "start"))
    b.append(text(380, ly + 22, "● r = 1：新旧策略相同" if zh else "● r = 1: new policy equals the old one", "xs"))
    return svg(760, 400, b, title)


# ===================================================== 6. multimodal =======
def gauss(x, m, s):
    return math.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def fig_multimodal(lang: str) -> str:
    zh = lang == "zh"
    title = "多峰动作分布：高斯策略取平均，扩散策略落在峰上" if zh else "Multi-modal actions: a Gaussian averages, a diffusion policy picks a mode"
    b = [text(380, 30, title, "h")]
    x0, x1, base, hmax = 70, 690, 250, 150
    X = lambda a: x0 + (a + 2.2) / 4.4 * (x1 - x0)
    true = lambda a: 0.5 * gauss(a, -1.1, 0.28) + 0.5 * gauss(a, 1.1, 0.28)
    fit = lambda a: gauss(a, 0.0, math.sqrt(0.28 ** 2 + 1.1 ** 2))
    peak = true(1.1)
    Y = lambda d: base - d / peak * hmax
    pts = [f"{X(a):.1f},{Y(true(a)):.1f}" for a in [(-2.2 + 4.4 * i / 300) for i in range(301)]]
    b.append(f'<polygon points="{X(-2.2):.1f},{base} {" ".join(pts)} {X(2.2):.1f},{base}" fill="#e7e5e0" stroke="{MUTED}" stroke-width="1.2"/>')
    pts = [f"{X(a):.1f},{Y(fit(a)):.1f}" for a in [(-2.2 + 4.4 * i / 300) for i in range(301)]]
    b.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ORANGE}" stroke-width="2.4" stroke-dasharray="7 4"/>')
    b.append(line(x0, base, x1 + 10, base, MUTED, 1.2, "muted"))
    # cups
    for a in (-1.1, 1.1):
        cx = X(a)
        b.append(f'<path d="M{cx-14},{base+14} L{cx+14},{base+14} L{cx+10},{base+40} L{cx-10},{base+40} Z" fill="{BLUE_SOFT}" stroke="{BLUE}" stroke-width="1.4"/>')
        b.append(f'<path d="M{cx+14},{base+20} q10,6 0,14" fill="none" stroke="{BLUE}" stroke-width="1.4"/>')
    # gaussian mean
    b.append(line(X(0), base, X(0), Y(fit(0)) - 6, ORANGE, 1.4, dash="3 3"))
    b.append(f'<path d="M{X(0)},{base+10} l-7,12 l14,0 z" fill="{ORANGE}"/>')
    b.append(text(X(0), base + 42, "高斯均值：撞在两杯之间" if zh else "Gaussian mean: lands between the cups", "s b", fill=ORANGE))
    # diffusion samples (deterministic pseudo-random)
    import random
    rng = random.Random(7)
    for i in range(26):
        m = -1.1 if i % 2 else 1.1
        a = rng.gauss(m, 0.22)
        b.append(circle(X(a), base - 6 - (i % 5) * 7, 4, fill=BLUE, stroke="#ffffff", w=1))
    # legend
    lx, ly = 470, 66
    b.append(f'<rect x="{lx}" y="{ly}" width="22" height="12" fill="#e7e5e0" stroke="{MUTED}"/>')
    b.append(text(lx + 30, ly + 11, "示教中的真实动作分布（两种都对）" if zh else "true action distribution in demos", "s", "start"))
    b.append(line(lx, ly + 30, lx + 22, ly + 30, ORANGE, 2.4, dash="7 4"))
    b.append(text(lx + 30, ly + 35, "MSE / 单高斯拟合" if zh else "MSE / single-Gaussian fit", "s", "start"))
    b.append(circle(lx + 11, ly + 54, 4, fill=BLUE, stroke="#fff", w=1))
    b.append(text(lx + 30, ly + 59, "扩散策略的采样动作" if zh else "samples from a diffusion policy", "s", "start"))
    b.append(text(x0, base + 70, "← 伸向左杯" if zh else "← reach left cup", "xs", "start"))
    b.append(text(x1, base + 70, "伸向右杯 →" if zh else "reach right cup →", "xs", "end"))
    return svg(760, 340, b, title)


# ===================================================== 7. GRPO =============
GRPO_REWARDS = [1, 0, 1, 1, 0, 0]


def grpo_adv(rs):
    m = sum(rs) / len(rs)
    sd = math.sqrt(sum((r - m) ** 2 for r in rs) / len(rs))
    return m, sd, [(r - m) / sd for r in rs]


def fig_grpo(lang: str) -> str:
    zh = lang == "zh"
    title = "PPO 与 GRPO：优势从哪里来" if zh else "PPO vs GRPO: where the advantage comes from"
    b = [text(380, 30, title, "h")]
    # left: PPO
    b.append(text(185, 62, "PPO（需要 Critic）" if zh else "PPO (needs a critic)", "t b"))

    def box(x, y, w, h, lab, fill, stroke, cls="s"):
        b.append(rect(x, y, w, h, fill, stroke, 1.4, 8))
        for i, s in enumerate(lab.split("\n")):
            b.append(text(x + w / 2, y + h / 2 + 5 + (i - (len(lab.split(chr(10))) - 1) / 2) * 16, s, cls, fill=INK))

    box(40, 85, 110, 40, "问题 q" if zh else "prompt q", "#f1efea", MUTED)
    box(40, 150, 110, 40, "策略 π" if zh else "policy π", BLUE_SOFT, BLUE)
    box(40, 215, 110, 40, "一个回答 o" if zh else "one answer o", "#f1efea", MUTED)
    box(210, 150, 130, 40, "价值网络 V" if zh else "value net V", ORANGE_SOFT, ORANGE)
    box(210, 215, 130, 40, "奖励 r" if zh else "reward r", GREEN_SOFT, GREEN)
    box(210, 280, 130, 40, "GAE 优势 Â" if zh else "GAE advantage Â", "#ffffff", INK)
    b.append(line(95, 125, 95, 148, INK, 1.4, "ink"))
    b.append(line(95, 190, 95, 213, INK, 1.4, "ink"))
    b.append(line(150, 235, 208, 235, INK, 1.4, "ink"))
    b.append(line(290, 255, 290, 278, INK, 1.4, "ink"))
    b.append(line(250, 190, 250, 278, ORANGE, 1.4, "orange"))
    b.append(text(275, 141, "与策略同规模、需单独训练" if zh else "as large as the policy, trained separately", "xs", "middle", ORANGE))
    b.append(line(380, 60, 380, 330, FRAME, 1.4))
    # right: GRPO
    b.append(text(575, 62, "GRPO（组内比较，去掉 Critic）" if zh else "GRPO (compare within a group, no critic)", "t b"))
    box(420, 85, 100, 40, "问题 q" if zh else "prompt q", "#f1efea", MUTED)
    box(420, 150, 100, 40, "策略 π" if zh else "policy π", BLUE_SOFT, BLUE)
    b.append(line(470, 125, 470, 148, INK, 1.4, "ink"))
    m, sd, adv = grpo_adv(GRPO_REWARDS)
    ox, oy = 548, 92
    b.append(text(ox + 90, oy - 4, ("采样 G = 6 个回答，验证器打分" if zh else "sample G = 6 answers, verifier scores"), "xs"))
    for i, (r, a) in enumerate(zip(GRPO_REWARDS, adv)):
        yy = oy + 6 + i * 26
        b.append(line(520, 170, ox - 2, yy + 9, FAINT, 1))
        b.append(rect(ox, yy, 40, 18, "#f1efea", MUTED, 1, 4))
        b.append(text(ox + 20, yy + 13, f"o{i+1}", "xs", fill=INK))
        ok = r == 1
        b.append(text(ox + 62, yy + 13, ("✓ 1" if ok else "✗ 0"), "s b", fill=GREEN if ok else RED))
        # advantage bar
        cx = ox + 150
        w = abs(a) * 40
        col = GREEN if a > 0 else RED
        bx = cx if a > 0 else cx - w
        b.append(f'<rect x="{bx:.1f}" y="{yy+2}" width="{w:.1f}" height="14" fill="{col}" opacity="0.8"/>')
        b.append(text(cx + (w + 6 if a > 0 else -w - 6), yy + 13, f"{a:+.0f}".replace("-", "−"), "xs", "start" if a > 0 else "end", col))
    b.append(line(ox + 150, oy + 2, ox + 150, oy + 6 + 6 * 26, MUTED, 1))
    b.append(text(ox + 150, oy + 6 + 6 * 26 + 16, "Âᵢ = (rᵢ − mean) / std", "ms"))
    note = (f"组均值 {m:g} 充当基线：答对的回答被推高，答错的被压低" if zh else
            f"Baseline = group mean ({m:g}): right answers up, wrong ones down")
    b.append(text(575, 330, note, "xs"))
    return svg(760, 350, b, title)


FIGS = {
    "rl-mdp-example": fig_mdp,
    "rl-return-discount": fig_return,
    "rl-gridworld-values": fig_grid,
    "rl-bellman-backup": fig_backup,
    "rl-ppo-clip": fig_ppo,
    "rl-multimodal-actions": fig_multimodal,
    "rl-grpo-vs-ppo": fig_grpo,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        for lang, suffix in (("zh", ""), ("en", "-en")):
            (OUT / f"{name}{suffix}.svg").write_text(fn(lang), encoding="utf-8")
    if "--print" in sys.argv:
        for pol in ({"H": "clean", "L": "charge"}, {"H": "clean", "L": "clean"}, {"H": "clean", "L": "wait"}):
            v = mdp_policy_value(pol)
            print("policy", pol, {k: round(x, 2) for k, x in v.items()})
        v, q = mdp_optimal()
        print("V*", {k: round(x, 2) for k, x in v.items()})
        print("Q*", {f"{s},{a}": round(x, 2) for (s, a), x in q.items()})
        gv, gp = grid_value_iteration()
        for y in range(GRID_H):
            print(" ".join(f"{gv[(x, y)]:+.2f}{gp.get((x, y), '*')}" if (x, y) in gv else "  ####" for x in range(GRID_W)))
        print("GRPO", grpo_adv(GRPO_REWARDS))


if __name__ == "__main__":
    main()
