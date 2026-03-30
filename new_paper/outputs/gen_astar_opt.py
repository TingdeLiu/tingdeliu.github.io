import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.interpolate import CubicSpline

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

COLS, ROWS = 10, 8

GRID_C   = '#ced4da'
PATH_C   = '#bee3f8'
FADED_C  = '#e8f4fd'
LINE_C   = '#2b6cb0'
KEY_C    = '#e53e3e'
CURVE_C  = '#276749'
START_C  = '#38a169'
END_C    = '#e53e3e'
BG       = '#eef2f7'

# A* staircase path (col, row), row 0 = top, row 7 = bottom
PATH = [
    (0,7),(1,7),(1,6),(2,6),(2,5),(3,5),(3,4),(4,4),
    (4,3),(5,3),(5,2),(6,2),(6,1),(7,1),(7,0),(8,0),(9,0)
]
KEYS = [(0,7),(3,4),(6,1),(9,0)]

def cc(col, row):
    """Cell center: row 0 at top → y = ROWS-row-0.5"""
    return (col + 0.5, ROWS - row - 0.5)

def draw_grid(ax, lw=0.7, alpha=1.0):
    for i in range(COLS + 1):
        ax.plot([i, i], [0, ROWS], color=GRID_C, lw=lw, alpha=alpha, zorder=1)
    for j in range(ROWS + 1):
        ax.plot([0, COLS], [j, j], color=GRID_C, lw=lw, alpha=alpha, zorder=1)

def setup(ax, title, subtitle):
    ax.set_xlim(-0.15, COLS + 0.15)
    ax.set_ylim(-0.15, ROWS + 0.15)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#adb5bd')
        sp.set_linewidth(1.5)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#2d3748', pad=7)
    ax.text(0.5, -0.055, subtitle, ha='center', fontsize=10, color='#718096',
            transform=ax.transAxes)

fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=BG)
plt.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.10, wspace=0.10)

# ──────────────────────────────────────────────
# Panel 1 · 原始栅格路径
# ──────────────────────────────────────────────
ax = axs[0]
setup(ax, '① 原始栅格路径', '锯齿形栅格路径（17节点）')
draw_grid(ax)

for c, r in PATH:
    ax.add_patch(patches.Rectangle((c, ROWS-1-r), 1, 1,
                                   lw=0, fc=PATH_C, zorder=2))

xs = [cc(c, r)[0] for c, r in PATH]
ys = [cc(c, r)[1] for c, r in PATH]
ax.plot(xs, ys, color=LINE_C, lw=2.5, solid_joinstyle='round', zorder=5)

sx, sy = cc(0, 7)
gx, gy = cc(9, 0)
ax.plot(sx, sy, 'o', color=START_C, ms=10, zorder=6)
ax.plot(gx, gy, 'o', color=END_C,   ms=10, zorder=6)
ax.text(sx,       sy - 0.72, 'S', ha='center', fontsize=9,
        color='#276749', fontweight='bold', zorder=7)
ax.text(gx + 0.05, gy + 0.58, 'G', ha='center', fontsize=9,
        color='#c53030', fontweight='bold', zorder=7)

# ──────────────────────────────────────────────
# Panel 2 · DP 精简关键点
# ──────────────────────────────────────────────
ax = axs[1]
setup(ax, '② DP 精简关键点', '保留 4 个关键转折点')
draw_grid(ax)

for c, r in PATH:
    ax.add_patch(patches.Rectangle((c, ROWS-1-r), 1, 1,
                                   lw=0, fc=FADED_C, zorder=2))

kx = [cc(c, r)[0] for c, r in KEYS]
ky = [cc(c, r)[1] for c, r in KEYS]
ax.plot(kx, ky, '--', color=KEY_C, lw=2.0, alpha=0.85, zorder=4)

labels  = ['$P_0$', '$P_1$', '$P_2$', '$P_3$']
offsets = [(-0.80, -0.60), (0.30, 0.48), (0.30, 0.48), (0.32, -0.62)]
for (c, r), lbl, (dx, dy) in zip(KEYS, labels, offsets):
    cx, cy = cc(c, r)
    ax.plot(cx, cy, 'o', color=KEY_C, ms=12,
            mec='white', mew=1.8, zorder=6)
    ax.text(cx + dx, cy + dy, lbl, fontsize=9,
            color='#c53030', fontweight='bold', zorder=7)

# ──────────────────────────────────────────────
# Panel 3 · B-样条平滑轨迹
# ──────────────────────────────────────────────
ax = axs[2]
setup(ax, '③ B-样条平滑轨迹', '连续可导的平滑轨迹')
draw_grid(ax)

kx_a = np.array([cc(c, r)[0] for c, r in KEYS])
ky_a = np.array([cc(c, r)[1] for c, r in KEYS])
t    = np.linspace(0, 1, len(KEYS))
csx  = CubicSpline(t, kx_a)
csy  = CubicSpline(t, ky_a)
tf   = np.linspace(0, 1, 400)
sx_c = csx(tf)
sy_c = csy(tf)

# Soft glow under curve
ax.plot(sx_c, sy_c, color=CURVE_C, lw=10, alpha=0.10,
        solid_capstyle='round', zorder=3)
ax.plot(sx_c, sy_c, color=CURVE_C, lw=3.0,
        solid_capstyle='round', zorder=5)

colors = [START_C, CURVE_C, CURVE_C, END_C]
for (c, r), col in zip(KEYS, colors):
    cx, cy = cc(c, r)
    ax.plot(cx, cy, 'o', color=col, ms=9, mec='white', mew=1.8, zorder=6)

# ──────────────────────────────────────────────
# Figure-level arrows  ①→②→③
# ──────────────────────────────────────────────
arrow_kw = dict(ha='center', va='center', fontsize=18,
                color='#555', transform=fig.transFigure)
fig.text(0.345, 0.47, '\u2192', **arrow_kw)
fig.text(0.655, 0.47, '\u2192', **arrow_kw)

fig.text(0.5, 0.96, 'A* 路径优化流程', ha='center',
         fontsize=15, fontweight='bold', color='#1a202c',
         transform=fig.transFigure)

out = 'C:/GitHub/Tingde.Liu.github.io/images/robotics_navigation/astar_optimization.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print('Saved:', out)
