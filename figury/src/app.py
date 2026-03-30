"""Figury 2D — interaktywny kalkulator pól figur geometrycznych."""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrowPatch
import streamlit as st

st.set_page_config(page_title="Figury 2D", page_icon="📐", layout="wide")

# Compact layout — no scrolling
st.markdown("""<style>
    .block-container {padding-top: 1.5rem !important; padding-bottom: 0rem;}
    h1 {font-size: 1.4rem !important; margin-top: 0.5rem !important; margin-bottom: 0.1rem !important; white-space: normal !important; overflow: visible !important; display: block !important;}
    h3 {font-size: 1.1rem !important; margin-bottom: 0.2rem !important;}
    .stMetric {padding: 0.2rem 0 !important;}
    [data-testid="stMetricValue"] {font-size: 1.3rem !important;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.facecolor": "#fafafa",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _new_fig(ncols=1, width=8):
    fig, axes = plt.subplots(1, ncols, figsize=(width, 3))
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, color="gray")
    fig.subplots_adjust(top=0.92, bottom=0.08)
    return fig, axes


def _label(ax, x, y, txt, fontsize=13, **kw):
    ax.text(x, y, txt, ha="center", va="center", fontsize=fontsize, fontweight="bold", **kw)


# ---------------------------------------------------------------------------
# 1. Circle
# ---------------------------------------------------------------------------

def draw_circle(vals):
    r = vals["r"]
    fig, (ax1, ax2) = _new_fig(2, 10)

    # -- left: filled circle with radius --
    circle = plt.Circle((0, 0), r, color="#6CB4EE", alpha=0.4)
    ax1.add_patch(circle)
    ax1.plot([0, r], [0, 0], "r-", lw=2)
    _label(ax1, r / 2, -r * 0.12, f"r = {r}", color="red")
    ax1.set_xlim(-r * 1.3, r * 1.3)
    ax1.set_ylim(-r * 1.3, r * 1.3)
    ax1.set_title("Koło", fontsize=14)

    # -- right: pizza → rectangle proof --
    n = 12  # number of slices
    colors = ["#6CB4EE", "#A8D8EA"]
    theta = np.linspace(0, 2 * np.pi, n + 1)
    slice_w = (np.pi * r) / (n / 2)  # width each slice occupies in rectangle

    for i in range(n):
        t0, t1 = theta[i], theta[i + 1]
        col = colors[i % 2]
        # draw each slice as a triangle approximation in the rectangle
        cx = (i - n / 2 + 0.5) * slice_w / 1.0
        if i < n // 2:
            # bottom row: point up
            tri = Polygon([
                [i * slice_w, 0],
                [(i + 1) * slice_w, 0],
                [(i + 0.5) * slice_w, r],
            ], closed=True, fc=col, ec="gray", lw=0.5, alpha=0.6)
        else:
            j = i - n // 2
            # top row: point down, interleaved
            tri = Polygon([
                [(j + 0.5) * slice_w, 0],
                [(j + 1.5) * slice_w, 0],
                [(j + 1) * slice_w, r],
            ], closed=True, fc=col, ec="gray", lw=0.5, alpha=0.6)
        ax2.add_patch(tri)

    total_w = (n // 2) * slice_w
    ax2.set_xlim(-slice_w * 0.5, total_w + slice_w * 0.5)
    ax2.set_ylim(-r * 0.3, r * 1.4)
    # labels
    ax2.annotate("", xy=(total_w, -r * 0.1), xytext=(0, -r * 0.1),
                 arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    _label(ax2, total_w / 2, -r * 0.22, f"πr = {np.pi * r:.1f}", color="red")
    ax2.annotate("", xy=(-slice_w * 0.3, r), xytext=(-slice_w * 0.3, 0),
                 arrowprops=dict(arrowstyle="<->", color="blue", lw=1.5))
    _label(ax2, -slice_w * 0.3, r / 2, f"r={r}", color="blue", rotation=90)
    ax2.set_title("Plasterki → prawie-prostokąt", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Triangle
# ---------------------------------------------------------------------------

def draw_triangle(vals):
    b, h = vals["b"], vals["h"]
    fig, (ax1, ax2) = _new_fig(2, 10)

    # choose an asymmetric apex so it looks like a general triangle
    apex_x = b * 0.35
    tri_pts = np.array([[0, 0], [b, 0], [apex_x, h]])

    # -- left: single triangle --
    ax1.add_patch(Polygon(tri_pts, closed=True, fc="#77DD77", alpha=0.4, ec="green", lw=2))
    # height line
    ax1.plot([apex_x, apex_x], [0, h], "k--", lw=1)
    _label(ax1, apex_x + b * 0.08, h / 2, f"h={h}")
    _label(ax1, b / 2, -h * 0.1, f"b={b}")
    ax1.set_xlim(-b * 0.15, b * 1.15)
    ax1.set_ylim(-h * 0.25, h * 1.2)
    ax1.set_title("Trójkąt", fontsize=14)

    # -- right: two triangles → parallelogram --
    ax2.add_patch(Polygon(tri_pts, closed=True, fc="#77DD77", alpha=0.4, ec="green", lw=2))
    # rotated copy (180° around midpoint of base-to-apex edge)
    mid = np.array([(b + apex_x) / 2, h / 2])
    tri2_pts = 2 * mid - tri_pts  # 180° rotation around mid
    ax2.add_patch(Polygon(tri2_pts, closed=True, fc="#FFD580", alpha=0.4, ec="orange", lw=2))
    ax2.set_xlim(-b * 0.15, b * 1.3)
    ax2.set_ylim(-h * 0.25, h * 1.3)
    ax2.set_title("2 trójkąty → równoległobok", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Square
# ---------------------------------------------------------------------------

def draw_square(vals):
    a = int(vals["a"])
    fig, (ax,) = _new_fig(1, 6)

    ax.add_patch(mpatches.Rectangle((0, 0), a, a, fc="#FDFD96", alpha=0.5, ec="#CCCC00", lw=2))
    # unit grid
    if a <= 15:
        for i in range(1, a):
            ax.plot([i, i], [0, a], color="gray", lw=0.5, alpha=0.4)
            ax.plot([0, a], [i, i], color="gray", lw=0.5, alpha=0.4)
    _label(ax, a / 2, -a * 0.1, f"a = {a}")
    _label(ax, -a * 0.12, a / 2, f"a = {a}", rotation=90)
    _label(ax, a / 2, a / 2, f"{a} × {a} = {a * a}", fontsize=16,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlim(-a * 0.2, a * 1.15)
    ax.set_ylim(-a * 0.2, a * 1.15)
    ax.set_title("Kwadrat", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Rectangle
# ---------------------------------------------------------------------------

def draw_rectangle(vals):
    a, b = int(vals["a"]), int(vals["b"])
    fig, (ax,) = _new_fig(1, 7)

    ax.add_patch(mpatches.Rectangle((0, 0), a, b, fc="#FFB347", alpha=0.4, ec="#CC8800", lw=2))
    # grid
    if max(a, b) <= 15:
        for i in range(1, a):
            ax.plot([i, i], [0, b], color="gray", lw=0.5, alpha=0.4)
        for j in range(1, b):
            ax.plot([0, a], [j, j], color="gray", lw=0.5, alpha=0.4)
        # highlight first row
        ax.add_patch(mpatches.Rectangle((0, 0), a, 1, fc="#FF8C00", alpha=0.25, ec="none"))
        # highlight first column
        ax.add_patch(mpatches.Rectangle((0, 0), 1, b, fc="#FF4500", alpha=0.2, ec="none"))
    _label(ax, a / 2, -b * 0.08, f"a = {a}")
    _label(ax, -a * 0.1, b / 2, f"b = {b}", rotation=90)
    _label(ax, a / 2, b / 2, f"{a} × {b} = {a * b}", fontsize=16,
           bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_xlim(-a * 0.2, a * 1.15)
    ax.set_ylim(-b * 0.2, b * 1.15)
    ax.set_title("Prostokąt", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Rhombus (side + angle)
# ---------------------------------------------------------------------------

def draw_rhombus(vals):
    a = vals["a"]
    alpha_deg = vals["alpha"]
    alpha = math.radians(alpha_deg)
    h = a * math.sin(alpha)
    # vertices: bottom-left at origin, sides of length a, angle alpha
    pts = np.array([
        [0, 0],
        [a, 0],
        [a + a * math.cos(alpha), h],
        [a * math.cos(alpha), h],
    ])
    cx, cy = pts.mean(axis=0)

    fig, (ax,) = _new_fig(1, 7)
    # 4 triangles from diagonals
    d_mid = (pts[0] + pts[2]) / 2  # center (diagonals cross here)
    tri_colors = ["#DDA0DD", "#B0E0E6", "#FFB6C1", "#98FB98"]
    for i in range(4):
        tri = [d_mid, pts[i], pts[(i + 1) % 4]]
        ax.add_patch(Polygon(tri, closed=True, fc=tri_colors[i], ec="gray", lw=1, alpha=0.6))
    # diagonals
    ax.plot([pts[0][0], pts[2][0]], [pts[0][1], pts[2][1]], "k--", lw=1.5)
    ax.plot([pts[1][0], pts[3][0]], [pts[1][1], pts[3][1]], "k--", lw=1.5)
    # outline
    ax.add_patch(Polygon(pts, closed=True, fc="none", ec="#8B008B", lw=2))
    # height line
    hx = pts[3][0]
    ax.plot([hx, hx], [0, h], "r--", lw=1.5)
    _label(ax, hx - a * 0.12, h / 2, f"h={h:.1f}", fontsize=11, color="red")
    # labels
    _label(ax, a / 2, -h * 0.13, f"a = {a}")
    # angle arc
    arc_r = a * 0.2
    arc_theta = np.linspace(0, alpha, 20)
    ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), "b-", lw=1.5)
    _label(ax, arc_r * 1.4 * math.cos(alpha / 2), arc_r * 1.4 * math.sin(alpha / 2),
           f"{alpha_deg}°", fontsize=10, color="blue")
    margin = max(a, h) * 0.2
    ax.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
    ax.set_ylim(-margin, h + margin)
    ax.set_title("Romb", fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Trapezoid
# ---------------------------------------------------------------------------

def draw_trapezoid(vals):
    a, b, h = vals["a"], vals["b"], vals["h"]
    fig, (ax1, ax2) = _new_fig(2, 10)

    # a = top base (shorter), b = bottom base (longer for typical look)
    offset = (b - a) / 2
    trap_pts = np.array([[0, 0], [b, 0], [offset + a, h], [offset, h]])

    # -- left: single trapezoid --
    ax1.add_patch(Polygon(trap_pts, closed=True, fc="#FF6961", alpha=0.35, ec="#CC0000", lw=2))
    ax1.plot([offset, offset], [0, h], "k--", lw=1)
    _label(ax1, b / 2, -h * 0.12, f"b = {b}")
    _label(ax1, offset + a / 2, h + h * 0.1, f"a = {a}")
    _label(ax1, offset - b * 0.07, h / 2, f"h={h}", fontsize=11)
    ax1.set_xlim(-b * 0.15, b * 1.15)
    ax1.set_ylim(-h * 0.25, h * 1.3)
    ax1.set_title("Trapez", fontsize=14)

    # -- right: two trapezoids → parallelogram --
    ax2.add_patch(Polygon(trap_pts, closed=True, fc="#FF6961", alpha=0.35, ec="#CC0000", lw=2))
    # flipped copy: rotate 180° around midpoint of right non-parallel side
    mid = np.array([(b + offset + a) / 2, h / 2])
    trap2_pts = 2 * mid - trap_pts
    ax2.add_patch(Polygon(trap2_pts, closed=True, fc="#FFD580", alpha=0.4, ec="#CC8800", lw=2))
    # label combined base
    all_x = np.concatenate([trap_pts[:, 0], trap2_pts[:, 0]])
    all_y = np.concatenate([trap_pts[:, 1], trap2_pts[:, 1]])
    ax2.set_xlim(all_x.min() - b * 0.15, all_x.max() + b * 0.15)
    ax2.set_ylim(all_y.min() - h * 0.25, all_y.max() + h * 0.25)
    ax2.set_title("2 trapezy → równoległobok", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Parallelogram
# ---------------------------------------------------------------------------

def draw_parallelogram(vals):
    a = vals["a"]
    b = vals["b"]
    alpha_deg = vals["alpha"]
    alpha = math.radians(alpha_deg)
    h = b * math.sin(alpha)

    pts = np.array([
        [0, 0],
        [a, 0],
        [a + b * math.cos(alpha), h],
        [b * math.cos(alpha), h],
    ])

    fig, (ax1, ax2) = _new_fig(2, 10)

    # -- left: parallelogram with height --
    ax1.add_patch(Polygon(pts, closed=True, fc="#87CEEB", alpha=0.4, ec="#4682B4", lw=2))
    # height line
    hx = pts[3][0]
    ax1.plot([hx, hx], [0, h], "r--", lw=1.5)
    _label(ax1, hx - a * 0.08, h / 2, f"h={h:.1f}", fontsize=11, color="red")
    _label(ax1, a / 2, -h * 0.14, f"a = {a}")
    _label(ax1, (pts[0][0] + pts[3][0]) / 2 - a * 0.08, h / 2, f"b={b}", fontsize=11)
    # angle arc
    arc_r = min(a, b) * 0.2
    arc_theta = np.linspace(0, alpha, 20)
    ax1.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), "b-", lw=1.5)
    _label(ax1, arc_r * 1.6 * math.cos(alpha / 2), arc_r * 1.6 * math.sin(alpha / 2),
           f"{alpha_deg}°", fontsize=10, color="blue")
    margin = max(a, h) * 0.15
    ax1.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + margin)
    ax1.set_ylim(-margin * 1.5, h + margin)
    ax1.set_title("Równoległobok", fontsize=14)

    # -- right: proof — cut triangle, move to other side → rectangle --
    # original parallelogram outline (faint)
    ax2.add_patch(Polygon(pts, closed=True, fc="#87CEEB", alpha=0.15, ec="#4682B4", lw=1, ls="--"))
    # the rectangle part (between the two height lines)
    offset = b * math.cos(alpha)
    rect_pts = np.array([[offset, 0], [a, 0], [a, h], [offset, h]])
    ax2.add_patch(Polygon(rect_pts, closed=True, fc="#87CEEB", alpha=0.4, ec="#4682B4", lw=2))
    # left triangle (to be cut)
    left_tri = np.array([[0, 0], [offset, 0], [offset, h]])
    ax2.add_patch(Polygon(left_tri, closed=True, fc="#FFD580", alpha=0.5, ec="#CC8800", lw=2, ls="--"))
    # moved triangle on the right
    right_tri = left_tri + np.array([a, 0])
    ax2.add_patch(Polygon(right_tri, closed=True, fc="#FFD580", alpha=0.5, ec="#CC8800", lw=2))
    # arrow showing the move
    ax2.annotate("", xy=(a + offset / 2, h * 0.5), xytext=(offset / 2, h * 0.5),
                 arrowprops=dict(arrowstyle="->", color="#CC8800", lw=2))
    ax2.set_xlim(pts[:, 0].min() - margin, pts[:, 0].max() + a * 0.15 + margin)
    ax2.set_ylim(-margin * 1.5, h + margin)
    ax2.set_title("Odetnij trójkąt → prostokąt a×h", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Shape configuration
# ---------------------------------------------------------------------------

SHAPES = {
    "🔵 Koło": {
        "title": "Pole koła",
        "sliders": [
            {"label": "Promień r", "key": "r", "min": 1.0, "max": 20.0, "step": 0.5},
        ],
        "formula": lambda v: rf"A = \pi \cdot {v['r']}^2 = {math.pi * v['r']**2:.2f}",
        "area": lambda v: math.pi * v["r"] ** 2,
        "intuition": (
            "Pomaluj koło na kolorowo, rozcinaj na coraz cieńsze plasterki "
            "i ułóż je w prawie-prostokąt. Wysokość = r, szerokość = πr, "
            "więc pole = r × πr = πr²"
        ),
        "draw": draw_circle,
    },
    "🔺 Trójkąt": {
        "title": "Pole trójkąta",
        "sliders": [
            {"label": "Podstawa b", "key": "b", "min": 1.0, "max": 20.0, "step": 0.5},
            {"label": "Wysokość h", "key": "h", "min": 1.0, "max": 20.0, "step": 0.5},
        ],
        "formula": lambda v: rf"A = \frac{{1}}{{2}} \cdot {v['b']} \cdot {v['h']} = {0.5 * v['b'] * v['h']:.2f}",
        "area": lambda v: 0.5 * v["b"] * v["h"],
        "intuition": (
            "Dwa takie same trójkąty zawsze składają się w równoległobok. "
            "Pole równoległoboku to b×h, więc jeden trójkąt to połowa."
        ),
        "draw": draw_triangle,
    },
    "🟨 Kwadrat": {
        "title": "Pole kwadratu",
        "sliders": [
            {"label": "Bok a", "key": "a", "min": 1.0, "max": 20.0, "step": 1.0},
        ],
        "formula": lambda v: rf"A = {int(v['a'])}^2 = {int(v['a'])**2}",
        "area": lambda v: v["a"] ** 2,
        "intuition": (
            "Ile małych kwadratów mieści się w tym dużym? "
            "Policz wzdłuż boku — a małych. Policz boki — a rzędów. "
            "Razem: a × a = a²"
        ),
        "draw": draw_square,
    },
    "🟧 Prostokąt": {
        "title": "Pole prostokąta",
        "sliders": [
            {"label": "Szerokość a", "key": "a", "min": 1.0, "max": 20.0, "step": 1.0},
            {"label": "Wysokość b", "key": "b", "min": 1.0, "max": 20.0, "step": 1.0},
        ],
        "formula": lambda v: rf"A = {int(v['a'])} \times {int(v['b'])} = {int(v['a']) * int(v['b'])}",
        "area": lambda v: v["a"] * v["b"],
        "intuition": (
            "Narysuj a kropek w górę i b kropek w prawo. "
            "Ile jest wszystkich kropek? a rzędów po b = a×b"
        ),
        "draw": draw_rectangle,
    },
    "🔷 Romb": {
        "title": "Pole rombu",
        "sliders": [
            {"label": "Bok a", "key": "a", "min": 1.0, "max": 20.0, "step": 0.5},
            {"label": "Kąt α (°)", "key": "alpha", "min": 10.0, "max": 170.0, "step": 1.0},
        ],
        "formula": lambda v: rf"A = {v['a']}^2 \cdot \sin({v['alpha']:.0f}°) = {v['a']**2 * math.sin(math.radians(v['alpha'])):.2f}",
        "area": lambda v: v["a"] ** 2 * math.sin(math.radians(v["alpha"])),
        "intuition": (
            "Romb to równoległobok o równych bokach. "
            "Wysokość h = a·sin(α). Pole = a × h = a² · sin(α). "
            "Przesuń kąt — zobacz jak romb się spłaszcza, a pole maleje!"
        ),
        "draw": draw_rhombus,
    },
    "🔺 Trapez": {
        "title": "Pole trapezu",
        "sliders": [
            {"label": "Podstawa górna a", "key": "a", "min": 1.0, "max": 20.0, "step": 0.5},
            {"label": "Podstawa dolna b", "key": "b", "min": 1.0, "max": 20.0, "step": 0.5},
            {"label": "Wysokość h", "key": "h", "min": 1.0, "max": 15.0, "step": 0.5},
        ],
        "formula": lambda v: rf"A = \frac{{1}}{{2}} \cdot ({v['a']} + {v['b']}) \cdot {v['h']} = {0.5 * (v['a'] + v['b']) * v['h']:.2f}",
        "area": lambda v: 0.5 * (v["a"] + v["b"]) * v["h"],
        "intuition": (
            "Dwa takie same trapezy składają się w równoległobok "
            "o podstawie a+b i wysokości h. Pole równoległoboku to (a+b)×h, "
            "więc jeden trapez to połowa."
        ),
        "draw": draw_trapezoid,
    },
    "▱ Równoległobok": {
        "title": "Pole równoległoboku",
        "sliders": [
            {"label": "Podstawa a", "key": "a", "min": 1.0, "max": 20.0, "step": 0.5},
            {"label": "Bok b", "key": "b", "min": 1.0, "max": 15.0, "step": 0.5},
            {"label": "Kąt α (°)", "key": "alpha", "min": 10.0, "max": 170.0, "step": 1.0},
        ],
        "formula": lambda v: rf"A = {v['a']} \cdot {v['b']} \cdot \sin({v['alpha']:.0f}°) = {v['a'] * v['b'] * math.sin(math.radians(v['alpha'])):.2f}",
        "area": lambda v: v["a"] * v["b"] * math.sin(math.radians(v["alpha"])),
        "intuition": (
            "Odetnij trójkąt z jednej strony i przyklej go z drugiej — "
            "dostaniesz prostokąt o podstawie a i wysokości h = b·sin(α). "
            "Pole = a × h. Zmień kąt — zobacz jak się zmienia wysokość!"
        ),
        "draw": draw_parallelogram,
    },
}


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    shape_names = list(SHAPES.keys())

    with st.sidebar:
        st.header("📐 Figury 2D")
        # Initialize session state if needed
        if "shape" not in st.session_state:
            st.session_state.shape = shape_names[0]
        
        selected = st.selectbox(
            "Wybierz figurę",
            shape_names,
            key="shape_selector",
        )
        st.session_state.shape = selected

    cfg = SHAPES[selected]
    st.title(f"{selected.split()[0]} {cfg['title']}")

    col1, col2 = st.columns([1, 2])

    # -- left column: sliders + formula --
    vals = {}
    with col1:
        st.subheader("Zmień wymiary")
        for s in cfg["sliders"]:
            vals[s["key"]] = st.slider(
                s["label"],
                min_value=s["min"],
                max_value=s["max"],
                value=(s["min"] + s["max"]) / 2,
                step=s["step"],
            )
        area = cfg["area"](vals)
        st.metric("Pole", f"{area:.2f}")
        st.latex(cfg["formula"](vals))

    # -- right column: visualization + intuition --
    with col2:
        fig = cfg["draw"](vals)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info(cfg["intuition"])


main()
