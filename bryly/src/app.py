"""Bryły 3D — interaktywny kalkulator objętości i pól powierzchni brył."""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import streamlit as st

st.set_page_config(page_title="Bryły 3D", page_icon="🧊", layout="wide")

st.markdown("""<style>
    .block-container {padding-top: 0.5rem; padding-bottom: 0rem;}
    h1 {font-size: 1.4rem !important; margin-bottom: 0.1rem !important; white-space: normal !important; overflow: visible !important;}
    h3 {font-size: 1.0rem !important; margin-bottom: 0.1rem !important;}
    .stMetric {padding: 0.1rem 0 !important;}
    [data-testid="stMetricValue"] {font-size: 1.2rem !important;}
    [data-testid="stMetricLabel"] {font-size: 0.8rem !important;}
    .stSlider {margin-bottom: -0.5rem !important;}
    [data-testid="stExpander"] {margin-bottom: 0 !important;}
    .element-container {margin-bottom: 0.2rem !important;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.facecolor": "#fafafa",
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_axes_equal(ax):
    """Set 3D axes to equal aspect ratio."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2
    for ctr, setter in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter([ctr - max_range, ctr + max_range])


def _new_combined_fig():
    """One figure: 3D subplot (left, wider) + 2D cross-section (right)."""
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.3)
    ax3d = fig.add_subplot(gs[0], projection="3d")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.set_facecolor("#fafafa")
    ax_cs = fig.add_subplot(gs[1])
    ax_cs.set_aspect("equal")
    ax_cs.grid(True, alpha=0.3, color="gray")
    return fig, ax3d, ax_cs


def _cylinder_mesh(r, h, z0=0, n=40):
    """Return (X, Y, Z) mesh for a cylinder surface."""
    theta = np.linspace(0, 2 * np.pi, n)
    z = np.array([z0, z0 + h])
    Theta, Z = np.meshgrid(theta, z)
    X = r * np.cos(Theta)
    Y = r * np.sin(Theta)
    return X, Y, Z


def _prism_edges(ax, a, b, h, x0=0, y0=0, z0=0, **kw):
    """Draw 12 wireframe edges of a rectangular prism."""
    kw.setdefault("color", "gray")
    kw.setdefault("alpha", 0.2)
    kw.setdefault("linestyle", "--")
    corners = np.array([
        [0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0],
        [0, 0, h], [a, 0, h], [a, b, h], [0, b, h],
    ]) + [x0, y0, z0]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), **kw)


# ---------------------------------------------------------------------------
# 1. Sphere
# ---------------------------------------------------------------------------

def draw_sphere(vals):
    r = vals["r"]
    cs_h = vals["cs_h"]
    ghost = vals.get("ghost", False)

    fig, ax, ax_cs = _new_combined_fig()

    # Sphere surface
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    X = r * np.outer(np.cos(u), np.sin(v))
    Y = r * np.outer(np.sin(u), np.sin(v))
    Z = r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(X, Y, Z, alpha=0.3, color="#6CB4EE")

    # Radius line
    ax.plot([0, r], [0, 0], [0, 0], "r-", lw=2)
    ax.text(r / 2, 0, -r * 0.15, f"r={r}", color="red", fontsize=11, fontweight="bold")

    # Cross-section circle on the 3D view
    cs_r2 = max(0, r**2 - cs_h**2)
    cs_r = math.sqrt(cs_r2)
    theta = np.linspace(0, 2 * np.pi, 60)
    ax.plot(cs_r * np.cos(theta), cs_r * np.sin(theta), cs_h, "m-", lw=2)
    ax.plot([0], [0], [cs_h], "mo", ms=4)

    # Ghost cylinder
    if ghost:
        Xc, Yc, Zc = _cylinder_mesh(r, 2 * r, z0=-r)
        ax.plot_wireframe(Xc, Yc, Zc, alpha=0.12, color="gray", linestyle="--")
        disk_theta = np.linspace(0, 2 * np.pi, 30)
        disk_r = np.linspace(0, r, 2)
        DT, DR = np.meshgrid(disk_theta, disk_r)
        DX, DY = DR * np.cos(DT), DR * np.sin(DT)
        for zz in [-r, r]:
            ax.plot_wireframe(DX, DY, np.full_like(DX, zz), alpha=0.08, color="gray")

    ax.set_title("Kula", fontsize=13)
    _set_axes_equal(ax)

    # Cross-section 2D
    if cs_r > 0.01:
        circle = plt.Circle((0, 0), cs_r, color="#6CB4EE", alpha=0.4)
        ax_cs.add_patch(circle)
        ax_cs.plot([0, cs_r], [0, 0], "m-", lw=1.5)
        ax_cs.text(cs_r / 2, -cs_r * 0.2, f"r={cs_r:.2f}", fontsize=10, color="purple")
        lim = r * 1.2
    else:
        ax_cs.plot(0, 0, "mo", ms=6)
        ax_cs.text(0.3, 0, "punkt (r=0)", fontsize=10)
        lim = 2
    ax_cs.set_xlim(-lim, lim)
    ax_cs.set_ylim(-lim, lim)
    ax_cs.set_title(f"Przekrój h={cs_h:.1f}", fontsize=11)

    return fig


# ---------------------------------------------------------------------------
# 2. Cone
# ---------------------------------------------------------------------------

def draw_cone(vals):
    r = vals["r"]
    h = vals["h"]
    cs_t = vals["cs_h"]
    ghost = vals.get("ghost", False)

    fig, ax, ax_cs = _new_combined_fig()

    # Cone surface
    z = np.linspace(0, h, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    Z, Theta = np.meshgrid(z, theta)
    R = r * (1 - Z / h)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    ax.plot_surface(X, Y, Z, alpha=0.3, color="#C8A2C8")

    # Base circle
    ax.plot(r * np.cos(theta), r * np.sin(theta), 0, "k-", lw=1)

    # Height + radius labels
    ax.plot([0, 0], [0, 0], [0, h], "b--", lw=1.5)
    ax.text(0.3, 0, h / 2, f"h={h}", color="blue", fontsize=10, fontweight="bold")
    ax.plot([0, r], [0, 0], [0, 0], "r-", lw=2)
    ax.text(r / 2, 0, -h * 0.06, f"r={r}", color="red", fontsize=10, fontweight="bold")

    # Slant height
    l = math.sqrt(r**2 + h**2)
    ax.plot([0, r], [0, 0], [h, 0], "g--", lw=1.5)
    ax.text(r / 2 + 0.3, 0, h / 2, f"l={l:.1f}", color="green", fontsize=10)

    # Cross-section ring
    cs_r = r * max(0, 1 - cs_t / h) if h > 0 else 0
    cs_theta = np.linspace(0, 2 * np.pi, 60)
    ax.plot(cs_r * np.cos(cs_theta), cs_r * np.sin(cs_theta), cs_t, "m-", lw=2)

    # Ghost cylinder
    if ghost:
        Xc, Yc, Zc = _cylinder_mesh(r, h)
        ax.plot_wireframe(Xc, Yc, Zc, alpha=0.12, color="gray", linestyle="--")

    ax.set_title("Stożek", fontsize=13)
    _set_axes_equal(ax)

    # Cross-section 2D
    if cs_r > 0.01:
        circle = plt.Circle((0, 0), cs_r, color="#C8A2C8", alpha=0.4)
        ax_cs.add_patch(circle)
        ax_cs.text(cs_r / 2, -cs_r * 0.25, f"r={cs_r:.2f}", fontsize=10, color="purple")
        lim = r * 1.2
    else:
        ax_cs.plot(0, 0, "mo", ms=6)
        ax_cs.text(0.3, 0, "wierzchołek (r=0)", fontsize=10)
        lim = 2
    ax_cs.set_xlim(-lim, lim)
    ax_cs.set_ylim(-lim, lim)
    ax_cs.set_title(f"Przekrój t={cs_t:.1f}", fontsize=11)

    return fig


# ---------------------------------------------------------------------------
# 3. Rectangular Prism
# ---------------------------------------------------------------------------

def _prism_faces(a, b, c):
    """Return list of 6 faces (each a list of 4 vertices) for a rectangular prism."""
    v = np.array([
        [0,0,0],[a,0,0],[a,b,0],[0,b,0],
        [0,0,c],[a,0,c],[a,b,c],[0,b,c],
    ])
    faces = [[v[j] for j in f] for f in
             [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)]]
    return faces


def draw_prism(vals):
    a, b, c = vals["a"], vals["b"], vals["c"]
    cs_t = vals["cs_h"]

    fig, ax, ax_cs = _new_combined_fig()

    faces = _prism_faces(a, b, c)
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=0.25, facecolor="#FFB347", edgecolor="k", linewidth=0.5))

    # Space diagonal
    d = math.sqrt(a**2 + b**2 + c**2)
    ax.plot([0, a], [0, b], [0, c], "r--", lw=2)
    ax.text(a / 2 + 0.3, b / 2, c / 2, f"d={d:.1f}", color="red", fontsize=10)

    # Labels
    ax.text(a / 2, -b * 0.15, 0, f"a={a}", fontsize=10, fontweight="bold")
    ax.text(a + a * 0.05, b / 2, 0, f"b={b}", fontsize=10, fontweight="bold")
    ax.text(-a * 0.1, 0, c / 2, f"c={c}", fontsize=10, fontweight="bold")

    # Cross-section plane indicator
    ax.plot([0, a, a, 0, 0], [0, 0, b, b, 0],
            [cs_t]*5, "m-", lw=2)

    ax.set_title("Prostopadłościan", fontsize=13)
    _set_axes_equal(ax)

    # Cross-section 2D
    rect = plt.Rectangle((0, 0), a, b, color="#FFB347", alpha=0.4)
    ax_cs.add_patch(rect)
    ax_cs.set_xlim(-1, a + 1)
    ax_cs.set_ylim(-1, b + 1)
    ax_cs.text(a / 2, b / 2, f"{a}×{b}", ha="center", va="center", fontsize=12)
    ax_cs.set_title(f"Przekrój t={cs_t:.1f}", fontsize=11)

    return fig


# ---------------------------------------------------------------------------
# 4. Cube
# ---------------------------------------------------------------------------

def draw_cube(vals):
    a = vals["a"]
    cs_t = vals["cs_h"]

    fig, ax, ax_cs = _new_combined_fig()

    faces = _prism_faces(a, a, a)
    ax.add_collection3d(Poly3DCollection(
        faces, alpha=0.2, facecolor="#FDFD96", edgecolor="k", linewidth=0.5))

    # Unit cubes grid for small a
    if a <= 5:
        ia = int(a)
        for i in range(ia + 1):
            for j in range(ia + 1):
                ax.plot([i, i], [0, a], [j, j], color="gray", alpha=0.15, lw=0.5)
                ax.plot([0, a], [i, i], [j, j], color="gray", alpha=0.15, lw=0.5)
                ax.plot([i, i], [j, j], [0, a], color="gray", alpha=0.15, lw=0.5)

    # Space diagonal
    d = a * math.sqrt(3)
    ax.plot([0, a], [0, a], [0, a], "r--", lw=2)
    ax.text(a / 2 + 0.3, a / 2, a / 2, f"d={d:.1f}", color="red", fontsize=10)

    ax.text(a / 2, -a * 0.12, 0, f"a={a}", fontsize=10, fontweight="bold")

    # Cross-section plane
    ax.plot([0, a, a, 0, 0], [0, 0, a, a, 0], [cs_t]*5, "m-", lw=2)

    ax.set_title("Sześcian", fontsize=13)
    _set_axes_equal(ax)

    # Cross-section 2D
    rect = plt.Rectangle((0, 0), a, a, color="#FDFD96", alpha=0.5, edgecolor="k")
    ax_cs.add_patch(rect)
    ax_cs.set_xlim(-1, a + 1)
    ax_cs.set_ylim(-1, a + 1)
    ax_cs.text(a / 2, a / 2, f"{a}×{a}", ha="center", va="center", fontsize=12)
    ax_cs.set_title(f"Przekrój t={cs_t:.1f}", fontsize=11)

    return fig


# ---------------------------------------------------------------------------
# 5. Cylinder
# ---------------------------------------------------------------------------

def draw_cylinder(vals):
    r = vals["r"]
    h = vals["h"]
    cs_t = vals["cs_h"]

    fig, ax, ax_cs = _new_combined_fig()

    # Lateral surface
    X, Y, Z = _cylinder_mesh(r, h)
    ax.plot_surface(X, Y, Z, alpha=0.25, color="#FFD580")

    # Top and bottom caps
    theta = np.linspace(0, 2 * np.pi, 40)
    dr = np.linspace(0, r, 2)
    DT, DR = np.meshgrid(theta, dr)
    DX, DY = DR * np.cos(DT), DR * np.sin(DT)
    ax.plot_surface(DX, DY, np.zeros_like(DX), alpha=0.25, color="#FFD580")
    ax.plot_surface(DX, DY, np.full_like(DX, h), alpha=0.25, color="#FFD580")

    # Labels
    ax.plot([0, r], [0, 0], [0, 0], "r-", lw=2)
    ax.text(r / 2, 0, -h * 0.06, f"r={r}", color="red", fontsize=10, fontweight="bold")
    ax.plot([r * 1.05, r * 1.05], [0, 0], [0, h], "b--", lw=1.5)
    ax.text(r * 1.15, 0, h / 2, f"h={h}", color="blue", fontsize=10, fontweight="bold")

    # Cross-section ring
    ax.plot(r * np.cos(theta), r * np.sin(theta), cs_t, "m-", lw=2)

    ax.set_title("Walec", fontsize=13)
    _set_axes_equal(ax)

    # Cross-section 2D: circle
    circle = plt.Circle((0, 0), r, color="#FFD580", alpha=0.4)
    ax_cs.add_patch(circle)
    ax_cs.set_xlim(-r * 1.3, r * 1.3)
    ax_cs.set_ylim(-r * 1.3, r * 1.3)
    ax_cs.set_title(f"Przekrój t={cs_t:.1f}", fontsize=11)
    ax_cs.text(0, 0, f"r={r}", ha="center", va="center", fontsize=10)
    rect_w = 2 * math.pi * r
    ax_cs.text(0, -r * 0.85, f"Rozwinięcie: {rect_w:.1f}×{h}", ha="center",
               fontsize=8, color="gray")

    return fig


# ---------------------------------------------------------------------------
# 6. Pyramid
# ---------------------------------------------------------------------------

def draw_pyramid(vals):
    a = vals["a"]
    h = vals["h"]
    cs_t = vals["cs_h"]
    ghost = vals.get("ghost", False)

    fig, ax, ax_cs = _new_combined_fig()

    apex = [a / 2, a / 2, h]
    base = np.array([[0,0,0],[a,0,0],[a,a,0],[0,a,0]])

    # 4 triangular faces + base
    tri_faces = [[base[i].tolist(), base[(i+1) % 4].tolist(), apex] for i in range(4)]
    base_face = [base[i].tolist() for i in range(4)]

    ax.add_collection3d(Poly3DCollection(
        tri_faces, alpha=0.25, facecolor="#FF6961", edgecolor="k", linewidth=0.5))
    ax.add_collection3d(Poly3DCollection(
        [base_face], alpha=0.2, facecolor="#FF6961", edgecolor="k", linewidth=0.5))

    # Height line
    ax.plot([a/2, a/2], [a/2, a/2], [0, h], "b--", lw=1.5)
    ax.text(a/2 + 0.3, a/2, h/2, f"h={h}", color="blue", fontsize=10, fontweight="bold")

    # Base label
    ax.text(a/2, -a*0.1, 0, f"a={a}", fontsize=10, fontweight="bold")

    # Slant height (apothem of face)
    sl = math.sqrt((a/2)**2 + h**2)
    ax.plot([a/2, a/2], [0, a/2], [0, h], "g--", lw=1.5)
    ax.text(a/2 + 0.3, a/4, h/2, f"l={sl:.1f}", color="green", fontsize=9)

    # Cross-section square on 3D
    cs_edge = a * max(0, 1 - cs_t / h) if h > 0 else 0
    if cs_edge > 0.01:
        off = (a - cs_edge) / 2
        sq = np.array([
            [off, off, cs_t], [off + cs_edge, off, cs_t],
            [off + cs_edge, off + cs_edge, cs_t], [off, off + cs_edge, cs_t], [off, off, cs_t],
        ])
        ax.plot(sq[:, 0], sq[:, 1], sq[:, 2], "m-", lw=2)

    # Ghost prism
    if ghost:
        _prism_edges(ax, a, a, h)

    ax.set_title("Ostrosłup", fontsize=13)
    _set_axes_equal(ax)

    # Cross-section 2D
    if cs_edge > 0.01:
        rect = plt.Rectangle((-cs_edge/2, -cs_edge/2), cs_edge, cs_edge,
                              color="#FF6961", alpha=0.4, edgecolor="k")
        ax_cs.add_patch(rect)
        ax_cs.text(0, 0, f"{cs_edge:.1f}×{cs_edge:.1f}", ha="center", va="center", fontsize=10)
        lim = a / 2 + 1
    else:
        ax_cs.plot(0, 0, "mo", ms=6)
        ax_cs.text(0.3, 0, "wierzchołek", fontsize=10)
        lim = 2
    ax_cs.set_xlim(-lim, lim)
    ax_cs.set_ylim(-lim, lim)
    ax_cs.set_title(f"Przekrój t={cs_t:.1f}", fontsize=11)

    return fig


# ---------------------------------------------------------------------------
# SOLIDS dictionary
# ---------------------------------------------------------------------------

SOLIDS = {
    "🔵 Kula": {
        "title": "Objętość i pole kuli",
        "sliders": [
            {"label": "Promień r", "key": "r", "min": 1.0, "max": 15.0, "step": 0.5},
        ],
        "cs_slider": {"label": "Wysokość przekroju h", "key": "cs_h",
                       "min_fn": lambda v: -v["r"], "max_fn": lambda v: v["r"], "step": 0.5},
        "has_ghost": True,
        "volume_formula": lambda v: (
            rf"V = \tfrac{{4}}{{3}}\pi \cdot {v['r']:.1f}^3 = {4/3*math.pi*v['r']**3:.2f}"
        ),
        "surface_formula": lambda v: (
            rf"S = 4\pi \cdot {v['r']:.1f}^2 = {4*math.pi*v['r']**2:.2f}"
        ),
        "volume": lambda v: 4 / 3 * math.pi * v["r"] ** 3,
        "surface": lambda v: 4 * math.pi * v["r"] ** 2,
        "intuition": (
            "Archimedes odkrył, że kula idealnie mieści się w walcu. "
            "Objętość kuli to dokładnie 2/3 objętości tego walca. "
            "Przekrój kuli na dowolnej wysokości to zawsze koło — ale jego promień "
            "się zmienia! Użyj twierdzenia Pitagorasa: promień przekroju = √(r² − h²)."
        ),
        "draw": draw_sphere,
    },
    "🟣 Stożek": {
        "title": "Objętość i pole stożka",
        "sliders": [
            {"label": "Promień r", "key": "r", "min": 1.0, "max": 15.0, "step": 0.5},
            {"label": "Wysokość h", "key": "h", "min": 1.0, "max": 20.0, "step": 0.5},
        ],
        "cs_slider": {"label": "Wysokość przekroju t", "key": "cs_h",
                       "min_fn": lambda v: 0.0, "max_fn": lambda v: v["h"], "step": 0.5},
        "has_ghost": True,
        "volume_formula": lambda v: (
            rf"V = \tfrac{{1}}{{3}}\pi \cdot {v['r']:.1f}^2 \cdot {v['h']:.1f}"
            rf" = {1/3*math.pi*v['r']**2*v['h']:.2f}"
        ),
        "surface_formula": lambda v: (
            rf"S = \pi \cdot {v['r']:.1f}^2 + \pi \cdot {v['r']:.1f} \cdot {math.sqrt(v['r']**2+v['h']**2):.1f}"
            rf" = {math.pi*v['r']**2 + math.pi*v['r']*math.sqrt(v['r']**2+v['h']**2):.2f}"
        ),
        "volume": lambda v: 1 / 3 * math.pi * v["r"] ** 2 * v["h"],
        "surface": lambda v: math.pi * v["r"] ** 2 + math.pi * v["r"] * math.sqrt(v["r"]**2 + v["h"]**2),
        "intuition": (
            "Stożek to dokładnie 1/3 walca o tym samym promieniu i wysokości. "
            "Przekrój na dowolnej wysokości to kółko — im wyżej kroisz, tym mniejsze. "
            "Na czubku promień = 0. Tworzącą (l) obliczasz z Pitagorasa: l = √(r² + h²)."
        ),
        "draw": draw_cone,
    },
    "🟧 Prostopadłościan": {
        "title": "Objętość i pole prostopadłościanu",
        "sliders": [
            {"label": "Długość a", "key": "a", "min": 1.0, "max": 15.0, "step": 0.5},
            {"label": "Szerokość b", "key": "b", "min": 1.0, "max": 15.0, "step": 0.5},
            {"label": "Wysokość c", "key": "c", "min": 1.0, "max": 15.0, "step": 0.5},
        ],
        "cs_slider": {"label": "Wysokość przekroju t", "key": "cs_h",
                       "min_fn": lambda v: 0.0, "max_fn": lambda v: v["c"], "step": 0.5},
        "has_ghost": False,
        "volume_formula": lambda v: (
            rf"V = {v['a']:.1f} \cdot {v['b']:.1f} \cdot {v['c']:.1f}"
            rf" = {v['a']*v['b']*v['c']:.2f}"
        ),
        "surface_formula": lambda v: (
            rf"S = 2({v['a']:.1f}\cdot{v['b']:.1f} + {v['b']:.1f}\cdot{v['c']:.1f}"
            rf" + {v['a']:.1f}\cdot{v['c']:.1f}) = {2*(v['a']*v['b']+v['b']*v['c']+v['a']*v['c']):.2f}"
        ),
        "volume": lambda v: v["a"] * v["b"] * v["c"],
        "surface": lambda v: 2 * (v["a"]*v["b"] + v["b"]*v["c"] + v["a"]*v["c"]),
        "intuition": (
            "To pudełko na klocki. Na dnie mieści się a×b klocków. "
            "Takich warstw jest c. Objętość = a·b·c. "
            "Powierzchnia to 3 pary ścian: góra+dół, przód+tył, lewo+prawo. "
            "Przekątna przestrzenna? Pitagoras dwa razy: "
            "najpierw przekątna dna √(a²+b²), potem z nią i wysokością √(a²+b²+c²)."
        ),
        "draw": draw_prism,
    },
    "🟨 Sześcian": {
        "title": "Objętość i pole sześcianu",
        "sliders": [
            {"label": "Krawędź a", "key": "a", "min": 1.0, "max": 15.0, "step": 0.5},
        ],
        "cs_slider": {"label": "Wysokość przekroju t", "key": "cs_h",
                       "min_fn": lambda v: 0.0, "max_fn": lambda v: v["a"], "step": 0.5},
        "has_ghost": False,
        "volume_formula": lambda v: (
            rf"V = {v['a']:.1f}^3 = {v['a']**3:.2f}"
        ),
        "surface_formula": lambda v: (
            rf"S = 6 \cdot {v['a']:.1f}^2 = {6*v['a']**2:.2f}"
        ),
        "volume": lambda v: v["a"] ** 3,
        "surface": lambda v: 6 * v["a"] ** 2,
        "intuition": (
            "Ile małych kostek zmieści się w środku? Na dnie masz a×a = a² kostek. "
            "Takich warstw jest a. Razem: a·a·a = a³. "
            "Ścian jest 6, każda to kwadrat a×a. "
            "Przekątna przestrzenna = a√3 — Pitagoras użyty dwa razy."
        ),
        "draw": draw_cube,
    },
    "🟠 Walec": {
        "title": "Objętość i pole walca",
        "sliders": [
            {"label": "Promień r", "key": "r", "min": 1.0, "max": 15.0, "step": 0.5},
            {"label": "Wysokość h", "key": "h", "min": 1.0, "max": 20.0, "step": 0.5},
        ],
        "cs_slider": {"label": "Wysokość przekroju t", "key": "cs_h",
                       "min_fn": lambda v: 0.0, "max_fn": lambda v: v["h"], "step": 0.5},
        "has_ghost": False,
        "volume_formula": lambda v: (
            rf"V = \pi \cdot {v['r']:.1f}^2 \cdot {v['h']:.1f}"
            rf" = {math.pi*v['r']**2*v['h']:.2f}"
        ),
        "surface_formula": lambda v: (
            rf"S = 2\pi \cdot {v['r']:.1f}^2 + 2\pi \cdot {v['r']:.1f} \cdot {v['h']:.1f}"
            rf" = {2*math.pi*v['r']**2 + 2*math.pi*v['r']*v['h']:.2f}"
        ),
        "volume": lambda v: math.pi * v["r"] ** 2 * v["h"],
        "surface": lambda v: 2 * math.pi * v["r"] ** 2 + 2 * math.pi * v["r"] * v["h"],
        "intuition": (
            "Walec to stos kółek. Każde kółko ma pole π·r². "
            "Takich kółek (warstw) jest h. Objętość = π·r²·h. "
            "Powierzchnia boczna? Rozwiń ją w myślach — "
            "to prostokąt o bokach 2πr (obwód koła) i h!"
        ),
        "draw": draw_cylinder,
    },
    "🔴 Ostrosłup": {
        "title": "Objętość i pole ostrosłupa",
        "sliders": [
            {"label": "Krawędź podstawy a", "key": "a", "min": 1.0, "max": 15.0, "step": 0.5},
            {"label": "Wysokość h", "key": "h", "min": 1.0, "max": 20.0, "step": 0.5},
        ],
        "cs_slider": {"label": "Wysokość przekroju t", "key": "cs_h",
                       "min_fn": lambda v: 0.0, "max_fn": lambda v: v["h"], "step": 0.5},
        "has_ghost": True,
        "volume_formula": lambda v: (
            rf"V = \tfrac{{1}}{{3}} \cdot {v['a']:.1f}^2 \cdot {v['h']:.1f}"
            rf" = {1/3*v['a']**2*v['h']:.2f}"
        ),
        "surface_formula": lambda v: (
            rf"S = {v['a']:.1f}^2 + 2 \cdot {v['a']:.1f} \cdot {math.sqrt((v['a']/2)**2+v['h']**2):.1f}"
            rf" = {v['a']**2 + 2*v['a']*math.sqrt((v['a']/2)**2+v['h']**2):.2f}"
        ),
        "volume": lambda v: 1 / 3 * v["a"] ** 2 * v["h"],
        "surface": lambda v: v["a"]**2 + 2 * v["a"] * math.sqrt((v["a"]/2)**2 + v["h"]**2),
        "intuition": (
            "Ostrosłup to dokładnie 1/3 graniastosłupa o tej samej podstawie i wysokości. "
            "Przekrój na dowolnej wysokości to kwadrat — im wyżej, tym mniejszy. "
            "Wysokość ściany bocznej (apotema) obliczysz z Pitagorasa: √((a/2)² + h²)."
        ),
        "draw": draw_pyramid,
    },
}


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    solid_names = list(SOLIDS.keys())

    with st.sidebar:
        st.header("🧊 Bryły 3D")
        selected = st.selectbox(
            "Wybierz bryłę",
            solid_names,
            index=solid_names.index(st.session_state.get("solid", solid_names[0])),
        )
        st.session_state.solid = selected

    cfg = SOLIDS[selected]
    st.title(f"{selected.split()[0]} {cfg['title']}")

    col1, col2 = st.columns([1, 2])

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

        # Cross-section slider with dynamic range
        cs = cfg["cs_slider"]
        cs_min = float(cs["min_fn"](vals))
        cs_max = float(cs["max_fn"](vals))
        if cs_max <= cs_min:
            cs_max = cs_min + cs["step"]
        vals["cs_h"] = st.slider(
            cs["label"],
            min_value=cs_min,
            max_value=cs_max,
            value=(cs_min + cs_max) / 2,
            step=cs["step"],
        )

        # Ghost overlay checkbox
        if cfg.get("has_ghost"):
            vals["ghost"] = st.checkbox("Pokaż bryłę okalającą", value=True)
        else:
            vals["ghost"] = False

        # Metrics
        vol = cfg["volume"](vals)
        surf = cfg["surface"](vals)
        st.metric("Objętość", f"{vol:.2f}")
        st.metric("Pole powierzchni", f"{surf:.2f}")

        # Formulas
        st.latex(cfg["volume_formula"](vals))
        st.latex(cfg["surface_formula"](vals))

    with col2:
        fig = cfg["draw"](vals)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.info(cfg["intuition"])


main()
