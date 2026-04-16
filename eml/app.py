"""
EML Symbolic Regression & Illustration — Streamlit App
======================================================
Explores the EML operator: eml(x, y) = exp(x) - ln(y),
a universal primitive for continuous mathematics.

Based on: Odrzywołek (2026), "All elementary functions from a single operator"
arXiv:2603.21852
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io, math, textwrap, json

# ─────────────────────────────────────────────
# Constants & helpers
# ─────────────────────────────────────────────

KNOWN_CONSTANTS = [
    ("0",    0.0),
    ("1",    1.0),
    ("-1",  -1.0),
    ("2",    2.0),
    ("-2",  -2.0),
    ("3",    3.0),
    ("0.5",  0.5),
    ("-0.5",-0.5),
    ("e",    math.e),
    ("-e",  -math.e),
    ("π",    math.pi),
    ("-π",  -math.pi),
    ("ln2",  math.log(2)),
    ("1/e",  1.0/math.e),
    ("√2",   math.sqrt(2)),
    ("π/2",  math.pi/2),
]

def snap_to_known(c: float, tol: float = 0.12) -> tuple[str, float]:
    """Snap a float to the nearest recognized constant within tolerance."""
    best_label, best_val, best_dist = f"{c:.4f}", c, float("inf")
    for label, val in KNOWN_CONSTANTS:
        d = abs(c - val)
        if d < best_dist and d < tol:
            best_label, best_val, best_dist = label, val, d
    return best_label, best_val


# ─────────────────────────────────────────────
# Differentiable EML tree (PyTorch)
# ─────────────────────────────────────────────

class EMLLeaf(nn.Module):
    """Leaf that learns to be either `x` or a constant via a soft gate."""

    def __init__(self):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(np.random.uniform(-1.5, 1.5)))
        self.const_val  = nn.Parameter(torch.tensor(np.random.uniform(-2.0, 2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate_logit * 4.0)   # steeper → sharper decision
        return g * x + (1.0 - g) * self.const_val

    def snap_expr(self) -> str:
        g = torch.sigmoid(self.gate_logit * 4.0).item()
        if g > 0.75:
            return "x"
        elif g < 0.25:
            label, _ = snap_to_known(self.const_val.item())
            return label
        else:
            # Mixed — show both
            c_label, _ = snap_to_known(self.const_val.item())
            return f"({g:.1f}·x+{1-g:.1f}·{c_label})"

    def snap_math(self) -> str:
        return self.snap_expr()


class EMLNode(nn.Module):
    """Internal node: eml(left, right) = exp(left) − ln(right)."""

    def __init__(self, depth: int):
        super().__init__()
        if depth <= 1:
            self.left  = EMLLeaf()
            self.right = EMLLeaf()
        else:
            self.left  = EMLNode(depth - 1)
            self.right = EMLNode(depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.left(x)
        b = self.right(x)
        a_safe = torch.clamp(a, -18.0, 18.0)
        b_safe = torch.clamp(b, 1e-12, None)
        return torch.exp(a_safe) - torch.log(b_safe)

    def snap_expr(self) -> str:
        return f"eml({self.left.snap_expr()}, {self.right.snap_expr()})"

    def snap_math(self) -> str:
        return f"exp({self.left.snap_math()}) − ln({self.right.snap_math()})"


class EMLTree(nn.Module):
    """Complete trainable EML tree of given depth."""

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        if depth == 0:
            self.root = EMLLeaf()
        else:
            self.root = EMLNode(depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.root(x)

    @property
    def n_leaves(self) -> int:
        return 2 ** self.depth if self.depth > 0 else 1

    def eml_string(self) -> str:
        return self.root.snap_expr()

    def math_string(self) -> str:
        return self.root.snap_math()


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_eml_tree(
    x_np: np.ndarray,
    y_np: np.ndarray,
    depth: int,
    n_restarts: int = 10,
    n_epochs: int = 3000,
    lr: float = 0.01,
    progress_callback=None,
) -> tuple:
    """Train with multiple restarts, return best model + loss history."""
    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)

    best_model = None
    best_loss  = float("inf")
    best_hist  = []

    total_steps = n_restarts * n_epochs
    step = 0

    for r in range(n_restarts):
        model = EMLTree(depth)
        opt   = optim.Adam(model.parameters(), lr=lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)
        hist  = []

        for epoch in range(n_epochs):
            opt.zero_grad()
            y_pred = model(x_t)
            loss = nn.MSELoss()(y_pred, y_t)

            if torch.isnan(loss) or torch.isinf(loss):
                hist.append(float("inf"))
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            hist.append(loss.item())

            step += 1
            if progress_callback and step % 50 == 0:
                progress_callback(step / total_steps)

        final = hist[-1] if hist else float("inf")
        if final < best_loss:
            best_loss  = final
            best_model = model
            best_hist  = hist

    if progress_callback:
        progress_callback(1.0)

    return best_model, best_loss, best_hist


# ─────────────────────────────────────────────
# Tree DOT visualization (regression mode)
# ─────────────────────────────────────────────

def _tree_to_dot(node, node_id=0) -> tuple[list[str], list[str], int]:
    """Recursively build DOT edges and labels."""
    nodes_stmts = []
    edge_stmts  = []

    if isinstance(node, EMLLeaf):
        label = node.snap_expr()
        nodes_stmts.append(
            f'  n{node_id} [label="{label}", shape=ellipse, '
            f'style=filled, fillcolor="#dbeafe", fontname="monospace"];'
        )
        return nodes_stmts, edge_stmts, node_id

    # EMLNode
    nodes_stmts.append(
        f'  n{node_id} [label="eml", shape=box, style=filled, '
        f'fillcolor="#fef3c7", fontname="monospace", fontsize=11];'
    )
    left_id  = node_id + 1
    ln, le, last = _tree_to_dot(node.left, left_id)
    nodes_stmts.extend(ln)
    edge_stmts.extend(le)
    edge_stmts.append(f"  n{node_id} -> n{left_id};")

    right_id = last + 1
    rn, re, last = _tree_to_dot(node.right, right_id)
    nodes_stmts.extend(rn)
    edge_stmts.extend(re)
    edge_stmts.append(f"  n{node_id} -> n{right_id};")

    return nodes_stmts, edge_stmts, last


def tree_to_dot(model: EMLTree) -> str:
    ns, es, _ = _tree_to_dot(model.root)
    return "digraph EML {\n  rankdir=TB;\n  node [fontsize=10];\n" + \
           "\n".join(ns) + "\n" + "\n".join(es) + "\n}"


# ─────────────────────────────────────────────
# Preset data generators (regression mode)
# ─────────────────────────────────────────────

PRESETS = {
    "exp(x)":      (lambda x: np.exp(x),               (-2, 2, 60)),
    "ln(x)":       (lambda x: np.log(x),               (0.1, 5, 60)),
    "x²":          (lambda x: x**2,                    (-3, 3, 60)),
    "2x + 1":      (lambda x: 2*x + 1,                 (-3, 3, 60)),
    "√x":          (lambda x: np.sqrt(x),              (0.01, 5, 60)),
    "1/x":         (lambda x: 1.0/x,                   (0.2, 5, 60)),
    "exp(x) − 1":  (lambda x: np.exp(x) - 1,           (-2, 2, 60)),
    "e^x − ln(x)": (lambda x: np.exp(x) - np.log(x),  (0.1, 3, 60)),
    "x³":          (lambda x: x**3,                    (-2, 2, 60)),
    "exp(−x²)":    (lambda x: np.exp(-x**2),           (-3, 3, 60)),
}


# ═════════════════════════════════════════════
# Illustration mode: expression helpers
# ═════════════════════════════════════════════

def eval_expr(expr, x_arr):
    """Evaluate a nested expression tuple on a numpy array.

    Expression format:
        ('x',)           — the variable x
        ('const', v)     — a numeric constant
        ('eml', L, R)    — eml(L, R) = exp(L) - ln(R)
    """
    if expr[0] == "x":
        return x_arr.astype(np.float64).copy()
    elif expr[0] == "const":
        return np.full_like(x_arr, expr[1], dtype=np.float64)
    elif expr[0] == "eml":
        a = eval_expr(expr[1], x_arr)
        b = eval_expr(expr[2], x_arr)
        a_safe = np.clip(a, -18, 18)
        b_safe = np.clip(b, 1e-12, None)
        return np.exp(a_safe) - np.log(b_safe)
    return np.zeros_like(x_arr, dtype=np.float64)


def expr_to_eml(expr) -> str:
    """Convert expression tuple to eml notation string."""
    if expr[0] == "x":
        return "x"
    elif expr[0] == "const":
        v = expr[1]
        if v == 1.0:
            return "1"
        if abs(v - math.e) < 1e-9:
            return "e"
        return f"{v:.4g}"
    elif expr[0] == "eml":
        return f"eml({expr_to_eml(expr[1])}, {expr_to_eml(expr[2])})"
    return "?"


def expr_to_math(expr) -> str:
    """Convert expression tuple to expanded math notation."""
    if expr[0] == "x":
        return "x"
    elif expr[0] == "const":
        v = expr[1]
        if v == 1.0:
            return "1"
        if abs(v - math.e) < 1e-9:
            return "e"
        return f"{v:.4g}"
    elif expr[0] == "eml":
        l_str = expr_to_math(expr[1])
        r_str = expr_to_math(expr[2])
        return f"exp({l_str}) - ln({r_str})"
    return "?"


def expr_to_dot_str(expr) -> str:
    """Build a complete graphviz DOT string for an expression tree."""
    def _build(ex, nid=0):
        nodes, edges = [], []
        if ex[0] in ("x", "const"):
            label = expr_to_eml(ex)
            nodes.append(
                f'  n{nid} [label="{label}", shape=ellipse, '
                f'style=filled, fillcolor="#dbeafe", fontname="monospace"];'
            )
            return nodes, edges, nid
        # eml node
        nodes.append(
            f'  n{nid} [label="eml", shape=box, style=filled, '
            f'fillcolor="#fef3c7", fontname="monospace", fontsize=11];'
        )
        lid = nid + 1
        ln, le, last = _build(ex[1], lid)
        nodes.extend(ln); edges.extend(le)
        edges.append(f"  n{nid} -> n{lid};")
        rid = last + 1
        rn, re, last = _build(ex[2], rid)
        nodes.extend(rn); edges.extend(re)
        edges.append(f"  n{nid} -> n{rid};")
        return nodes, edges, last

    ns, es, _ = _build(expr)
    return (
        "digraph EML {\n  rankdir=TB;\n  node [fontsize=10];\n"
        + "\n".join(ns) + "\n" + "\n".join(es) + "\n}"
    )


# ─────────────────────────────────────────────
# Decomposition catalog
# ─────────────────────────────────────────────
# Each decomposition is algebraically verified.
# Grammar: S -> 1 | x | eml(S, S)
# eml(a, b) = exp(a) - ln(b)

DECOMPOSITIONS = [
    {
        "name": "exp(x)",
        "formula": "e^x",
        "eml": "eml(x, 1)",
        "expr": ("eml", ("x",), ("const", 1.0)),
        "ref_fn": lambda x: np.exp(x),
        "domain": (-2, 3),
        "depth": 1,
        "steps": [
            ("Goal", "Construct exp(x) using only eml"),
            ("Apply", "eml(x, 1) = exp(x) - ln(1)"),
            ("Simplify", "ln(1) = 0, so eml(x, 1) = exp(x)"),
        ],
    },
    {
        "name": "e (Euler's number)",
        "formula": "e",
        "eml": "eml(1, 1)",
        "expr": ("eml", ("const", 1.0), ("const", 1.0)),
        "ref_fn": lambda x: np.full_like(x, math.e),
        "domain": (-2, 2),
        "depth": 1,
        "steps": [
            ("Goal", "Produce the constant e"),
            ("Apply", "eml(1, 1) = exp(1) - ln(1) = e - 0 = e"),
        ],
    },
    {
        "name": "exp(x) - ln(x)",
        "formula": "e^x - ln(x)",
        "eml": "eml(x, x)",
        "expr": ("eml", ("x",), ("x",)),
        "ref_fn": lambda x: np.exp(x) - np.log(x),
        "domain": (0.1, 3),
        "depth": 1,
        "steps": [
            ("Goal", "Construct exp(x) - ln(x)"),
            ("Apply", "eml(x, x) = exp(x) - ln(x) -- direct from the definition!"),
        ],
    },
    {
        "name": "e - ln(x)",
        "formula": "e - ln(x)",
        "eml": "eml(1, x)",
        "expr": ("eml", ("const", 1.0), ("x",)),
        "ref_fn": lambda x: math.e - np.log(x),
        "domain": (0.1, 5),
        "depth": 1,
        "steps": [
            ("Goal", "Construct e - ln(x)"),
            ("Apply", "eml(1, x) = exp(1) - ln(x) = e - ln(x)"),
        ],
    },
    {
        "name": "e - x (linear!)",
        "formula": "e - x",
        "eml": "eml(1, eml(x, 1))",
        "expr": ("eml", ("const", 1.0), ("eml", ("x",), ("const", 1.0))),
        "ref_fn": lambda x: math.e - x,
        "domain": (-2, 5),
        "depth": 2,
        "steps": [
            ("Goal", "Construct the linear function e - x"),
            ("Inner", "eml(x, 1) = exp(x)"),
            ("Outer", "eml(1, exp(x)) = exp(1) - ln(exp(x)) = e - x"),
            ("Key insight", "ln and exp cancel, leaving a linear function!"),
        ],
    },
    {
        "name": "exp(x) - 1",
        "formula": "e^x - 1",
        "eml": "eml(x, eml(1, 1))",
        "expr": ("eml", ("x",), ("eml", ("const", 1.0), ("const", 1.0))),
        "ref_fn": lambda x: np.exp(x) - 1,
        "domain": (-2, 2),
        "depth": 2,
        "steps": [
            ("Goal", "Construct exp(x) - 1"),
            ("Inner", "eml(1, 1) = e"),
            ("Outer", "eml(x, e) = exp(x) - ln(e) = exp(x) - 1"),
        ],
    },
    {
        "name": "exp(exp(x))",
        "formula": "e^(e^x)",
        "eml": "eml(eml(x, 1), 1)",
        "expr": ("eml", ("eml", ("x",), ("const", 1.0)), ("const", 1.0)),
        "ref_fn": lambda x: np.exp(np.exp(x)),
        "domain": (-2, 1.5),
        "depth": 2,
        "steps": [
            ("Goal", "Construct the double exponential exp(exp(x))"),
            ("Inner", "eml(x, 1) = exp(x)"),
            ("Outer", "eml(exp(x), 1) = exp(exp(x)) - ln(1) = exp(exp(x))"),
        ],
    },
    {
        "name": "exp(e) (constant)",
        "formula": "e^e",
        "eml": "eml(eml(1, 1), 1)",
        "expr": ("eml", ("eml", ("const", 1.0), ("const", 1.0)), ("const", 1.0)),
        "ref_fn": lambda x: np.full_like(x, math.exp(math.e)),
        "domain": (-2, 2),
        "depth": 2,
        "steps": [
            ("Goal", "Produce the constant e^e ≈ 15.15"),
            ("Inner", "eml(1, 1) = e"),
            ("Outer", "eml(e, 1) = exp(e) - ln(1) = exp(e)"),
        ],
    },
    {
        "name": "ln(x)",
        "formula": "ln(x)",
        "eml": "eml(1, eml(eml(1, x), 1))",
        "expr": (
            "eml",
            ("const", 1.0),
            ("eml", ("eml", ("const", 1.0), ("x",)), ("const", 1.0)),
        ),
        "ref_fn": lambda x: np.log(x),
        "domain": (0.1, 5),
        "depth": 3,
        "steps": [
            ("Goal", "Construct ln(x) -- the deepest basic derivation"),
            ("Step 1", "eml(1, x) = exp(1) - ln(x) = e - ln(x)"),
            (
                "Step 2",
                "eml(e - ln(x), 1) = exp(e - ln(x)) = e^e / x",
            ),
            ("Step 3", "eml(1, e^e/x) = e - ln(e^e/x) = e - (e - ln(x)) = ln(x)"),
            ("Key insight", "Three levels of nesting to 'invert' the exponential"),
        ],
    },
    {
        "name": "0 (zero from nothing)",
        "formula": "0",
        "eml": "eml(1, eml(eml(1, 1), 1))",
        "expr": (
            "eml",
            ("const", 1.0),
            ("eml", ("eml", ("const", 1.0), ("const", 1.0)), ("const", 1.0)),
        ),
        "ref_fn": lambda x: np.zeros_like(x),
        "domain": (-2, 2),
        "depth": 3,
        "steps": [
            ("Goal", "Produce the constant 0 from only the constant 1 and eml"),
            ("Step 1", "eml(1, 1) = e"),
            ("Step 2", "eml(e, 1) = exp(e)"),
            ("Step 3", "eml(1, exp(e)) = e - ln(exp(e)) = e - e = 0"),
            ("Key insight", "Zero emerges from self-cancellation at depth 3"),
        ],
    },
]


# ─────────────────────────────────────────────
# Illustration mode: NAND analogy
# ─────────────────────────────────────────────

def _nand_gate_dot(label, inputs, color="#e0f2fe"):
    """Small graphviz for a NAND-based construction."""
    nodes = []
    edges = []
    nid = 0
    # Build tree from a nested structure: ('NAND', left, right) or a string leaf
    def _build(tree, parent_id=None):
        nonlocal nid
        my_id = nid
        nid += 1
        if isinstance(tree, str):
            nodes.append(
                f'  n{my_id} [label="{tree}", shape=ellipse, '
                f'style=filled, fillcolor="#dbeafe", fontname="monospace", fontsize=10];'
            )
        else:
            nodes.append(
                f'  n{my_id} [label="NAND", shape=box, style=filled, '
                f'fillcolor="#e0e7ff", fontname="monospace", fontsize=10];'
            )
            for child in tree[1:]:
                child_id = nid
                _build(child)
                edges.append(f"  n{my_id} -> n{child_id};")
        return my_id

    _build(inputs)
    return (
        f"digraph {label} {{\n  rankdir=TB;\n  node [fontsize=9];\n"
        + "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}"
    )


def render_nand_analogy():
    """Render the NAND vs EML universal-primitive comparison."""
    st.markdown("""
### The Universal Primitive

In digital logic, **NAND** is a *universal gate*: every Boolean function
(AND, OR, NOT, XOR, ...) can be built from NAND alone. The **EML operator**
plays exactly the same role for continuous mathematics: every elementary
function can be built from `eml` alone.
""")

    col_nand, col_eml = st.columns(2)

    # ── Left: NAND ──
    with col_nand:
        st.markdown("#### NAND gate (digital logic)")
        st.markdown("**Definition:** NAND(A, B) = NOT(A AND B)")
        st.markdown("""
| A | B | NAND |
|:-:|:-:|:----:|
| 0 | 0 |  1   |
| 0 | 1 |  1   |
| 1 | 0 |  1   |
| 1 | 1 |  0   |
""")
        st.markdown("**NOT** from NAND: &ensp; NOT(A) = NAND(A, A)")
        st.graphviz_chart(
            _nand_gate_dot("NOT", ("NAND", "A", "A")),
            use_container_width=True,
        )

        st.markdown("**AND** from NAND: &ensp; AND = NAND(NAND(A,B), NAND(A,B))")
        st.graphviz_chart(
            _nand_gate_dot("AND", ("NAND", ("NAND", "A", "B"), ("NAND", "A", "B"))),
            use_container_width=True,
        )

        st.markdown("**OR** from NAND: &ensp; OR = NAND(NAND(A,A), NAND(B,B))")
        st.graphviz_chart(
            _nand_gate_dot("OR", ("NAND", ("NAND", "A", "A"), ("NAND", "B", "B"))),
            use_container_width=True,
        )

    # ── Right: EML ──
    with col_eml:
        st.markdown("#### EML operator (continuous math)")
        st.markdown(r"**Definition:** $\text{eml}(a, b) = e^a - \ln(b)$")
        st.markdown("""
| Left (a) | Right (b) | eml(a, b) |
|:--------:|:---------:|:---------:|
| x | 1 | exp(x) |
| 1 | 1 | e |
| 1 | x | e - ln(x) |
| x | x | exp(x) - ln(x) |
""")
        st.markdown("**exp(x)** from EML:")
        st.graphviz_chart(
            expr_to_dot_str(("eml", ("x",), ("const", 1.0))),
            use_container_width=True,
        )

        st.markdown("**e - x** from EML (a linear function!):")
        st.graphviz_chart(
            expr_to_dot_str(
                ("eml", ("const", 1.0), ("eml", ("x",), ("const", 1.0)))
            ),
            use_container_width=True,
        )

        st.markdown("**ln(x)** from EML (depth 3):")
        st.graphviz_chart(
            expr_to_dot_str(
                ("eml", ("const", 1.0),
                 ("eml", ("eml", ("const", 1.0), ("x",)), ("const", 1.0)))
            ),
            use_container_width=True,
        )

    st.info(
        "Just as **every** Boolean circuit reduces to NAND gates, **every** "
        "elementary function (exp, ln, sin, cos, sqrt, arithmetic, ...) "
        "reduces to nested EML applications. EML is the continuous-math NAND.",
        icon="\u2194",
    )


# ─────────────────────────────────────────────
# Illustration mode: Decomposition explorer
# ─────────────────────────────────────────────

def render_explorer():
    """Interactive explorer showing how known functions decompose into EML."""
    st.markdown("---")
    st.markdown("### Decomposition Explorer")
    st.markdown(
        "Select a function to see how it decomposes into EML operations, "
        "with a step-by-step derivation and live verification plot."
    )

    names = [d["name"] for d in DECOMPOSITIONS]
    idx = st.selectbox("Choose a function", range(len(names)),
                       format_func=lambda i: f"{names[i]}  (depth {DECOMPOSITIONS[i]['depth']})")
    d = DECOMPOSITIONS[idx]

    col_info, col_tree = st.columns([1, 1])

    with col_info:
        st.markdown(f"**Target function:** &ensp; {d['formula']}")
        st.markdown(f"**EML expression:** &ensp; `{d['eml']}`")
        st.markdown(f"**Tree depth:** &ensp; {d['depth']}")
        st.markdown("")
        st.markdown("**Derivation:**")
        for label, text in d["steps"]:
            if label.startswith("Key"):
                st.markdown(f"> *{label}: {text}*")
            else:
                st.markdown(f"- **{label}:** &ensp; {text}")

    with col_tree:
        st.graphviz_chart(expr_to_dot_str(d["expr"]), use_container_width=True)

    # Verification plot
    lo, hi = d["domain"]
    x_arr = np.linspace(lo, hi, 300).astype(np.float64)
    y_ref = d["ref_fn"](x_arr)
    y_eml = eval_expr(d["expr"], x_arr)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x_arr, y_ref, color="#3b82f6", linewidth=2.5, alpha=0.7,
            label=f"{d['formula']} (original)")
    ax.plot(x_arr, y_eml, color="#ef4444", linewidth=1.5, linestyle="--",
            alpha=0.9, label=f"{d['eml']} (EML)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Verification: original vs EML composition", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    max_err = float(np.max(np.abs(y_ref - y_eml)))
    if max_err < 1e-10:
        st.caption("Exact match (error < 1e-10)")
    else:
        st.caption(f"Maximum absolute error: {max_err:.2e}")


# ─────────────────────────────────────────────
# Illustration mode: Build Your Own
# ─────────────────────────────────────────────

def render_build_your_own():
    """Interactive composer: combine EML blocks to build new functions."""
    st.markdown("---")
    st.markdown("### Build Your Own")
    st.markdown(
        "Compose EML blocks step by step to build functions from scratch. "
        "Start with **x** and **1**, create new expressions, save them, "
        "and use them as building blocks for deeper compositions."
    )

    # Initialize workspace
    if "workspace" not in st.session_state:
        st.session_state.workspace = {
            "x": ("x",),
            "1": ("const", 1.0),
        }
    ws = st.session_state.workspace

    # Compose controls
    available = list(ws.keys())

    col_l, col_r = st.columns(2)
    with col_l:
        left_name = st.selectbox("Left input (a)", available, key="byo_left")
    with col_r:
        right_name = st.selectbox("Right input (b)", available, key="byo_right",
                                  index=min(1, len(available) - 1))

    left_expr = ws[left_name]
    right_expr = ws[right_name]
    result_expr = ("eml", left_expr, right_expr)

    eml_str = expr_to_eml(result_expr)
    math_str = expr_to_math(result_expr)

    st.markdown(f"**EML notation:** &ensp; `{eml_str}`")
    st.markdown(f"**Expanded math:** &ensp; `{math_str}`")

    # Domain controls
    col_lo, col_hi = st.columns(2)
    with col_lo:
        x_lo = st.number_input("x min", value=0.1, step=0.5, key="byo_xlo")
    with col_hi:
        x_hi = st.number_input("x max", value=5.0, step=0.5, key="byo_xhi")

    if x_lo >= x_hi:
        st.warning("x min must be less than x max")
        return

    # Tree + plot side by side
    col_plot, col_tree = st.columns([1.2, 1])

    with col_plot:
        x_arr = np.linspace(x_lo, x_hi, 300).astype(np.float64)
        y_arr = eval_expr(result_expr, x_arr)

        # Filter out extreme values for cleaner plot
        valid = np.isfinite(y_arr)
        if valid.sum() < 2:
            st.warning("Expression produces no finite values on this domain. Try adjusting x range.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot(x_arr[valid], y_arr[valid], color="#8b5cf6", linewidth=2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(eml_str, fontsize=10, fontfamily="monospace")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_tree:
        st.graphviz_chart(expr_to_dot_str(result_expr), use_container_width=True)

    # Save to workspace
    col_name, col_save, col_reset = st.columns([3, 1, 1])
    with col_name:
        default_name = eml_str if len(eml_str) < 40 else f"expr_{len(ws)}"
        save_name = st.text_input("Name this expression", value=default_name, key="byo_name")
    with col_save:
        st.markdown("&nbsp;")  # vertical alignment
        if st.button("Save", key="byo_save", type="primary"):
            if save_name and save_name not in ("x", "1"):
                st.session_state.workspace[save_name] = result_expr
                st.rerun()
    with col_reset:
        st.markdown("&nbsp;")
        if st.button("Reset", key="byo_reset"):
            st.session_state.workspace = {
                "x": ("x",),
                "1": ("const", 1.0),
            }
            st.rerun()

    # Show workspace
    custom_entries = {k: v for k, v in ws.items() if k not in ("x", "1")}
    if custom_entries:
        with st.expander(f"Workspace ({len(custom_entries)} saved expressions)", expanded=False):
            for name, expr in custom_entries.items():
                st.markdown(f"- **{name}** = `{expr_to_math(expr)}`")


# ═════════════════════════════════════════════
# Streamlit UI
# ═════════════════════════════════════════════

st.set_page_config(page_title="EML Explorer", layout="wide", page_icon="🌳")

# Header
st.markdown("""
# EML: One Operator to Rule Them All

The **EML operator** $\\text{eml}(a, b) = e^a - \\ln(b)$ is a universal primitive
for continuous mathematics — the analog of the NAND gate in digital logic. Every
elementary function can be built from EML alone.

*Based on Odrzywołek (2026) — "All elementary functions from a single binary operator"
([arXiv:2603.21852](https://arxiv.org/abs/2603.21852))*
""")

tab_illustration, tab_regression = st.tabs(["Illustration", "Regression"])

# ═════════════════════════════════════════════
# ILLUSTRATION MODE
# ═════════════════════════════════════════════

with tab_illustration:
    render_nand_analogy()
    render_explorer()
    render_build_your_own()

# ═════════════════════════════════════════════
# REGRESSION MODE (original functionality)
# ═════════════════════════════════════════════

with tab_regression:
    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        st.subheader("Data Source")
        source = st.radio("Choose input method", ["Preset function", "Paste CSV data"], index=0)

        if source == "Preset function":
            preset_name = st.selectbox("Function", list(PRESETS.keys()))
            noise_std   = st.slider("Noise σ", 0.0, 0.5, 0.0, 0.01,
                                    help="Add Gaussian noise to test robustness")
            fn, (lo, hi, n) = PRESETS[preset_name]
            x_data = np.linspace(lo, hi, n).astype(np.float32)
            y_data = fn(x_data).astype(np.float32)
            if noise_std > 0:
                y_data += np.random.normal(0, noise_std, size=y_data.shape).astype(np.float32)
        else:
            csv_text = st.text_area(
                "Paste x,y values (one pair per line)",
                "0.0, 1.0\n0.5, 1.6487\n1.0, 2.7183\n1.5, 4.4817\n2.0, 7.3891",
                height=180,
            )
            try:
                rows = [r.strip().split(",") for r in csv_text.strip().split("\n") if r.strip()]
                x_data = np.array([float(r[0]) for r in rows], dtype=np.float32)
                y_data = np.array([float(r[1]) for r in rows], dtype=np.float32)
            except Exception as e:
                st.error(f"CSV parse error: {e}")
                st.stop()

        st.divider()
        st.subheader("Tree & Training")
        depth      = st.slider("EML tree depth", 1, 4, 2,
                               help="Depth 1 → 2 leaves, depth 4 → 16 leaves")
        n_restarts = st.slider("Random restarts", 3, 30, 10,
                               help="More restarts → better chance of finding global minimum")
        n_epochs   = st.slider("Epochs per restart", 500, 8000, 3000, step=500)
        lr         = st.select_slider("Learning rate",
                                      options=[0.002, 0.005, 0.01, 0.02, 0.05], value=0.01)

        st.divider()
        run_btn = st.button("Run Symbolic Regression", type="primary", use_container_width=True)

        st.divider()
        st.caption(
            f"Tree has **{2**depth}** trainable leaves · "
            f"**{2**depth - 1}** eml nodes"
        )

    # ── Main area ────────────────────────────
    col_data, col_result = st.columns([1, 1])

    with col_data:
        st.subheader("Input Data")
        fig_data, ax_data = plt.subplots(figsize=(5, 3.2))
        ax_data.scatter(x_data, y_data, s=14, alpha=0.7, color="#3b82f6", edgecolors="none")
        ax_data.set_xlabel("x")
        ax_data.set_ylabel("y")
        ax_data.set_title("Data points", fontsize=11)
        ax_data.grid(True, alpha=0.3)
        fig_data.tight_layout()
        st.pyplot(fig_data)
        plt.close(fig_data)
        st.caption(f"{len(x_data)} points · x ∈ [{x_data.min():.2f}, {x_data.max():.2f}]")

    # ── Run regression ───────────────────────
    if run_btn:
        with col_result:
            st.subheader("Regression Results")
            progress = st.progress(0, text="Training EML trees ...")

            model, loss, hist = train_eml_tree(
                x_data, y_data, depth,
                n_restarts=n_restarts,
                n_epochs=n_epochs,
                lr=lr,
                progress_callback=lambda p: progress.progress(p, text=f"Training ... {p*100:.0f}%"),
            )
            progress.empty()

            # ── Metrics ──
            st.metric("Final MSE Loss", f"{loss:.2e}")

            # ── Fit plot ──
            x_plot = np.linspace(x_data.min(), x_data.max(), 300).astype(np.float32)
            with torch.no_grad():
                y_fit = model(torch.tensor(x_plot)).numpy()

            fig_fit, ax_fit = plt.subplots(figsize=(5, 3.2))
            ax_fit.scatter(x_data, y_data, s=14, alpha=0.5, color="#94a3b8", label="data", edgecolors="none")
            ax_fit.plot(x_plot, y_fit, color="#ef4444", linewidth=2, label="EML fit")
            ax_fit.set_xlabel("x")
            ax_fit.set_ylabel("y")
            ax_fit.set_title("Fit vs Data", fontsize=11)
            ax_fit.legend(fontsize=9)
            ax_fit.grid(True, alpha=0.3)
            fig_fit.tight_layout()
            st.pyplot(fig_fit)
            plt.close(fig_fit)

        # ── Full-width results below ─────────
        st.divider()

        col_eml, col_math = st.columns(2)
        with col_eml:
            st.subheader("EML Form")
            eml_str = model.eml_string()
            st.code(eml_str, language=None)

        with col_math:
            st.subheader("Expanded Math Form")
            math_str = model.math_string()
            st.code(math_str, language=None)

        # ── Loss curve ──
        st.subheader("Training Loss (best restart)")
        fig_loss, ax_loss = plt.subplots(figsize=(7, 2.5))
        ax_loss.semilogy(hist, color="#8b5cf6", linewidth=1)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE (log)")
        ax_loss.set_title("Convergence", fontsize=11)
        ax_loss.grid(True, alpha=0.3)
        fig_loss.tight_layout()
        st.pyplot(fig_loss)
        plt.close(fig_loss)

        # ── Tree visualization ──
        st.subheader("EML Tree Structure")
        try:
            dot_src = tree_to_dot(model)
            st.graphviz_chart(dot_src, use_container_width=True)
        except Exception:
            st.code(eml_str, language=None)
            st.caption("(Install graphviz for visual tree rendering)")

        # ── Leaf details ──
        with st.expander("Leaf parameter details"):
            leaves = []
            def _collect_leaves(node, path="root"):
                if isinstance(node, EMLLeaf):
                    g = torch.sigmoid(node.gate_logit * 4.0).item()
                    c = node.const_val.item()
                    snap_label, snap_val = snap_to_known(c)
                    leaves.append({
                        "path": path,
                        "gate": f"{g:.3f}",
                        "raw_const": f"{c:.4f}",
                        "snapped": snap_label,
                        "type": "x" if g > 0.75 else ("const" if g < 0.25 else "mixed"),
                    })
                elif isinstance(node, EMLNode):
                    _collect_leaves(node.left, path + ".L")
                    _collect_leaves(node.right, path + ".R")

            _collect_leaves(model.root)
            st.dataframe(leaves, use_container_width=True)

    elif not run_btn:
        with col_result:
            st.subheader("Results")
            st.info("Configure parameters in the sidebar, then press **Run Symbolic Regression**.")


# ── Footer ───────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; opacity:0.5; font-size:0.85rem">
EML Explorer · Grammar: S → 1 | x | eml(S, S) · eml(a, b) = exp(a) − ln(b)<br>
Paper: Odrzywołek, A. (2026). arXiv:2603.21852
</div>
""", unsafe_allow_html=True)
