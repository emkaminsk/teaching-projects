# 🌳 EML Symbolic Regression

Discover closed-form mathematical formulas from numerical data using the **EML operator**:

```
eml(x, y) = exp(x) − ln(y)
```

Based on Odrzywołek (2026), *"All elementary functions from a single binary operator"* — [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)

## How it works

1. You provide (x, y) data — either from preset functions or pasted CSV
2. The app builds a **full binary tree** of depth d, where every internal node is an `eml` operation and every leaf is a trainable parameter (learned to be either `x` or a constant)
3. **Adam optimizer** trains all leaf parameters via gradient descent on MSE loss
4. After convergence, leaf values are **snapped** to recognized constants (0, 1, e, π, etc.)
5. The recovered EML expression, expanded math form, and tree structure are displayed

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
EML Tree (depth=2):

        eml
       /    \
     eml    eml
    /   \  /   \
   x    1  x    1

Each leaf: value = σ(gate) · x + (1 − σ(gate)) · c
where gate and c are trainable parameters.
```

## Tips

- **Depth 1** recovers exp(x), e−ln(x), and similar single-application functions
- **Depth 2** can recover ln(x), simple compositions
- **Depth 3–4** needed for more complex functions but training gets harder
- Increase **random restarts** (15–30) for deeper trees
- Add small noise (σ ≈ 0.01) to test robustness of recovery
- Functions requiring complex-domain intermediate steps (sin, cos) are not recoverable in this real-valued prototype

## Limitations

- **Real-valued only**: The paper's full construction works over ℂ (complex numbers). This prototype operates in ℝ, so trigonometric functions that require complex intermediaries (via Euler's formula) cannot be recovered. A complex-valued extension is a natural next step.
- **Depth ceiling**: At depth 4 (16 leaves), the optimization landscape becomes highly non-convex. Multiple restarts help but don't guarantee global optima.
- **Snapping heuristic**: The constant-recognition step uses a tolerance-based nearest-match to a known set. Exotic constants may not be recognized.

## License

MIT — The underlying mathematical discovery is from the referenced paper (CC BY 4.0).
