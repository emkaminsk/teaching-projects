Read this paper https://arxiv.org/abs/2603.21852

The author, Andrzej Odrzywołek, found that a single binary operator:
eml(x, y) = exp(x) − ln(y)
together with just the constant 1, can generate every standard elementary function: sin, cos, sqrt, log, all arithmetic operations, and even constants like e, π, and i.
This is the continuous-mathematics analog of the NAND gate in digital logic. Just as every Boolean circuit can be built from NAND alone, every scientific-calculator function can be built from EML alone. Nobody expected such a primitive to exist for continuous math.
Simple examples: exp(x) = eml(x, 1) (since ln(1) = 0, the log term vanishes), e = eml(1, 1) (which is just exp(1) − 0), and ln(x) = eml(1, eml(eml(1, x), 1)) — a nested composition three levels deep.

Symbolic regression is the problem of discovering the mathematical formula that generated a dataset. You have data points (x, y) and you want to find f such that y = f(x) — not just a numerical fit, but the exact closed-form expression, like y = sin(x²) or y = ln(x) + π.
Traditionally this is brutally hard because you must search over a combinatorial space of different function types, different nestings, different constants. Each candidate formula has a different shape — one uses sin, another uses log, another uses both.

The idea is to generate a streamlit app that could simulate the regression part of this finding. User provides 2 dimensional vector data ( x, y) and sets tree depth. The app implemented the search part and shows the end tree, the eml form and the approximated function.