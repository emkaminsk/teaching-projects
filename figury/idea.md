Build a Python Streamlit app that teaches a 12-year-old how to calculate the area of 2D geometric figures. The app must be interactive, visual, and — most importantly — teach intuition first, memorization never.

## Figures to Cover
1. Circle
2. Triangle
3. Square / Rectangle
4. Parallelogram
5. Rhombus ("romb")
6. Trapezoid

## Core Principle
For EACH figure, the app must explain WHY the formula works before asking the student to use it. Think "narrative first, calculation second." Use animations, visual proofs, and plain Polish language a 12-year-old actually understands.

## Structure Per Figure (repeatable module)

### Step 1 — "Dlaczego to działa?" (Why does this work?)
- A short, animated or step-by-step visual explanation
- For circle: show how cutting it into pizza slices and rearranging approximates a rectangle. Derive πr² from this intuition.
- For triangle: show two identical triangles forming a parallelogram. Derive ½bh from this.
- For trapezoid: show two identical trapezoids forming a parallelogram. Derive ½(a+b)h.
- Use ASCII art, matplotlib, or manim-style visual proofs rendered with Python

### Step 2 — "Sprawdź się" (Quiz yourself)
- The student inputs values for the figure
- App validates the answer with full step-by-step solution shown
- Wrong answer → gentle hint, not just "wrong"

### Step 3 — "Wyzwów siebie" (Challenge mode)
- Random values, time pressure (optional), score tracking
- Leaderboard stored in st.session_state (no backend needed)

## UI Requirements
- Left sidebar: figure selector with icons
- Main area: explanation + interactive widget
- st.session_state to track progress across figures
- Polish language throughout ("ty" not "Pan/Pani")
- Colorful but not overwhelming — muted background, bright accent colors for shapes
- Emojis allowed but sparingly
- Responsive: works on tablet and desktop

## Technical Stack
- Python 3.10+
- Streamlit
- matplotlib (for shape drawing and animations)
- st.session_state for progress persistence
- No external API calls — fully offline

## Visual Style
- Background: soft light gray or cream (#F5F5F5 or similar)
- Shape fill colors: distinct pastels per figure (blue for circle, green for triangle, orange for trapezoid, etc.)
- Font: default Streamlit but larger (18px+ for body text)
- Each shape drawn with matplotlib.plt or st.pyplot()

## Interaction Pattern
1. Student selects figure from sidebar
2. App shows: "Czy wiesz dlaczego pole koła to πr²?" with animated visual
3. Student clicks "Pokaż dowód" (Show proof) to reveal the intuition
4. Student enters values in a form
5. App shows: "Twój wynik: X. Poprawna odpowiedź: Y" with full calculation shown

## Bonus Features (if straightforward)
- Dark mode toggle
- "Print worksheet" button that generates random problems as st.download_button
- Sound effects (optional, toggleable)

## What to Avoid
- Do NOT just show the formula and ask for numbers
- Do NOT use technical terms without explaining them in plain Polish
- Do NOT be boring or textbook-like
- Do NOT require login or external services

Write the complete, working Streamlit app. Include all modules in one file or a small number of clean files. The app should be runnable with `streamlit run app.py` after installing requirements.txt dependencies: streamlit, matplotlib.
