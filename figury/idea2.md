Build a Python Streamlit app that teaches a 12-year-old to calculate the area of 2D geometric figures. The core interaction model must mirror a reference app: **sliders on the left panel update a live visualization on the right — instantly, with no submit button**. Think of it as a "math microscope" — the student moves a slider and immediately sees the shape grow or shrink, with the result updating in real time.

## Core Interaction Pattern (COPY THIS EXACTLY)

LEFT COLUMN (1/3 width):

• st.slider() for each dimension (e.g., radius, base, height)
• Live st.metric() cards showing current input values
• Shape selector or preset buttons

RIGHT COLUMN (2/3 width):

• st.pyplot() matplotlib figure — updates live as sliders move
• st.info (http://st.info/)() caption below the chart with plain-language intuition


## Figures and Their Visual Proofs

### 1. Circle — "Pizza slice proof"
- Slider: radius r (1–20, step 0.5)
- Live visualization: circle with radius line labeled, area shown as filled region
- Formula: A = πr²
- Intuition (show as st.info): "Pomaluj koło na kolorowo, rozcinaj na coraz cieńsze plasterki i ułóż je w prawie-prostokąt. Wysokość = r, szerokość = πr, więc pole = r × πr = πr²"
- Show both the circle AND the rearranged "pizza → rectangle" approximation as an animated or side-by-side view

### 2. Triangle — "Two triangles make a parallelogram"
- Sliders: base b (1–20), height h (1–20)
- Live visualization: triangle with labeled base and height; button/animation showing second identical triangle rotating to form a parallelogram
- Formula: A = ½bh
- Intuition: "Dwa takie same trójkąty zawsze składają się w równoległobok. Pole równoległoboku to b×h, więc jeden trójkąt to połowa."

### 3. Square — "Count the squares"
- Sliders: side a (1–20)
- Live visualization: grid of unit squares filling the square, counter showing total
- Formula: A = a²
- Intuition: "Ile małych kwadratów mieści się w tym dużym? Policz wzdłuż boku — a małych. Policz boki — a rzędów. Razem: a × a = a²"

### 4. Rectangle — "Area is just counting"
- Sliders: width a (1–20), height b (1–20)
- Live visualization: grid filling the rectangle, highlight a row and a column to show a×b
- Formula: A = a×b
- Intuition: "Narysuj a kropek w górę i b kropek w prawo. Ile jest wszystkich kropek? a rzędów po b = a×b"

### 5. Rhombus ("Romb") — "Two diagonals make 4 triangles"
- Sliders: diagonal d1 (1–20), diagonal d2 (1–20)
- Live visualization: rhombus with both diagonals drawn, crossing at right angle, 4 right triangles highlighted in different colors
- Formula: A = ½×d1×d2
- Intuition: "Przekątne dzielą rąb na 4 jednakowe trójkąty prostokątne. Połącz je w prostokąt o bokach ½d1 i ½d2. Prostokąt ma pole ½d1 × ½d2, czyli ½d1d2."

### 6. Trapezoid ("Trapez") — "Two trapezoids make a parallelogram"
- Sliders: base a (1–20), base b (1–20), height h (1–15)
- Live visualization: one trapezoid, button to show second identical one flipped and joined to make a parallelogram
- Formula: A = ½(a+b)h
[29.03.2026 21:19] Mireq: - Intuition: "Dwa takie same trapezy składają się w równoległobok o podstawie a+b i wysokości h. Pole równoległoboku to (a+b)×h, więc jeden trapez to połowa."

## UI Structure (PER FIGURE)
For each of the 6 figures, replicate this layout:
[29.03.2026 21:19] Mireq: st.sidebar: st.radio() or st.selectbox() to pick the figure

st.title("🔵 Pole koła" / "🔺 Pole trójkąta" etc.)

LEFT col1, RIGHT col2 = st.columns([1, 2])

col1:

• st.subheader("Zmień wymiary")
• st.slider() for each dimension, with st.metric() showing current value
• Formula display: st.latex() showing the formula with current values substituted

col2:

• st.pyplot() — matplotlib figure, updates live
• st.info (http://st.info/)("Intuition in plain Polish")
• Optional: st.caption() with "fun fact"

REPEAT for each figure — keep the same layout pattern so navigation feels consistent


## Visualization Style
- Background: white or very light gray
- Shape fill: semi-transparent pastel (blue, green, orange, yellow, purple, red — one per shape)
- Grid lines: light gray, alpha 0.3
- Labels: font size 14+, clearly readable
- For proof animations: use matplotlib animation or show two side-by-side states (before/after)
- All text in Polish, casual register ("tu", "twoje pole", not "należy obliczyć")

## Technical Requirements
- Python 3.10+, Streamlit, matplotlib, numpy
- `st.session_state` to persist the last-selected figure between navigation
- No submit button — all updates are `@st.fragment` or simply live as sliders move
- Responsive: works on tablet (1024px+) and desktop
- No external API calls — fully offline
- Page icon: 📐 or 🧮

## What Makes This App Exceptional
- The visualization is not an illustration — it IS the calculation. Moving the radius slider from 3 to 6 doesn't just change a number; the circle on screen literally doubles in size and the area number doubles in real time.
- The "proof intuition" is always visible below the chart, not hidden behind a button. The student sees why the formula works WHILE they interact with it.
- The layout never changes format between figures — only the shape and numbers change. This predictability reduces cognitive load.

## File Structure
Single `app.py` file (no need to split into modules — keep it under 500 lines for Opus).

## Bonus (only if it doesn't complicate the code)
- Dark mode toggle
- A "Losuj zadanie" (random problem) button: picks random values, asks student to compute, then reveals answer with step-by-step
- Print-friendly worksheet export via `st.download_button`

## Anti-patterns to Avoid
- Do NOT show the formula in a static image that doesn't update
- Do NOT require the student to press "Oblicz" (calculate) — numbers must update as they drag
- Do NOT use "Pan/Pani" — use "ty" throughout
- Do NOT make the student scroll to see the chart
- Do NOT skip the intuition explanation — it's the whole point

Write the complete, runnable app. The student should be able to run it with: `streamlit run app.py`
