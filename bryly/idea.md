Build a Python Streamlit app that teaches an 18-year-old to calculate the volume, surface area, and cross-section area of 3D geometric solids, with usage of formulas (e.g., Pythagorean formula). The core interaction model must mirror a reference app: **sliders on the left panel update a live visualization on the right — instantly, with no submit button**. Think of it as a "math microscope" — the student moves a slider and immediately sees the solid grow or shrink, with volume and surface area updating in real time. A cross-section slider lets the student "slice" the solid at any height and see the resulting 2D shape.

## Core Interaction Pattern (COPY THIS EXACTLY)

LEFT COLUMN (1/3 width):

• st.slider() for each dimension (e.g., radius, height, edge length)
• Live st.metric() cards showing current input values
• Cross-section height slider (where applicable)
• Solid selector or preset buttons

RIGHT COLUMN (2/3 width):

• st.pyplot() matplotlib 3D figure — updates live as sliders move
• Second smaller st.pyplot() — 2D cross-section at chosen height
• st.info() caption below the chart with plain-language intuition


## Solids and Their Visual Proofs

### 1. Sphere ("Kula") — "Archimedes' cylinder trick"
- Slider: radius r (1–15, step 0.5)
- Live visualization: 3D wireframe/surface sphere with radius line labeled
- Cross-section slider: height of slice (-r to r) — shows a circle whose radius changes as the slice moves up/down; cross-section radius = sqrt(r² - h²) (Pythagorean theorem in action)
- Volume formula: V = 4/3 · π · r³
- Surface area formula: S = 4 · π · r²
- Intuition (show as st.info): "Archimedes odkrył, że kula idealnie mieści się w walcu. Objętość kuli to dokładnie 2/3 objętości tego walca. Przekrój kuli na dowolnej wysokości to zawsze koło — ale jego promień się zmienia! Użyj twierdzenia Pitagorasa: promień przekroju = √(r² − h²)."
- Show the enclosing cylinder as a ghost wireframe overlay to illustrate the 2/3 relationship

### 2. Cone ("Stożek") — "One third of a cylinder"
- Sliders: radius r (1–15, step 0.5), height h (1–20)
- Live visualization: 3D cone with transparent surface, radius, height, and slant height l labeled; l = √(r² + h²) shown as Pythagorean relationship
- Cross-section slider: height of slice (0 to h) — shows a circle whose radius shrinks linearly from r at the base to 0 at the apex; cross-section radius at height t = r · (1 − t/h)
- Volume formula: V = 1/3 · π · r² · h
- Surface area formula: S = π · r² + π · r · l, where l = √(r² + h²)
- Intuition: "Stożek to dokładnie 1/3 walca o tym samym promieniu i wysokości. Przekrój na dowolnej wysokości to kółko — im wyżej kroisz, tym mniejsze. Na czubku promień = 0. Tworzącą (l) obliczasz z Pitagorasa: l = √(r² + h²)."
- Show ghost cylinder overlay to illustrate the 1/3 relationship

### 3. Rectangular Prism ("Prostopadłościan") — "A box of blocks"
- Sliders: length a (1–15), width b (1–15), height c (1–15)
- Live visualization: 3D wireframe cuboid with semi-transparent faces, edges labeled
- Cross-section slider: height of slice (0 to c) — shows rectangle a × b (constant); option to switch to vertical cross-section showing a × c or b × c
- Volume formula: V = a · b · c
- Surface area formula: S = 2(ab + bc + ac)
- Space diagonal: d = √(a² + b² + c²) — shown as a labeled line through the solid (double application of Pythagorean theorem)
- Intuition: "To pudełko na klocki. Na dnie mieści się a×b klocków. Takich warstw jest c. Objętość = a·b·c. Powierzchnia to 3 pary ścian: góra+dół, przód+tył, lewo+prawo. Przekątna przestrzenna? Pitagoras dwa razy: najpierw przekątna dna √(a²+b²), potem z nią i wysokością √(a²+b²+c²)."

### 4. Cube ("Sześcian") — "Stack the layers"
- Slider: edge a (1–15)
- Live visualization: 3D wireframe cube with semi-transparent faces; for small a (≤5), show individual unit cubes inside
- Cross-section slider: height of slice (0 to a) — always shows a square a × a
- Volume formula: V = a³
- Surface area formula: S = 6 · a²
- Space diagonal: d = a · √3 — labeled inside the cube
- Intuition: "Ile małych kostek zmieści się w środku? Na dnie masz a×a = a² kostek. Takich warstw jest a. Razem: a·a·a = a³. Ścian jest 6, każda to kwadrat a×a. Przekątna przestrzenna = a√3 — Pitagoras użyty dwa razy."

### 5. Cylinder ("Walec") — "Stack of circles"
- Sliders: radius r (1–15, step 0.5), height h (1–20)
- Live visualization: 3D cylinder with transparent surface, radius and height labeled
- Cross-section slider: height of slice (0 to h) — always shows a circle of radius r (horizontal); option to show vertical cross-section (rectangle 2r × h)
- Volume formula: V = π · r² · h
- Surface area formula: S = 2 · π · r² + 2 · π · r · h
- Intuition: "Walec to stos kółek. Każde kółko ma pole π·r². Takich kółek (warstw) jest h. Objętość = π·r²·h. Powierzchnia boczna? Rozwiń ją w myślach — to prostokąt o bokach 2πr (obwód koła) i h!"
- Show "unrolled" lateral surface as a rectangle in side-by-side view

### 6. Pyramid ("Ostrosłup") — "One third of a prism"
- Sliders: base edge a (1–15), height h (1–20)
- Live visualization: 3D square-based pyramid with transparent faces, base edge and height labeled; slant height of face = √((a/2)² + h²)
- Cross-section slider: height of slice (0 to h) — shows a square that shrinks from a×a at base to 0 at apex; cross-section edge at height t = a · (1 − t/h)
- Volume formula: V = 1/3 · a² · h
- Surface area formula: S = a² + 2 · a · √((a/2)² + h²)
- Intuition: "Ostrosłup to dokładnie 1/3 graniastosłupa o tej samej podstawie i wysokości. Przekrój na dowolnej wysokości to kwadrat — im wyżej, tym mniejszy. Wysokość ściany bocznej (apotema) obliczysz z Pitagorasa: √((a/2)² + h²)."
- Show ghost rectangular prism overlay to illustrate the 1/3 relationship


## UI Structure (PER SOLID)
For each of the 6 solids, replicate this layout:

st.sidebar: st.radio() or st.selectbox() to pick the solid

st.title("🔵 Kula" / "🟩 Sześcian" / "🟧 Walec" etc.)

LEFT col1, RIGHT col2 = st.columns([1, 2])

col1:

• st.subheader("Zmień wymiary")
• st.slider() for each dimension, with st.metric() showing current value
• st.slider() for cross-section height
• Formula display: st.latex() showing volume AND surface area formulas with current values substituted
• st.metric() for computed Volume and Surface Area

col2:

• st.pyplot() — 3D matplotlib figure (Axes3D), updates live
• Second smaller st.pyplot() — 2D cross-section at chosen height
• st.info("Intuition in plain Polish")
• Optional: st.caption() with "fun fact"

REPEAT for each solid — keep the same layout pattern so navigation feels consistent


## Visualization Style
- Background: white or very light gray
- Solid fill: semi-transparent pastel (blue for sphere, green for cube, orange for cuboid, yellow for cylinder, purple for cone, red for pyramid)
- Use matplotlib 3D projections (Axes3D) for the main view
- Cross-section plot: 2D matplotlib with the slice shape filled in matching color
- Grid lines: light gray, alpha 0.3
- Labels: font size 14+, clearly readable
- For proof overlays (ghost cylinder around cone/sphere, prism around pyramid): wireframe with alpha 0.2
- All text in Polish, casual register ("tu", "twoja objętość", not "należy obliczyć")

## Technical Requirements
- Python 3.10+, Streamlit, matplotlib, numpy
- `st.session_state` to persist the last-selected solid between navigation
- No submit button — all updates are `@st.fragment` or simply live as sliders move
- Responsive: works on tablet (1024px+) and desktop
- No external API calls — fully offline
- Page icon: 📐 or 🧊

## What Makes This App Exceptional
- The 3D visualization is not an illustration — it IS the calculation. Moving the radius slider from 3 to 6 doesn't just change a number; the sphere on screen literally grows and the volume/surface area numbers update in real time.
- The cross-section slider is the killer feature — dragging it up and down lets the student SEE how the slice shape changes, building geometric intuition that static textbook diagrams cannot.
- The Pythagorean theorem appears naturally: in slant heights, space diagonals, and cross-section radii — the student sees WHY it matters in 3D geometry.
- Ghost overlays (cylinder around cone/sphere, prism around pyramid) make the 1/3 and 2/3 relationships tangible.
- The "proof intuition" is always visible below the chart, not hidden behind a button. The student sees why the formula works WHILE they interact with it.
- The layout never changes format between solids — only the shape and numbers change. This predictability reduces cognitive load.

## File Structure
Single `app.py` file (no need to split into modules — keep it under 600 lines for Opus).

## Bonus (only if it doesn't complicate the code)
- Dark mode toggle
- A "Losuj zadanie" (random problem) button: picks random values, asks student to compute volume or surface area, then reveals answer with step-by-step
- Print-friendly worksheet export via `st.download_button`
- Toggle to show/hide the ghost overlay (enclosing cylinder/prism)

## Anti-patterns to Avoid
- Do NOT show the formula in a static image that doesn't update
- Do NOT require the student to press "Oblicz" (calculate) — numbers must update as they drag
- Do NOT use "Pan/Pani" — use "ty" throughout
- Do NOT make the student scroll to see the 3D view
- Do NOT skip the intuition explanation — it's the whole point
- Do NOT use plotly or other heavy 3D libraries — matplotlib Axes3D is sufficient and keeps dependencies minimal

Write the complete, runnable app. The student should be able to run it with: `streamlit run app.py`
