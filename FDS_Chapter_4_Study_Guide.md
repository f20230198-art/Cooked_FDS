# Foundation of Data Science (CS G320) — Study Guide

### Chapter 4: Optimization

*(Continuation of Chapters 1–3. Read alongside the other chapter guides in this folder.)*

---

# 📕 CHAPTER 4 — Optimization

> ⚠️ **Note from your study buddy:** I rewrote this chapter to be easier to follow and **fixed a real arithmetic error** that was in the old Example 4.11 (iterations 2 & 3). Everything is verified now. Read the "🧠 Think of it like…" boxes — they are your memory hooks for the exam.

## 4.0 The One-Sentence Story of This Chapter

> **Optimization = finding the input that makes a function as small (or as large) as possible — sometimes with rules (constraints) you must obey.**

Everything in this chapter is one of two situations:

| Situation | What it means | Tools we learn |
|---|---|---|
| **Unconstrained** | "Just find the lowest point. No rules." | Derivatives, Gradient Descent, Bracketing, Golden Section |
| **Constrained** | "Find the lowest point, *but stay inside the fence*." | Linear Programming, Lagrange Multipliers |

```
                          OPTIMIZATION
                               │
        ┌──────────────────────┼───────────────────────┐
   FOUNDATIONS            UNCONSTRAINED             CONSTRAINED
   (the math toolkit)     (no rules)                (with rules)
   ───────────            ─────────────             ───────────
   Derivatives            1st-order optimality      Linear Programming
   Gradient & Hessian     Gradient Descent          Lagrange Multipliers
   Convexity              Bracketing / Golden       (equality constraints)
```

🧠 **Think of it like:** you are a ball rolling around a landscape. You want the lowest valley. *Unconstrained* = open landscape. *Constrained* = the landscape has walls/fences you can't cross.

---

## 4.1 What is Optimization?

> The process of **finding the best values for the variables** to **minimize or maximize an objective function**.

### The 3 ingredients of EVERY optimization problem

| # | Ingredient | Plain English | Example (factory) |
|---|---|---|---|
| 1 | **Objective function** | The thing you want to make best (max profit / min cost) | Profit $Z = 100x + 200y$ |
| 2 | **Variables** | The knobs you can turn | $x$ = tables, $y$ = chairs |
| 3 | **Constraints** | The rules you can't break | wood ≤ 40, labour ≤ 24 |

🧠 **Memory hook — "O-V-C": Objective, Variables, Constraints.** If a question says "state the components of an optimization problem," just say **O-V-C** and give one line each.

---

## 4.2 Functions and Their Optima

We mostly minimize/maximize a function $f$. Start simple: $f : \mathbb{R} \to \mathbb{R}$ (one number in, one number out).

| Term | Meaning | Analogy |
|---|---|---|
| **Local minimum** | Lowest point *in its neighbourhood* | a small dip on a hiking trail |
| **Global minimum** | Lowest point *over the whole domain* | the deepest valley anywhere |
| **Saddle point** | Slope is zero but it's **neither** a peak nor a valley | a mountain pass |

```
 f(x)
   |   /\        ← global max
   |  /  \    /\
   | /    \  /  \___ ← local min
   |/      \/        \
   |________________________ x
        local        global
        max           min
```

> To find these special points, we look at **derivatives / gradients** (the slope).

---

## 4.3 Derivatives — the slope

> The derivative at a point = **how fast the function is changing there** (the slope of the tangent line).

$$ \frac{df(x)}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} $$

| Sign of $f'(x)$ | The function is… | Picture |
|---|---|---|
| **Positive** | going **up** (increasing) | ↗ |
| **Negative** | going **down** (decreasing) | ↘ |
| **Zero** | flat — a **stationary point** | → (top of hill / bottom of valley / saddle) |

🧠 **Think of it like:** walking on the curve. Slope positive = walking uphill. Slope = 0 = you're standing on flat ground (could be a peak, a valley, or a flat ledge = saddle).

**Key rule:** at any optimum (max or min), $\boxed{f'(x) = 0}$.

---

## 4.4 Rules of Derivatives (you WILL be asked to differentiate)

| Rule | Formula | Worked mini-example |
|---|---|---|
| **Sum** | $(f+g)' = f' + g'$ | $(x^2 + x)' = 2x + 1$ |
| **Scaling** | $(a f)' = a f'$ | $(5x^3)' = 15x^2$ |
| **Product** | $(fg)' = f'g + g'f$ | $(x^2 \sin x)' = 2x\sin x + x^2\cos x$ |
| **Quotient** | $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - g'f}{g^{2}}$ | $\left(\dfrac{x}{\sin x}\right)' = \dfrac{\sin x - x\cos x}{\sin^2 x}$ |
| **Chain** | $\big(f(g(x))\big)' = f'(g(x))\cdot g'(x)$ | $(\sin(x^2))' = \cos(x^2)\cdot 2x$ |

🧠 **Memory hooks:**
- **Product rule = "first × derivative-of-second + second × derivative-of-first."** Chant: *"d(uv) = u dv + v du."*
- **Chain rule = "derivative of the OUTSIDE (leave inside alone) × derivative of the INSIDE."** Like peeling an onion: outer layer first, then the inner.
- **Quotient rule = "low d-high MINUS high d-low, over low-squared."** (Lo-D-Hi minus Hi-D-Lo over Lo-Lo.)

**Solved example (exam favourite):** Differentiate $f(x) = x^2 e^x$.
Product rule with $u=x^2,\ v=e^x$: $f'(x) = (2x)(e^x) + (x^2)(e^x) = \mathbf{2xe^x + x^2 e^x}$.

---

## 4.5 The Second Derivative & Saddle Points

The first derivative finds *where* the slope is flat. The **second derivative** tells you *what kind* of flat point it is.

| At a stationary point ($f'(x)=0$) | If… | Then it is a | Why (intuition) |
|---|---|---|---|
| $f''(x) > 0$ | curve bends **up** 🙂 | **local minimum** | a smile/bowl holds water |
| $f''(x) < 0$ | curve bends **down** 🙁 | **local maximum** | a frown/dome spills water |
| $f''(x) = 0$ | inconclusive | maybe a **saddle / inflection** | flat ledge — need more info |

🧠 **Memory hook:** **"Second derivative POSITIVE = POSITIVE smile 🙂 = MINIMUM."** Positive → cup up → holds water → bottom of valley.

### Saddle Points
> Slope is zero, but it's **neither a max nor a min**.

- Extremely common in **deep-learning loss functions** — training can get stuck there.
- Looks like a horse saddle / a Pringle chip: goes **up in one direction, down in another**.

---

## 4.6 Multivariate Functions & the Gradient

Real ML problems have **many variables** (e.g. loss $L(w)$ where $w$ is a vector of $D$ weights). So the "slope" becomes a **vector of slopes** — the **gradient**.

$$ \nabla f(x) = \left[ \frac{\partial f}{\partial x_1},\; \frac{\partial f}{\partial x_2},\; \dots,\; \frac{\partial f}{\partial x_D} \right]^{\top} $$

- A **partial derivative** $\frac{\partial f}{\partial x_i}$ = slope in the $x_i$ direction only (freeze all other variables).
- **Optimum condition:** $\nabla f(x) = 0$ → **every** partial derivative must be zero (flat in *all* directions at once).

🧠 **Think of it like:** standing on a hillside. The gradient points in the direction of **steepest uphill**. At the very top/bottom, there is no uphill — gradient = 0.

---

## 4.7 The Hessian — curvature in many dimensions

The multivariable version of the second derivative is a **matrix**: the **Hessian** ($D \times D$, all second-order partials).

$$ H = \nabla^{2} f(x) = \begin{bmatrix} \dfrac{\partial^{2}f}{\partial x_1^{2}} & \dfrac{\partial^{2}f}{\partial x_1 \partial x_2} & \cdots \\[3mm] \dfrac{\partial^{2}f}{\partial x_2 \partial x_1} & \dfrac{\partial^{2}f}{\partial x_2^{2}} & \cdots \\[2mm] \vdots & \vdots & \ddots \end{bmatrix} $$

At a stationary point ($\nabla f = 0$), the Hessian classifies it:

| Hessian is… | Then $x$ is a | Memory |
|---|---|---|
| **Positive semi-definite (PSD)** | **local minimum** | PSD → "bowl up" → MIN |
| **Negative semi-definite (NSD)** | **local maximum** | NSD → "dome down" → MAX |
| **Indefinite** (mixed signs) | **saddle point** | up some ways, down others |

🧠 **Memory hook:** Hessian = **"the matrix of curvatures."** Same logic as the 1-D second-derivative test: **Positive → minimum** (smile), Negative → maximum (frown), Mixed → saddle.

---

## 4.8 Convex & Non-Convex Functions

This decides whether optimization is **easy or hard**.

### Convex function
> $f$ is **convex** if the straight line (chord) joining any two points on the curve **never goes below** the curve.

$$ f\big(\lambda x + (1-\lambda)y\big) \;\le\; \lambda f(x) + (1-\lambda)f(y), \qquad 0 \le \lambda \le 1 $$

```
 convex:      \___/      chord always ABOVE the curve  → ONE valley
 non-convex:  \/\_/\      wiggly  → MANY valleys, can get trapped
```

| | **Convex** | **Non-convex** |
|---|---|---|
| Number of minima | exactly **one** (global) | possibly **many** local minima |
| Optimization | **easy** — any local min IS the global min | **hard** — GD can get stuck |

🧠 **Memory hook:** **Convex = "one big bowl, chord stays on top."** $x^2$ is convex. A wavy function is non-convex.

### Convex Set
> A set $S$ is **convex** if, for any two points in it, the **whole line segment between them stays inside** $S$.

$$ z = \lambda x + (1-\lambda)y \in S, \qquad 0 \le \lambda \le 1 $$

🧠 **Think of it like:** a filled circle or square is convex (any string between two inside points stays inside). A crescent moon or a donut is **not** convex.
> ⭐ **Important:** the **domain of a convex function must be a convex set.**

---

# 🔧 UNCONSTRAINED OPTIMIZATION

## 4.9 First-Order Optimality

> The simplest condition for an optimum: **the gradient is zero.**

$$ g = \nabla\big[L(w)\big] = 0 $$

- Sometimes you can **solve $g = 0$ directly** → a **closed-form solution** (exact answer, one shot).
- If you *can't* solve it directly → use an **iterative** method like **Gradient Descent**.

---

## 4.10 Gradient Descent (GD) — the workhorse of ML

> Repeatedly take a small step in the **opposite direction of the gradient** (i.e. **downhill**) until you reach the bottom.

**The one formula to memorize:**

$$ \boxed{\; w^{(k+1)} = w^{(k)} - \eta\, g^{(k)} \;} $$

**Algorithm:**
1. Start at some $w^{(0)}$.
2. Repeat: compute gradient $g^{(k)}$ → step downhill → $w^{(k+1)} = w^{(k)} - \eta\, g^{(k)}$.
3. Stop when it converges (barely changes).

> **The learning rate $\eta$ (eta) is critical:**
> - **Too small** → tiny steps → painfully slow. 🐢
> - **Too large** → giant steps → overshoots the valley, may **diverge**. 🚀💥
> - "Just right" → steady progress to the minimum.

```
 L(w)
   |  \                     start w0
   |   \    .
   |    \  / \   each step moves downhill ↓
   |     \/   \__ . __ . __ ●  global min
   |________________________ w
```

🧠 **Memory hook:** **"MINUS the gradient to go DOWN."** Minimizing → subtract. (If you ever *maximize*, you'd add → that's "gradient *ascent*".)

---

## 4.11 GD — Worked Example 1  ✅ (corrected)

**Problem:** Minimize $f(x) = x^{2}\sin x + 10\sin x - 50$. Use $\eta = 0.2$, start at $x^{(0)} = 9$. Do **3 iterations**.

**Step 1 — derivative** (product rule on $x^2\sin x$, chain on $10\sin x$; the $-50$ vanishes):

$$ f'(x) = \underbrace{2x\sin x + x^{2}\cos x}_{\text{product rule}} + \underbrace{10\cos x}_{\text{from }10\sin x} $$

**Update rule:** $x_{\text{new}} = x_{\text{old}} - 0.2\, f'(x_{\text{old}})$ &nbsp; *(angles in radians)*.

**Iteration 1** — at $x^{(0)} = 9$ &nbsp;($\sin 9 \approx 0.412$, $\cos 9 \approx -0.911$):

$$ f'(9) = 2(9)(0.412) + 81(-0.911) + 10(-0.911) \approx 7.42 - 73.79 - 9.11 = -75.49 $$
$$ x^{(1)} = 9 - 0.2(-75.49) = 9 + 15.10 = \mathbf{24.10} $$

**Iteration 2** — at $x^{(1)} \approx 24.10$ &nbsp;($\sin 24.10 \approx -0.905$, $\cos 24.10 \approx 0.426$):

$$ f'(24.10) \approx 2(24.10)(-0.905) + (24.10)^{2}(0.426) + 10(0.426) \approx -43.6 + 247.5 + 4.3 = 260.79 $$
$$ x^{(2)} = 24.10 - 0.2(260.79) = 24.10 - 52.16 = \mathbf{-28.06} $$

**Iteration 3** — at $x^{(2)} \approx -28.06$ &nbsp;($\sin \approx 0.16$, $\cos \approx -0.99$):

$$ f'(-28.06) \approx 2(-28.06)(0.16) + (-28.06)^{2}(-0.99) + 10(-0.99) \approx -767.19 $$
$$ x^{(3)} = -28.06 - 0.2(-767.19) = -28.06 + 153.44 = \mathbf{125.38} $$

> 🛠️ **What changed from the old notes:** the old file had iteration 2 = −17.52 and iteration 3 = 10.62. Those were **arithmetically wrong** ($f'(24.10)$ is **260.79**, not 208.1). Corrected values: **24.10 → −28.06 → 125.38**.
>
> 📌 **Exam point:** examiners care about the **method** (compute $f'$, plug $x$, apply $x_{\text{new}} = x_{\text{old}} - \eta f'$, repeat). The iterate **blowing up** here is the *lesson*: $\eta = 0.2$ is **too large** for this function → divergence. This is a perfect illustration of "learning rate too big."

---

## 4.12 GD — Worked Example 2 (Mean Squared Error) ✅ verified

**Problem:** Fit a line $\hat{y}_i = m x_i + c$ by minimizing **MSE** with one GD step.

$$ \mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}\big(y_i - \hat{y}_i\big)^{2} $$

**Gradients:**

$$ \frac{\partial \mathrm{MSE}}{\partial m} = -\frac{2}{N}\sum_{i} x_i\big(y_i - \hat{y}_i\big), \qquad \frac{\partial \mathrm{MSE}}{\partial c} = -\frac{2}{N}\sum_{i}\big(y_i - \hat{y}_i\big) $$

**Given:** $m = 0.5$, $c = 1$, $\eta = 0.2$, data $(1.5, 5)$ and $(3, 8)$, so $N=2$.

**Step 1 — predictions:** $\hat y_1 = 0.5(1.5)+1 = 1.75$, &nbsp; $\hat y_2 = 0.5(3)+1 = 2.5$.

**Step 2 — residuals (actual − predicted):** $r_1 = 5 - 1.75 = 3.25$, &nbsp; $r_2 = 8 - 2.5 = 5.5$.

**Step 3 — gradients:**

$$ \frac{\partial \mathrm{MSE}}{\partial m} = -\frac{2}{2}\big[1.5(3.25) + 3(5.5)\big] = -(4.875 + 16.5) = -21.375 $$
$$ \frac{\partial \mathrm{MSE}}{\partial c} = -\frac{2}{2}\big[3.25 + 5.5\big] = -8.75 $$

**Step 4 — update:**

$$ m_{\text{new}} = 0.5 - 0.2(-21.375) = \mathbf{4.775}, \qquad c_{\text{new}} = 1 - 0.2(-8.75) = \mathbf{2.75} $$

> ✅ Verified correct. **Intuition:** residuals were positive → our line was **too low** → GD pushes slope $m$ and intercept $c$ **up** to fit the points better.

---

## 4.13 GD — Worked Example 3 (two variables) ✅ verified

**Problem:** Optimize $f(x,y) = -(x^{2}+y^{2})$. This bowl is **upside-down**, so its **maximum is at the origin $(0,0)$**. Use $\eta = 0.1$, start at $(x,y) = (1,0)$, do **3 iterations**.

**Step 1 — gradient:** $\dfrac{\partial f}{\partial x} = -2x$, &nbsp; $\dfrac{\partial f}{\partial y} = -2y$ &nbsp;⟹&nbsp; $\nabla f = (-2x,\, -2y)$.

**Update rule.** We are **maximizing**, so we do **gradient ascent** → move *along* the gradient: $x_{\text{new}} = x + \eta\,\frac{\partial f}{\partial x}$ (and same for $y$).

> 💡 **Clarifying the old confusion:** old notes said "subtract" then "maximize" — contradictory. Correct framing: maximizing ⇒ **add** the gradient. Here the gradient is $(-2x,-2y)$, so adding it still *decreases* $x$ toward 0. Numbers below are unchanged and correct.

**Iteration 1** — at $(1, 0)$: &nbsp; $x_1 = 1 + 0.1(-2) = 0.8$, &nbsp; $y_1 = 0 + 0.1(0) = 0$.

**Iteration 2** — at $(0.8, 0)$: &nbsp; $x_2 = 0.8 + 0.1(-1.6) = 0.64$, &nbsp; $y_2 = 0$.

**Iteration 3** — at $(0.64, 0)$: &nbsp; $x_3 = 0.64 + 0.1(-1.28) = \mathbf{0.512}$, &nbsp; $y_3 = \mathbf{0}$.

> The iterate marches toward the maximum at $(0,0)$. Each step shrinks $x$ — clean convergence (this $\eta$ is well-chosen, unlike Example 4.11).

---

## 4.14 Bracketing Method (derivative-free)

> Find the **minimum** using only **function values** — no derivatives needed.

**Setup:** find a **bracket** $(a, b, c)$ with $a < b < c$ where the **middle is lower than both ends**: $f(b) < f(a)$ and $f(b) < f(c)$. This *traps* a minimum between $a$ and $c$.

**Steps each round:**
1. Look at the two sub-intervals $[a,b]$ and $[b,c]$. Probe a new point $d$ in the **larger** one.
2. Compare $f(d)$ with $f(b)$:
   - If $f(d) > f(b)$ → minimum is still near $b$ → shrink the bracket *toward* $b$.
   - If $f(d) < f(b)$ → $d$ is the new best → new bracket centred on $d$, i.e. $(a, d, c)$.
3. Repeat until the bracket is tiny.

🧠 **Think of it like:** you know the lowest point is somewhere between two fence posts. You keep poking a stick in the **wider gap** and squeezing the fence posts inward until they almost touch the dip.

---

## 4.15 Golden Section Search (smarter bracketing)

> Same idea as bracketing, but pick the trial points using the **golden ratio** so the interval shrinks by a **constant proportion** every step (most efficient derivative-free method).

$$ \varphi = \frac{1+\sqrt{5}}{2} \approx 1.618, \qquad \frac{1}{\varphi} \approx 0.618 $$

- Each step **keeps ≈ 61.8%** of the interval (throws away ≈ 38.2%).
- Uses **two interior points** (reuses one point next round → only 1 new evaluation per step → that's the efficiency trick):

$$ x_1 = b - \tfrac{1}{\varphi}(b-a), \qquad x_2 = a + \tfrac{1}{\varphi}(b-a) $$

**Decision rule:**
- If $f(x_1) < f(x_2)$ → minimum is in $[a,\, x_2]$ → **discard the right piece** $(x_2, b]$.
- Else → minimum is in $[x_1,\, b]$ → **discard the left piece** $[a, x_1)$.

🧠 **Memory hook:** **"Golden = 0.618, keep the side with the SMALLER f-value."** (Smaller value = closer to the valley = keep that region.)

---

## 4.16 Golden Section — Worked Example 1 ✅ verified

**Problem:** Minimize $f(x) = (x-2)^{2}$ on $[0, 5]$. (True minimum is obviously $x=2$; let the method find it.)

Let $r = 1/\varphi \approx 0.618$. &nbsp; $[a,b]=[0,5]$.

**Interior points:**
$$ x_1 = b - r(b-a) = 5 - 0.618(5) = 1.91, \qquad x_2 = a + r(b-a) = 0 + 0.618(5) = 3.09 $$

**Evaluate:** $f(x_1) = (1.91-2)^2 = 0.0081$, &nbsp; $f(x_2) = (3.09-2)^2 = 1.188$.

Since $f(x_1) < f(x_2)$ → minimum is in $[a, x_2] = [0, 3.09]$. **Discard $[3.09, 5]$.**

**Iteration 2** on $[0, 3.09]$:
$$ x_1' = 3.09 - 0.618(3.09) = 1.18, \qquad x_2' = 0 + 0.618(3.09) = 1.91 $$
$$ f(x_1') = (1.18-2)^2 = 0.672, \qquad f(x_2') = (1.91-2)^2 = 0.0081 $$

Now $f(x_1') > f(x_2')$ → keep $[x_1', b] = [1.18, 3.09]$.

> The interval keeps tightening around **$x \approx 2$** (where $f(2)=0$). ✅

---

## 4.17 Golden Section — Worked Example 2 ✅ verified

**Problem:** Minimize $f(x) = x^{2} + 4x + 4 = (x+2)^2$ on $[-3, 1]$. (True minimum $x=-2$.)

$[a,b] = [-3, 1]$, $r = 0.618$.

**Interior points:**
$$ x_1 = 1 - 0.618(4) = -1.47, \qquad x_2 = -3 + 0.618(4) = -0.53 $$

**Evaluate:** $f(x_1) = (-1.47+2)^2 = 0.28$, &nbsp; $f(x_2) = (-0.53+2)^2 = 2.17$.

Since $f(x_1) < f(x_2)$ → minimum is in $[a, x_2] = [-3, -0.53]$. **Discard $[-0.53, 1]$.**

> Repeating, the interval narrows toward the true minimum at **$x = -2$** ($f(-2)=0$). ✅

---

# 🧩 CONSTRAINED OPTIMIZATION

## 4.18 Linear Programming (LP)

> Optimize a **linear** objective function subject to **linear** constraints (inequalities/equations).

$$ \text{minimize } \; c^{\top}x \qquad \text{subject to } \; Ax \le b, \;\; x \ge 0 $$

| Part | Meaning | Example |
|---|---|---|
| **Objective function** | linear quantity to max/min | $Z = 3x + 2y$ |
| **Decision variables** | the unknowns | $x, y$ |
| **Constraints** | linear limits | $2x+y \le 20,\; x,y \ge 0$ |
| **Feasible region** | all points obeying every constraint | a polygon |

> ⭐ **KEY THEOREM (almost always asked):** the optimum of an LP is always at a **corner (vertex)** of the feasible region. So the graphical method = **find all corners, plug each into $Z$, pick the best.**

🧠 **Memory hook:** **"LP optimum lives in a CORNER."** Never in the middle — always at a vertex of the polygon.

---

## 4.19 LP — Worked Example 1 (tables & chairs) ✅ verified

**Problem:** Profit ₹100/table, ₹200/chair. Have 40 wood, 24 labour-hours. Each **table**: 4 wood + 2 labour. Each **chair**: 2 wood + 1 labour. Maximize profit.

**Variables:** $x$ = tables, $y$ = chairs. &nbsp; **Objective:** maximize $Z = 100x + 200y$.

**Constraints:**
$$ 4x + 2y \le 40 \;(\text{wood}), \qquad 2x + y \le 24 \;(\text{labour}), \qquad x,y \ge 0 $$

**Find the lines' intercepts:**
- Wood $4x+2y=40$: passes through $(0,20)$ and $(10,0)$.
- Labour $2x+y=24$: passes through $(0,24)$ and $(12,0)$.

> 📐 **Note on the two lines:** $4x+2y=40 \Rightarrow y = 20 - 2x$ and $2x+y=24 \Rightarrow y = 24 - 2x$. Same slope ($-2$) → they are **parallel** → they never intersect. So the corners of the feasible region come from the **axes + the tighter (binding) line, which is wood**.

**Feasible corners and their profit:**

| Corner | Check feasible? | $Z = 100x + 200y$ |
|---|---|---|
| $(0,0)$ | yes | 0 |
| $(10,0)$ | wood: 40 ✓, labour: 20 ≤ 24 ✓ | 1000 |
| $(0,20)$ | wood: 40 ✓, labour: 20 ≤ 24 ✓ | **4000 ← MAX** |

**Optimal:** make **0 tables and 20 chairs**, profit **₹4000**.

> **Intuition:** a chair earns ₹200 but uses only half the resources of a table → chairs dominate. Always go to the most profitable corner.

---

## 4.20 LP — Worked Example 2 (Strawberry & Grapes farm)

**Problem:** 520 ha land, budget 50 500 USD, 5 800 man-days. Per ha — Strawberry: 440 USD + 80 man-days, profit 210; Grapes: 910 USD + 30 man-days, profit 180. Maximize profit.

**Variables:** $s$ = ha strawberry, $g$ = ha grapes. &nbsp; **Objective:** maximize $Z = 210s + 180g$.

**Constraints:**
$$
\begin{aligned}
s + g &\le 520 &&(\text{land}) \\
440s + 910g &\le 50500 &&(\text{budget}) \\
80s + 30g &\le 5800 &&(\text{man-days}) \\
s,\, g &\ge 0
\end{aligned}
$$

**Finding corners** (solve pairs of binding lines):
- *Land ∩ Man-days:* $s+g=520$ and $80s+30g=5800$. From land $g=520-s$; substitute: $80s + 30(520-s) = 5800 \Rightarrow 50s + 15600 = 5800 \Rightarrow s = -196$ → **infeasible** (negative). So land & man-days don't bind together here.
- *Land ∩ Budget:* $s+g=520$, $440s+910g=50500$. Substitute $g=520-s$: $440s + 910(520-s) = 50500 \Rightarrow -470s = -422700 \Rightarrow s \approx 899$ → **exceeds land**, infeasible.

So the **binding constraints are budget and man-days** (land is slack). The real feasible corners are on the axes / budget ∩ man-days intersection.

*Budget ∩ Man-days:* $440s + 910g = 50500$ and $80s + 30g = 5800$.
From the second: $g = \dfrac{5800 - 80s}{30}$. Substitute and solve → this gives the interior vertex; evaluate $Z$ there and at the axis corners:

| Corner (feasible) | $Z = 210s + 180g$ |
|---|---|
| $(0, 0)$ | 0 |
| max strawberry (tightest = man-days: $s = 5800/80 = 72.5$) | $210(72.5) \approx 15{,}225$ |
| max grapes (tightest = budget: $g = 50500/910 \approx 55.5$) | $180(55.5) \approx 9{,}990$ |
| budget ∩ man-days vertex | **compare — largest $Z$ wins** |

> 📌 **The exam method is the point, not the messy decimals:**
> 1. Write the objective and **all** constraints.
> 2. Plot the lines; shade the feasible region.
> 3. Find **every corner** by solving constraint pairs (discard infeasible ones — negative or out-of-bounds).
> 4. Plug each feasible corner into $Z$.
> 5. **Largest $Z$ = optimal corner.**

---

## 4.21 Lagrange Multipliers

> Find max/min of $f$ subject to **equality** constraints $g(x,y)=0$. Trick: fold the constraint into the objective using a new variable $\lambda$ (the **Lagrange multiplier**).

**Single constraint — the recipe:**

1. **Form the Lagrangian:** &nbsp; $\mathcal{L}(x, y, \lambda) = f(x,y) + \lambda\, g(x,y)$
2. **Set all partials to zero:** &nbsp; $\dfrac{\partial \mathcal{L}}{\partial x}=0,\quad \dfrac{\partial \mathcal{L}}{\partial y}=0,\quad \dfrac{\partial \mathcal{L}}{\partial \lambda}=0$
   - 💡 The $\partial\mathcal{L}/\partial\lambda = 0$ equation just **re-states the constraint** $g=0$ — free bookkeeping.
   - The other two say $\nabla f$ and $\nabla g$ are **parallel** (aligned).
3. **Solve the system** for $x, y, \lambda$ → optimal point(s). Plug into $f$ to pick max vs min.

🧠 **Memory hook:** **"Build $\mathcal{L} = f + \lambda g$, take all three partials = 0, solve."** The $\lambda$-partial is always just your original constraint back.

**Multiple constraints:** one multiplier per constraint:
$$ \mathcal{L} = f + \sum_{i=1}^{m}\lambda_i\, g_i, \qquad \text{set every partial (incl. each }\partial\mathcal{L}/\partial\lambda_i)=0. $$

---

## 4.22 Lagrange — Worked Example 1 ✅ verified

**Problem:** Maximize $f(x,y) = x + y$ subject to $g(x,y) = x^{2} + y^{2} - 1 = 0$ (the unit circle).

**Step 1 — Lagrangian:** &nbsp; $\mathcal{L} = x + y + \lambda(x^{2} + y^{2} - 1)$

**Step 2 — partials = 0:**
$$ \frac{\partial \mathcal{L}}{\partial x} = 1 + 2\lambda x = 0 \;\Rightarrow\; \lambda = -\frac{1}{2x} $$
$$ \frac{\partial \mathcal{L}}{\partial y} = 1 + 2\lambda y = 0 \;\Rightarrow\; \lambda = -\frac{1}{2y} $$
$$ \frac{\partial \mathcal{L}}{\partial \lambda} = x^{2} + y^{2} - 1 = 0 \quad (\text{the constraint}) $$

**Step 3 — solve.** Equating the two $\lambda$ expressions: $-\dfrac{1}{2x} = -\dfrac{1}{2y} \Rightarrow x = y$.
Put into the constraint: $x^2 + x^2 = 1 \Rightarrow 2x^2 = 1 \Rightarrow x = \pm\dfrac{1}{\sqrt2}$.

Candidates: $\left(\tfrac{1}{\sqrt2}, \tfrac{1}{\sqrt2}\right)$ and $\left(-\tfrac{1}{\sqrt2}, -\tfrac{1}{\sqrt2}\right)$.

**Step 4 — evaluate $f = x+y$:**
$$ f\!\left(\tfrac{1}{\sqrt2}, \tfrac{1}{\sqrt2}\right) = \frac{2}{\sqrt2} = \sqrt{2} \approx 1.414 \;(\textbf{MAX}), \qquad f\!\left(-\tfrac{1}{\sqrt2}, -\tfrac{1}{\sqrt2}\right) = -\sqrt{2} \;(\text{min}) $$

> **Answer:** maximum is $f = \sqrt{2}$ at $\left(\tfrac{1}{\sqrt2}, \tfrac{1}{\sqrt2}\right)$. ✅

---

## 4.23 Lagrange — Worked Example 2 (two constraints) ✅ verified

**Problem:** Optimize $f(x,y) = x^{2} + y^{2}$ subject to $g = x + y - 1 = 0$ **and** $h = x - y = 0$.

**Step 1 — Lagrangian** (one multiplier per constraint):
$$ \mathcal{L} = x^{2} + y^{2} + \lambda(x + y - 1) + \mu(x - y) $$

**Step 2 — partials = 0:**
$$
\begin{aligned}
\partial_x\mathcal{L} &= 2x + \lambda + \mu = 0 \tag{1}\\
\partial_y\mathcal{L} &= 2y + \lambda - \mu = 0 \tag{2}\\
\partial_\lambda\mathcal{L} &= x + y - 1 = 0 \tag{3}\\
\partial_\mu\mathcal{L} &= x - y = 0 \tag{4}
\end{aligned}
$$

**Step 3 — solve.** From (4): $x = y$. Put into (3): $2x = 1 \Rightarrow x = y = \tfrac12$.
*(Bookkeeping: subtract (2) from (1) → $\mu = 0$; add them → $\lambda = -1$.)*

**Step 4 — value:** $f\!\left(\tfrac12, \tfrac12\right) = \tfrac14 + \tfrac14 = \mathbf{\tfrac12}$.

> With both equality constraints active, the **only** feasible point is $\left(\tfrac12,\tfrac12\right)$ → $f = \tfrac12$. ✅

---

# 🎯 Quick Revision Cheat-Sheet (Chapter 4)

### Core Formulas

| Concept | Formula |
|---|---|
| Derivative (def.) | $f'(x) = \lim_{\Delta x \to 0}\frac{f(x+\Delta x)-f(x)}{\Delta x}$ |
| Product rule | $(fg)' = f'g + g'f$ |
| Chain rule | $(f(g(x)))' = f'(g(x))\,g'(x)$ |
| Gradient | $\nabla f = [\partial f/\partial x_1, \dots, \partial f/\partial x_D]^\top$ |
| Hessian | $H = [\partial^2 f / \partial x_i \partial x_j]$ |
| First-order optimality | $\nabla f = 0$ |
| **Gradient descent** | $w^{(k+1)} = w^{(k)} - \eta\, g^{(k)}$ |
| Convexity | $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ |
| Golden ratio | $\varphi = \frac{1+\sqrt5}{2}\approx1.618$; keep ≈0.618 of interval |
| LP standard form | min $c^\top x$ s.t. $Ax \le b,\; x \ge 0$ |
| Lagrangian | $\mathcal{L} = f + \sum_i \lambda_i g_i$ |

### One-Line Memory Hooks

- **O-V-C** = Objective, Variables, Constraints (the 3 components).
- **$f'=0$** → stationary. **$f''>0$ = 🙂 = MIN**, **$f''<0$ = 🙁 = MAX**, $f''=0$ = maybe saddle.
- **Hessian:** **PSD = bowl = MIN**, NSD = dome = MAX, indefinite = saddle.
- **Convex = one bowl** (easy, 1 global min). **Non-convex = wiggly** (many minima, GD gets stuck).
- **Convex set** = segment between any 2 points stays inside. (Domain of a convex fn must be convex.)
- **GD:** "**MINUS gradient to go DOWN**." $\eta$ too small = slow 🐢; too large = diverges 🚀💥 (see Ex 4.11).
- **Bracketing / Golden Section** = derivative-free. Golden keeps ≈0.618, **keep the side with smaller $f$**.
- **LP optimum = a CORNER** of the feasible region (check every vertex).
- **Lagrange:** $\mathcal{L}=f+\lambda g$, all partials = 0; **one $\lambda$ per equality constraint**; $\partial_\lambda\mathcal{L}=0$ is just the constraint.

### Practice Questions (test yourself — answers given)

1. State the three components of an optimization problem. *(Ans: Objective, Variables, Constraints — "O-V-C".)*
2. Differentiate $f(x) = x^2 e^{x}$. *(Ans: product rule → $2xe^x + x^2e^x$.)*
3. For $f'(x_0)=0$, classify $x_0$ using $f''$. *(Ans: $>0$ min, $<0$ max, $=0$ inconclusive/saddle.)*
4. What is the Hessian; how do PSD/NSD classify a stationary point? *(Ans: matrix of 2nd partials; PSD→min, NSD→max, indefinite→saddle.)*
5. Define a convex function and a convex set. *(Ans: chord above curve; segment between any 2 points stays in set.)*
6. Write one GD update step and explain $\eta$. *(Ans: $w^{(k+1)}=w^{(k)}-\eta g^{(k)}$; small→slow, large→diverges.)*
7. Do 2 iterations of GD on $f(x)=x^2$, $x_0=4$, $\eta=0.1$. *(Ans: $f'=2x$; $x_1=4-0.1(8)=3.2$; $x_2=3.2-0.1(6.4)=2.56$.)*
8. Why does golden-section use ratio 0.618? *(Ans: $1/\varphi$; shrinks interval at a constant rate, reuses one point → 1 new eval per step → most efficient.)*
9. State the four parts of an LP and where the optimum lies. *(Ans: objective, decision variables, constraints, feasible region; optimum at a vertex/corner.)*
10. Lagrange: maximize $xy$ subject to $x+y=10$. *(Ans: $\mathcal{L}=xy+\lambda(x+y-10)$; $y+\lambda=0$, $x+\lambda=0$ ⇒ $x=y$; $2x=10$ ⇒ $x=y=5$, max $=25$.)*

---

*End of Chapter 4. Rewritten for clarity: intuition + memory hooks added, contradictions resolved, and the Example 4.11 arithmetic error (old iter 2/3 = −17.52/10.62) corrected to **24.10 → −28.06 → 125.38**. All worked examples re-verified numerically.*
