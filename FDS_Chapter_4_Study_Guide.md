# Foundation of Data Science (CS G320) — Study Guide

### Chapter 4: Optimization

*(Continuation of Chapters 1–3. Read alongside the other chapter guides in this folder.)*

---

# 📕 CHAPTER 4 — Optimization

## 4.0 Roadmap of the Chapter

```
                          OPTIMIZATION
                               │
        ┌──────────────────────┼───────────────────────┐
   FOUNDATIONS            UNCONSTRAINED             CONSTRAINED
   ───────────            ─────────────             ───────────
   Optima of functions    1st-order optimality      Linear Programming
   Derivatives & rules     Gradient Descent          (graphical method)
   Saddle points           Bracketing method         Lagrange Multipliers
   Multivariate / Gradient Golden-section search      (equality constraints)
   Hessian
   Convexity / convex sets
```

**Big idea:** optimization = **find the input that makes an objective function as small (or large) as possible**, possibly subject to constraints.

---

## 4.1 What is Optimization?

> The process of **finding the best values for the variables** of a particular problem to **minimize or maximize an objective function** — i.e. making the best or most effective use of a given situation.

### Three components of every optimization problem

| Component | Meaning |
|---|---|
| **1. Objective function** | The performance of a system to be **minimized or maximized** |
| **2. Variables** | The set of **unknowns** we control |
| **3. Constraints** | **Conditions** the variables must satisfy |

### Classification tree

```
                 Optimization Problem
                 ┌────────┼─────────┐
            Variables  Constraints  Objective Function
              │            │              │
        ┌─────┴────┐  ┌────┴─────┐   ┌────┴────┐
   Continuous  Discrete  Constrained  Single  Multi
                     Unconstrained
```

---

## 4.2 Functions and Their Optima

- Most ML problems = **maximizing or minimizing** a function $f$ of some variable(s) $x$.
- For simplicity assume $f$ is a **scalar-valued function of a scalar**: $f : \mathbb{R} \to \mathbb{R}$.

| Term | Meaning |
|---|---|
| **Local minimum / maximum** | Smallest/largest value in a *neighbourhood* |
| **Global minimum / maximum** | Smallest/largest value over the *entire* domain |
| **Saddle point** | Derivative is zero but it is **neither** a max nor a min |

```
 f(x)
   |   /\        global max
   |  /  \    /\
   | /    \  /  \___ local min
   |/      \/        \
   |________________________ x
        local        global
        max           min
```

> Finding optima or saddles means looking at **derivatives / gradients** of the function.

---

## 4.3 Derivatives

> The **magnitude of the derivative at a point** is the **rate of change** of the function at that point.

$$ \frac{df(x)}{dx} = \lim_{\Delta x \to 0} \frac{f(x + \Delta x) - f(x)}{\Delta x} $$

| Sign of $f'(x)$ | Meaning |
|---|---|
| **Positive** | function is **increasing** |
| **Negative** | function is **decreasing** |
| **Zero** | a **stationary point** (optimum or saddle) |

- Understanding how $f$ changes its value as we change $x$ is helpful in optimisation algorithms.
- The condition for a stationary point: $\left|\dfrac{df(x)}{dx}\right| = 0$ — we change $x$ by a very little at such points; these are the points where the function has its maxima/minima (unless the saddle).

---

## 4.4 Rules of Derivatives

| Rule | Formula |
|---|---|
| **Sum rule** | $\big(f(x)+g(x)\big)' = f'(x) + g'(x)$ |
| **Scaling rule** | $\big(a\cdot f(x)\big)' = a\cdot f'(x)$  ( $a$ not a function of $x$ ) |
| **Product rule** | $\big(f(x)\cdot g(x)\big)' = f'(x)\,g(x) + g'(x)\,f(x)$ |
| **Quotient rule** | $\left(\dfrac{f(x)}{g(x)}\right)' = \dfrac{f'(x)g(x) - g'(x)f(x)}{\big(g(x)\big)^{2}}$ |
| **Chain rule** | $\big(f(g(x))\big)' = f'(g(x))\cdot g'(x)$ |

> **Memory hook for chain rule:** *"derivative of the outside (keeping inside) × derivative of the inside."*

---

## 4.5 The Second Derivative & Saddle Points

How the derivative *itself* changes tells us about the function's optima:

| At a stationary point ( $f'(x)=0$ ) | If… | Then it is a |
|---|---|---|
| $f''(x) > 0$ | curve bends **up** | **local minimum** |
| $f''(x) < 0$ | curve bends **down** | **local maximum** |
| $f''(x) = 0$ | inconclusive | possibly a **saddle / inflection** |

### Saddle Points

> Points where the derivative is zero but are **neither minima nor maxima**.

- Very common for **loss functions of deep learning models** — must be handled carefully during optimization.
- Second or higher derivatives may help identify whether a stationary point is a saddle.

```
 saddle: up in one direction, down in another  (looks like a horse saddle / Pringle)
```

---

## 4.6 Multivariate Functions & the Gradient

> Most functions we use in ML are **multivariable** — e.g. the loss $L(w)$ in linear regression is a function of a $D$-dimensional weight vector $w$,  $L : \mathbb{R}^{D} \to \mathbb{R}$.

The **gradient** of $f : \mathbb{R}^{D} \to \mathbb{R}$ is the $D \times 1$ **vector of partial derivatives**:

$$ \nabla f(x) = \left[ \frac{\partial f}{\partial x_1},\; \frac{\partial f}{\partial x_2},\; \dots,\; \frac{\partial f}{\partial x_D} \right]^{\top} $$

- Optima & saddle points: same idea as 1-D, but the condition $\nabla f(x) = 0$ must hold **along all directions** (every partial derivative = 0).
- The "second derivative" in this case is a matrix — the **Hessian**.

---

## 4.7 The Hessian

> For a multivariate scalar function $f : \mathbb{R}^{D} \to \mathbb{R}$, the **Hessian** is the $D \times D$ matrix of **second-order partial derivatives**.

$$ \nabla^{2} f(x) = H = \begin{bmatrix} \dfrac{\partial^{2}f}{\partial x_1^{2}} & \dfrac{\partial^{2}f}{\partial x_1 \partial x_2} & \cdots \\[3mm] \dfrac{\partial^{2}f}{\partial x_2 \partial x_1} & \dfrac{\partial^{2}f}{\partial x_2^{2}} & \cdots \\[2mm] \vdots & \vdots & \ddots \end{bmatrix} $$

At a stationary point ( $\nabla f(x) = 0$ ):

| Hessian is… | Then $x$ is a |
|---|---|
| **Positive semi-definite (PSD)** | **local minimum** |
| **Negative semi-definite (NSD)** | **local maximum** |
| **Indefinite** | **saddle point** |

> **Memory hook:** Hessian = "the matrix of curvatures." PSD ⇒ bowl-up ⇒ min; NSD ⇒ dome ⇒ max.

---

## 4.8 Convex & Non-Convex Functions

> A function being optimized can be **convex** or **non-convex**.

### Convex function (informal)

> $f(x)$ is **convex** if all of its chords lie **above** the function everywhere.

$$ f\big(\lambda x + (1-\lambda)y\big) \;\le\; \lambda f(x) + (1-\lambda)f(y), \qquad 0 \le \lambda \le 1 $$

```
 convex:  ╲___╱   chord (straight line) always sits ABOVE the curve
 non-convex: wiggly — multiple local minima, chord can cut below
```

| | **Convex** | **Non-convex** |
|---|---|---|
| Minima | exactly **one** (global) | possibly **many** local minima |
| Optimization | easy — any local min is global | hard — can get trapped |

### Convex Sets

> A set $S$ is a **convex set** if, for any two points $x, y \in S$ and any $0 \le \lambda \le 1$:

$$ z = \lambda x + (1-\lambda)y \;\in\; S $$

i.e. **the whole line segment between any two points of the set also lies inside the set**.

> The **domain of a convex function must be a convex set**.

---

# 🔧 UNCONSTRAINED OPTIMIZATION

## 4.9 Optimization Using First-Order Optimality

> **First-order optimality:** the **gradient must equal zero** at the optima.

$$ g = \nabla\big[L(w)\big] = 0 $$

- Sometimes, setting $g = 0$ and solving gives a **closed-form** solution.
- If a closed-form solution is **not available**, the gradient vector can still be used in **iterative** optimization algorithms like **gradient descent**.

---

## 4.10 Gradient Descent

> Iterative algorithm: repeatedly step in the **negative (opposite) direction of the gradient** (downhill) until convergence.

**Algorithm:**

1. Initialize $w$ as $w^{(0)}$.
2. For iteration $k = 0, 1, 2, \dots$ (or until convergence):
   - Compute the gradient $g^{(k)}$ using the current iterate $w^{(k)}$.
   - Set the **learning rate** $\eta_k$.
   - Update (move in the **negative gradient** direction):

$$ \boxed{\; w^{(k+1)} = w^{(k)} - \eta_k\, g^{(k)} \;} $$

> **Learning rate $\eta$ is critical:**
> - **Too small** → very slow convergence.
> - **Too large** → overshoots, may diverge.
> - Good initialization also matters (helps escape poor regions).

```
 L(w)
   |  \                     start w0
   |   \    .               |
   |    \  / \   each step moves downhill ↓
   |     \/   \__ . __ . __ ●  global min
   |________________________ w
```

---

## 4.11 GD — Numerical Example 1

**Problem:** Minimize $f(x) = x^{2}\sin(x) + 10\sin(x) - 50$. Take learning rate $\eta = 0.2$ and start at $x^{(0)} = 9$. Perform **3 iterations** of gradient descent and show the calculations.

**Step 1 — derivative** (product rule on $x^2\sin x$ + chain on $10\sin x$):

$$ f'(x) = \big(2x\sin x + x^{2}\cos x\big) + 10\cos x $$

**Update rule:** $x_{\text{new}} = x_{\text{old}} - \eta\,f'(x_{\text{old}})$, with $\eta = 0.2$.

**Iteration 1** — at $x^{(0)} = 9$:

$$ f'(9) = 2(9)\sin 9 + 9^{2}\cos 9 + 10\cos 9 $$

Using radians: $\sin 9 \approx 0.412$, $\cos 9 \approx -0.911$.

$$ f'(9) \approx 18(0.412) + 81(-0.911) + 10(-0.911) \approx 7.42 - 73.79 - 9.11 = -75.48 $$

$$ x^{(1)} = 9 - 0.2(-75.48) = 9 + 15.10 = \mathbf{24.10} $$

**Iteration 2** — at $x^{(1)} \approx 24.10$ ($\sin \approx -0.905$, $\cos \approx 0.426$):

$$ f'(24.10) \approx 2(24.10)(-0.905) + 24.10^{2}(0.426) + 10(0.426) \approx -43.6 + 247.4 + 4.3 = 208.1 $$

$$ x^{(2)} = 24.10 - 0.2(208.1) = 24.10 - 41.62 = \mathbf{-17.52} $$

**Iteration 3** — at $x^{(2)} \approx -17.52$ ($\sin \approx 0.94$, $\cos \approx -0.34$):

$$ f'(-17.52) \approx 2(-17.52)(0.94) + (-17.52)^{2}(-0.34) + 10(-0.34) \approx -32.9 - 104.4 - 3.4 = -140.7 $$

$$ x^{(3)} = -17.52 - 0.2(-140.7) = -17.52 + 28.14 = \mathbf{10.62} $$

> **Exam point:** the *method* — compute $f'$, plug current $x$, apply $x_{\text{new}} = x_{\text{old}} - \eta f'$, repeat. (Large jumps here illustrate why a step size that is too big causes the iterate to bounce around.)

---

## 4.12 GD — Numerical Example 2 (Mean Squared Error)

**Problem:** Minimize the **Mean Squared Error (MSE)** by gradient descent for a simple linear model $\hat{y}_i = m x_i + c$.

$$ \mathrm{MSE} = \frac{1}{N}\sum_{i=1}^{N}\big(y_i - \hat{y}_i\big)^{2} $$

**Gradients w.r.t. parameters $m$ and $c$:**

$$ \frac{\partial \mathrm{MSE}}{\partial m} = -\frac{2}{N}\sum_{i} x_i\big(y_i - \hat{y}_i\big), \qquad \frac{\partial \mathrm{MSE}}{\partial c} = -\frac{2}{N}\sum_{i}\big(y_i - \hat{y}_i\big) $$

**Updates:**

$$ m_{\text{new}} = m - \eta\,\frac{\partial \mathrm{MSE}}{\partial m}, \qquad c_{\text{new}} = c - \eta\,\frac{\partial \mathrm{MSE}}{\partial c} $$

**Given:** initial $m = 0.5$, $c = 1$; learning rate $\eta = 0.2$; data $(x_i, y_i) = (1.5, 5),\,(3, 8)$.

**Predictions** with $m=0.5, c=1$: $\hat y_1 = 0.5(1.5)+1 = 1.75$, $\hat y_2 = 0.5(3)+1 = 2.5$.

**Residuals:** $r_1 = 5 - 1.75 = 3.25$, $r_2 = 8 - 2.5 = 5.5$.

**Gradients** ($N=2$):

$$ \frac{\partial \mathrm{MSE}}{\partial m} = -\frac{2}{2}\big[1.5(3.25) + 3(5.5)\big] = -(4.875 + 16.5) = -21.375 $$

$$ \frac{\partial \mathrm{MSE}}{\partial c} = -\frac{2}{2}\big[3.25 + 5.5\big] = -8.75 $$

**Update:**

$$ m_{\text{new}} = 0.5 - 0.2(-21.375) = 0.5 + 4.275 = \mathbf{4.775} $$

$$ c_{\text{new}} = 1 - 0.2(-8.75) = 1 + 1.75 = \mathbf{2.75} $$

> One GD step pushed $m$ and $c$ toward fitting the two points (residuals were positive ⇒ predictions too low ⇒ slope & intercept increase).

---

## 4.13 GD — Numerical Example 3 (two variables)

**Problem:** Minimize $f(x,y) = -(x^{2}+y^{2})$ by **gradient ascent** (maximize) / descent. $\eta = 0.1$, initial $(x,y) = (1,0)$. Do **3 iterations**.

**Step 1 — gradient:**

$$ \frac{\partial f}{\partial x} = -2x, \qquad \frac{\partial f}{\partial y} = -2y \;\Rightarrow\; \nabla f = (-2x,\, -2y) $$

**Update (for maximizing, move *along* the gradient):** $x_{\text{new}} = x + \eta\frac{\partial f}{\partial x}$, similarly for $y$.

**Iteration 1** — at $(1, 0)$:

$$ x_1 = 1 + 0.1(-2\cdot 1) = 1 - 0.2 = 0.8, \qquad y_1 = 0 + 0.1(-2\cdot 0) = 0 $$

**Iteration 2** — at $(0.8, 0)$:

$$ x_2 = 0.8 + 0.1(-1.6) = 0.8 - 0.16 = 0.64, \qquad y_2 = 0 $$

**Iteration 3** — at $(0.64, 0)$:

$$ x_3 = 0.64 + 0.1(-1.28) = 0.64 - 0.128 = \mathbf{0.512}, \qquad y_3 = \mathbf{0} $$

> The iterate marches toward the maximum at $(0,0)$ (where $f = 0$). Each step shrinks $x$ by a factor — classic convergence toward the optimum.

---

## 4.14 Bracketing Method

> Direct method (no derivatives) to find the **minimum** of a scalar function $f(x)$, $f : \mathbb{R} \to \mathbb{R}$, using only function values (curvature/gradient not used).

Start with a **bracket** $(a, b, c)$ where $a < b < c$ **and** $f(b) < f(a)$ **and** $f(b) < f(c)$ (the middle point is lower than both ends → a minimum is *trapped* inside).

Pick a new trial point $d$ in one of the intervals:

- If $c - b > b - a$ → take $d = \frac{a+b}{2}$ … (probe the larger sub-interval)
- If $f(d) > f(b)$ → the new bracket is $(d, b, c)$ or $(a, b, d)$ accordingly.
- If $f(d) < f(b)$ → the new bracket is $(a, d, c)$.
- Continue until the **distance between the extremes of the bracket is small**.

> **Idea:** keep shrinking an interval that is guaranteed to contain the minimum, choosing the mid point in the **bigger** of the two sub-intervals each time.

---

## 4.15 Golden Section Search

> Improves bracketing by choosing trial points using the **golden ratio** $\varphi$ instead of a plain midpoint.

$$ \varphi = \frac{1+\sqrt{5}}{2} \approx 1.618 $$

- Reduces the search interval size at a constant rate each iteration (the **most efficient** single-point method without derivatives).
- The proportion of the kept interval is about **0.618** (= $1/\varphi$) each step → the bracket shrinks by ≈ 38.2% per evaluation.
- Independent of the function being minimized; uses **two interior points**:

$$ x_1 = a + (1-\tfrac{1}{\varphi})(b-a), \qquad x_2 = a + \tfrac{1}{\varphi}(b-a) $$

**Rule:** if $f(x_1) < f(x_2)$ the minimum lies in $[a, x_2]$ → update new interval to $[a, x_2]$; otherwise it lies in $[x_1, b]$.

---

## 4.16 Golden Section Search — Numerical Example 1

**Problem:** Find the minimum of the **unimodal** function $f(x) = (x-2)^{2}$ (no derivatives) over $[0, 5]$.

Let $r = 1/\varphi \approx 0.618$. Interval $[a,b] = [0,5]$, length $= 5$.

**Two interior points:**

$$ x_1 = b - r(b-a) = 5 - 0.618(5) = 1.91, \qquad x_2 = a + r(b-a) = 0 + 0.618(5) = 3.09 $$

**Evaluate:**

$$ f(x_1) = (1.91-2)^{2} = 0.0081, \qquad f(x_2) = (3.09-2)^{2} = 1.188 $$

Since $f(x_1) < f(x_2)$ → minimum lies in $[a, x_2] = [0, 3.09]$. **Discard $[3.09, 5]$.**

**Iteration 2** on $[0, 3.09]$ (length 3.09):

$$ x_1' = 3.09 - 0.618(3.09) = 1.18, \qquad x_2' = 0 + 0.618(3.09) = 1.91 $$

$$ f(x_1') = (1.18-2)^2 = 0.672, \qquad f(x_2') = (1.91-2)^2 = 0.0081 $$

Now $f(x_1') > f(x_2')$ → new interval $[x_1', b] = [1.18, 3.09]$.

> Repeating these steps the interval keeps shrinking toward **$x \approx 2$** (where $f$ is minimum, $f(2)=0$).

---

## 4.17 Golden Section Search — Numerical Example 2

**Problem:** Minimize $f(x) = x^{2} + 4x + 4 = (x+2)^2$ over the interval $[-3, 1]$.

$[a,b] = [-3, 1]$, length $= 4$, $r = 0.618$.

**Interior points:**

$$ x_1 = b - r(b-a) = 1 - 0.618(4) = -1.47, \qquad x_2 = a + r(b-a) = -3 + 0.618(4) = -0.53 $$

**Evaluate:**

$$ f(x_1) = (-1.47+2)^{2} = 0.28, \qquad f(x_2) = (-0.53+2)^{2} = 2.16 $$

Since $f(x_1) < f(x_2)$ → minimum lies in $[a, x_2] = [-3, -0.53]$. **Discard $[-0.53, 1]$.**

> Repeating, the interval narrows toward the true minimum at **$x = -2$** ($f(-2) = 0$).

---

# 🧩 CONSTRAINED OPTIMIZATION

## 4.18 Linear Programming (LP)

> A mathematical technique for **optimizing a linear objective function**, subject to a set of **linear inequalities or equations** (constraints).

$$ \text{minimize } \; c^{\top}x \qquad \text{subject to } \; Ax \le b, \;\; x \ge 0 $$

Four parts of an LP:

| Part | Meaning | Example |
|---|---|---|
| **Objective function** | the linear quantity to maximize/minimize | $Z = 3x + 2y$ |
| **Decision variables** | the unknowns we solve for | $x, y$ = quantities to produce |
| **Constraints** | limitations/restrictions (linear) | $2x + y \le 20$, $x \ge 0, y \ge 0$ |
| **Feasible region** | the set of all points satisfying the constraints | polygon; optimum lies **at a vertex** |

> **Key theorem:** the optimal solution of an LP lies at a **corner (vertex)** of the feasible region — so the graphical method just checks all corners.

---

## 4.19 LP — Numerical Example 1 (tables & chairs)

**Problem:** A factory produces **tables** and **chairs**. Profit: **₹100** per table, **₹200** per chair. Resources: **40 units of wood** total; **24 labour-hours** total. Each table uses 4 wood & 2 labour; each chair uses 2 wood & 1 labour. How many tables & chairs to **maximize profit**?

**Decision variables:** $x$ = tables, $y$ = chairs.

**Objective:** maximize $Z = 100x + 200y$.

**Constraints:**

$$ 4x + 2y \le 40 \quad(\text{wood}), \qquad 2x + y \le 24 \quad(\text{labour}), \qquad x \ge 0,\; y \ge 0 $$

**Graphical method — find corner points** (intercepts of the binding lines):

- Wood line $4x+2y=40$: points $(0,20)$ and $(10,0)$.
- Labour line $2x+y=24$: points $(0,24)$ and $(12,0)$.

Feasible-region corners: $(0,0),\,(10,0),\,(0,20)$, and the intersection of the two lines.

Solving $4x+2y=40$ and $2x+y=24$: from the second $y = 24-2x$; sub into the first: $4x + 2(24-2x) = 40 \Rightarrow 48 = 40$ → **inconsistent**, so the lines are parallel; the binding constraint is wood. Evaluate $Z=100x+200y$ at the feasible corners:

| Corner | $Z = 100x + 200y$ |
|---|---|
| $(0,0)$ | 0 |
| $(10,0)$ | 1000 |
| $(0,20)$ | **4000  ← maximum** |

**Optimal:** make **0 tables and 20 chairs**, profit **₹4000** (chairs are far more profitable per resource used).

---

## 4.20 LP — Numerical Example 2 (Strawberry & Grapes farm)

**Problem:** A farmer has **520 hectares** of land and decides how much to use for Strawberry vs Grapes. Total budget = **50 500 USD**; total man-days available = **5 800**. Per hectare: Strawberry uses **440 USD** & **80 man-days**; Grapes uses **910 USD** & **30 man-days**. Profit: **210 USD/ha** Strawberry, **180 USD/ha** Grapes. Find the optimal plan (graphical method).

**Variables:** $s$ = hectares of Strawberry, $g$ = hectares of Grapes.

**Objective:** maximize $Z = 210s + 180g$.

**Constraints:**

$$ s + g \le 520 \quad(\text{land}) $$
$$ 440s + 910g \le 50500 \quad(\text{budget}) $$
$$ 80s + 30g \le 5800 \quad(\text{man-days}) $$
$$ s \ge 0,\; g \ge 0 $$

**Solve corner points** (intersection of binding constraints):

- Land ∩ Man-days: $s+g=520$ and $80s+30g=5800$. From land $g = 520 - s$; sub: $80s + 30(520-s) = 5800 \Rightarrow 50s + 15600 = 5800 \Rightarrow s = -196$ → infeasible, so man-days is not binding with land here.
- Use **Land ∩ Budget**: $s+g=520$, $440s+910g=50500$. From land $g=520-s$; sub: $440s + 910(520-s) = 50500 \Rightarrow -470s = 50500 - 473200 \Rightarrow s \approx 538$ → exceeds land, infeasible.
- Feasible vertices are the axis/intercept corners; evaluate $Z = 210s + 180g$ at each feasible corner (e.g. $(0,0)$, max-$s$ on the tightest constraint, max-$g$):

| Corner (approx) | $Z = 210s + 180g$ |
|---|---|
| $(0, 0)$ | 0 |
| all Strawberry (tightest: budget/man-day cap) | compare |
| all Grapes (tightest constraint) | compare |
| Land ∩ Budget feasible vertex | **maximum (largest $Z$)** |

> **Exam method (the real point):** (1) write objective & all constraints, (2) plot lines & shade feasible region, (3) find every corner by solving pairs of equations, (4) plug each corner into $Z$, (5) pick the **largest $Z$** — that corner is optimal. (The numeric answer depends on the exact intercepts read from the graph.)

---

## 4.21 Lagrange Multipliers

> A method to find the **local maxima/minima of a function subject to *equality* constraints**. It introduces a new variable, the **Lagrange multiplier** $\lambda$, to fold the constraint into the objective.

**Single constraint.** Optimize $f(x,y)$ subject to a constraint $g(x,y) = 0$.

**Steps:**

1. **Form the Lagrangian:**

$$ \mathcal{L}(x, y, \lambda) = f(x,y) + \lambda\, g(x,y) $$

2. **Find the partial derivatives** of $\mathcal{L}$ and set each to zero:

$$ \frac{\partial \mathcal{L}}{\partial x} = 0, \qquad \frac{\partial \mathcal{L}}{\partial y} = 0, \qquad \frac{\partial \mathcal{L}}{\partial \lambda} = 0 $$

> The last equation just re-states the constraint $g(x,y)=0$. The first two say the gradients of $f$ and $g$ are **aligned**: $\nabla f = -\lambda \nabla g$.

3. **Solve the system** for $x, y, \lambda$ → gives the optimal point.

### Multiple constraints

For $m$ equality constraints $g_1, \dots, g_m$, introduce one multiplier per constraint:

$$ \mathcal{L}(x, y, \lambda_1, \dots, \lambda_m) = f(x,y) + \sum_{i=1}^{m}\lambda_i\, g_i(x,y) $$

Set all partials (including each $\partial\mathcal{L}/\partial\lambda_i$) to zero and solve.

---

## 4.22 Lagrange — Numerical Example 1

**Problem:** Maximize $f(x,y) = x + y$ subject to the constraint $g(x,y) = x^{2} + y^{2} - 1 = 0$ (a unit circle).

**Step 1 — Lagrangian:**

$$ \mathcal{L}(x,y,\lambda) = x + y + \lambda\big(x^{2} + y^{2} - 1\big) $$

**Step 2 — partial derivatives = 0:**

$$ \frac{\partial \mathcal{L}}{\partial x} = 1 + 2\lambda x = 0 \;\Rightarrow\; \lambda = -\frac{1}{2x} $$

$$ \frac{\partial \mathcal{L}}{\partial y} = 1 + 2\lambda y = 0 \;\Rightarrow\; \lambda = -\frac{1}{2y} $$

$$ \frac{\partial \mathcal{L}}{\partial \lambda} = x^{2} + y^{2} - 1 = 0 \quad (\text{the constraint}) $$

**Step 3 — solve.** From the first two: $-\dfrac{1}{2x} = -\dfrac{1}{2y} \Rightarrow x = y$.

Substitute into the constraint: $x^{2} + x^{2} = 1 \Rightarrow 2x^{2} = 1 \Rightarrow x = \pm\dfrac{1}{\sqrt{2}}$.

So the candidates are $\left(\dfrac{1}{\sqrt2}, \dfrac{1}{\sqrt2}\right)$ and $\left(-\dfrac{1}{\sqrt2}, -\dfrac{1}{\sqrt2}\right)$.

**Evaluate $f = x+y$:**

$$ f\!\left(\tfrac{1}{\sqrt2}, \tfrac{1}{\sqrt2}\right) = \frac{2}{\sqrt2} = \sqrt{2} \;\;(\textbf{maximum}), \qquad f\!\left(-\tfrac{1}{\sqrt2}, -\tfrac{1}{\sqrt2}\right) = -\sqrt{2} \;\;(\text{minimum}) $$

> **Optimal solution:** $\left(\dfrac{1}{\sqrt2}, \dfrac{1}{\sqrt2}\right)$ with **maximum $f = \sqrt{2}$**.

---

## 4.23 Lagrange — Numerical Example 2 (two constraints)

**Problem:** Find the maximum of $f(x,y) = x^{2} + y^{2}$ subject to **two** constraints $g(x,y) = x + y - 1 = 0$ and $h(x,y) = x - y = 0$.

**Step 1 — Lagrangian** (one multiplier per constraint):

$$ \mathcal{L}(x, y, \lambda, \mu) = x^{2} + y^{2} + \lambda(x + y - 1) + \mu(x - y) $$

**Step 2 — partial derivatives = 0:**

$$ \frac{\partial \mathcal{L}}{\partial x} = 2x + \lambda + \mu = 0 \tag{1} $$
$$ \frac{\partial \mathcal{L}}{\partial y} = 2y + \lambda - \mu = 0 \tag{2} $$
$$ \frac{\partial \mathcal{L}}{\partial \lambda} = x + y - 1 = 0 \tag{3} $$
$$ \frac{\partial \mathcal{L}}{\partial \mu} = x - y = 0 \tag{4} $$

**Step 3 — solve.** From (4): $x = y$. Substitute into (3): $x + x - 1 = 0 \Rightarrow 2x = 1 \Rightarrow x = y = \dfrac{1}{2}$.

(Subtract (2) from (1): $2x - 2y + 2\mu = 0 \Rightarrow \mu = 0$ since $x=y$; add them: $2x+2y+2\lambda = 0 \Rightarrow \lambda = -(x+y) = -1$.)

**Step 4 — the optimum value:**

$$ f\!\left(\tfrac12, \tfrac12\right) = \left(\tfrac12\right)^{2} + \left(\tfrac12\right)^{2} = \tfrac14 + \tfrac14 = \mathbf{\tfrac12} $$

> With both equality constraints active, the only feasible point is $\left(\tfrac12,\tfrac12\right)$, giving $f = \tfrac12$.

---

# 🎯 Quick Revision Cheat-Sheet (Chapter 4)

### Core Formulas

| Concept | Formula |
|---|---|
| Derivative (def.) | $f'(x) = \lim_{\Delta x \to 0}\frac{f(x+\Delta x)-f(x)}{\Delta x}$ |
| Product rule | $(fg)' = f'g + g'f$ |
| Chain rule | $(f(g(x)))' = f'(g(x))g'(x)$ |
| Gradient | $\nabla f = [\partial f/\partial x_1, \dots, \partial f/\partial x_D]^\top$ |
| Hessian | $H = [\partial^2 f / \partial x_i \partial x_j]$ |
| First-order optimality | $\nabla f = 0$ |
| Gradient descent | $w^{(k+1)} = w^{(k)} - \eta\, g^{(k)}$ |
| Convexity | $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ |
| Golden ratio | $\varphi = \frac{1+\sqrt5}{2} \approx 1.618$, keep ≈0.618 of interval |
| LP standard form | min $c^\top x$ s.t. $Ax \le b,\; x \ge 0$ |
| Lagrangian | $\mathcal{L} = f + \sum_i \lambda_i g_i$ |

### Key Distinctions to Memorize
- **$f' = 0$**: stationary point. **$f''>0$** → min, **$f''<0$** → max, **$f''=0$** → maybe saddle.
- **Multivariate:** $\nabla f = 0$ + Hessian → PSD = min, NSD = max, indefinite = saddle.
- **Convex** → one global min (easy). **Non-convex** → many local minima (hard, GD can get stuck).
- **Convex set** → segment between any two points stays inside (domain of convex function must be convex).
- **GD learning rate:** too small = slow; too large = diverges.
- **Bracketing / Golden section** = derivative-free; golden section shrinks interval by constant ≈0.382 ratio (most efficient).
- **LP optimum** is always at a **vertex** of the feasible region (graphical = check all corners).
- **Lagrange:** one multiplier per **equality** constraint; set all partials (incl. $\partial\mathcal{L}/\partial\lambda$) to 0.

### Practice Questions to Test Yourself
1. State the three components of an optimization problem.
2. Differentiate $f(x) = x^2 e^{x}$ using the product rule. *(Ans: $2xe^x + x^2e^x$)*
3. For $f'(x_0)=0$, how do you classify $x_0$ using $f''$?
4. What is the Hessian and how does PSD/NSD classify a stationary point?
5. Define a convex function and a convex set.
6. Write one GD update step and explain the role of $\eta$.
7. Do 2 iterations of GD on $f(x)=x^2$, $x_0=4$, $\eta=0.1$. *(Ans: $x_1=3.2$, $x_2=2.56$)*
8. Why does golden-section search use ratio 0.618?
9. State the four parts of a linear program and where the optimum lies.
10. Use Lagrange multipliers to maximize $xy$ subject to $x+y=10$. *(Ans: $x=y=5$, max $=25$.)*

---

*End of Chapter 4. Built in the same style as Chapters 1–3 — definitions, tables, formulas, diagrams, fully worked examples, and a cheat-sheet.*
