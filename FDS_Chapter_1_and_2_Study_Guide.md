# Foundation of Data Science (CS G320) — Study Guide

### Chapters 1 & 2: Introduction to FDS + Probability, Bayesian & Frequentist Approach

---

> ⚠️ **Note from your study buddy:** I rewrote these two chapters to be easier to follow and added **intuition + memory hooks** (the "🧠 Think of it like…" boxes). **All math was re-checked by direct computation and is correct** — no number errors here (unlike old Ch 3 §3.20 and Ch 4 §4.11, which I fixed in those files). Trust the worked examples below.

---

# 📘 CHAPTER 1 — Introduction to Foundation of Data Science

## 1.0 The One-Sentence Story of This Chapter

> **Data Science = drawing useful, reliable conclusions from incomplete data, using computation.**

Chapter 1 has **no formulas** — it's all definitions and distinctions. Examiners ask "define / compare / list" questions here. So your job is to **memorize crisp one-liners and the comparison tables**. The memory hooks below are built for exactly that.

---

## 1.1 What is Data Science?

**Data Science** = an area that **manages, manipulates, extracts, and interprets *knowledge* from huge amounts of data**.

- It is a **multidisciplinary field** built to tackle **big data** challenges.
- But it applies to **all data — big *and* small** (not only giant datasets).

### The 3 Pillars of Data Science (the Venn Diagram)

Data Science sits where **three circles overlap**:

```
        ┌─────────────────────┐
        │  MATHEMATICS &      │
        │  STATISTICS         │
        │  (Probability,      │
        │   Regression,       │
        │   Linear Algebra)   │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐         ┌──────────────────────┐
        │                     │         │  COMPUTER SCIENCE    │
        │     DATA SCIENCE    │◄───────►│  (ML, Algorithms,    │
        │   (centre = all 3   │         │   Data Structures,   │
        │    overlap)         │         │   Programming)       │
        └──────────┬──────────┘         └──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  DOMAIN KNOWLEDGE   │
        │ (Business, Medicine,│
        │  Engineering, Law)  │
        └─────────────────────┘
```

| Pillar | Why it is central |
|---|---|
| **Statistics** | How to make **robust conclusions from incomplete information**. |
| **Computing** | Lets us apply analysis to **large, diverse data** — text, images, video, sensors, not just numbers. |
| **Domain Knowledge** | Lets you **ask the right questions** and **interpret answers correctly**. |

🧠 **Memory hook — "S-C-D": Statistics, Computing, Domain.** The three pillars. Remember: *"DS is more than the sum of its parts because of the **applications**."*

---

## 1.2 What does "Foundation of Data Science" mean?

> **Definition:** *Drawing **useful** conclusions from data using computation.*

Three core activities:

| Activity | What it does | Tool |
|---|---|---|
| **Exploration** | Find patterns in data | **Visualizations** |
| **Inference** | Check if the patterns are **reliable** (will they hold on new data?) | **Randomization** |
| **Prediction** | Make informed guesses about the unknown | **Machine Learning** |

🧠 **Memory hook — "E-I-P / V-R-M":** **E**xplore → **V**isualize, **I**nfer → **R**andomize, **P**redict → **M**achine learning. Chant: *"Explore-Infer-Predict, Visualize-Randomize-ML."*

---

## 1.3 What a Foundation in DS Requires

- Not just *knowing* techniques — **recognizing how they apply to real scenarios**.
- Any collected data is an **incomplete description** of reality.
- **Central challenge of DS:** make **reliable conclusions from partial information**.

### Two Essential Tools

| Tool | What it lets you do | Example |
|---|---|---|
| **Computation** | Use *all* available info, not just a summary | Look at the **whole range** of temperatures, not just the average |
| **Randomness** | Imagine the many ways incomplete data could be "completed" | Generate **many plausible scenarios** consistent with what you saw |

🧠 **Think of it like:** you only have a few puzzle pieces. **Computation** = use every piece you have. **Randomness** = imagine all the ways the missing pieces could fit.

---

## 1.4 Statistical Techniques

- Statistics has always faced the **same challenge as DS**: robust conclusions from incomplete info.
- Its big gift: a **consistent, precise vocabulary** linking **observations → conclusions**.

Three core inferential problems in this course:
1. **Testing hypotheses**
2. **Estimating confidence**
3. **Predicting unknown quantities**

🧠 **Memory hook — "Test, Confidence, Predict"** (the 3 inferential problems).

---

## 1.5 Data Science Goes *Beyond* Statistics

DS **extends** statistics using: Computing · Data visualization · Machine learning · Optimization · Access to information.

- **Fast computers + Internet** → anyone can reach millions of articles, encyclopedias, databases, media.
- **Challenge:** real data often **doesn't follow neat patterns** or standard equations.
- **Warning:** interesting variation **gets hidden** if you over-rely on **simple summaries (like averages)**.

🧠 **One-liner:** *"Averages can lie — they hide the interesting variation."*

---

## 1.6 Data Science vs Machine Learning

| **Machine Learning** | **Data Science** |
|---|---|
| Develop new individual models | Explore many models, build & tune hybrids |
| **Prove** mathematical properties | **Understand empirical** properties |
| A few clean, small datasets | **Massive** datasets |
| Publish a paper 🙂 | **Take action!** |

🧠 **Memory hook:** **ML = "prove & publish." DS = "explore & act."**

---

## 1.7 Data Science vs Data Engineering

| Aspect | **Data Science** | **Data Engineering** |
|---|---|---|
| Approach | Scientific (Exploration) | Engineering (Development) |
| Problems | **Unbounded** | **Bounded** |
| Path to solution | Iterative, exploratory, nonlinear | Mostly linear |
| Education | More is better (PhDs common) | BS / self-taught |
| Presentation skills | Important | Less so |
| Research experience | Important | Less so |
| Programming skills | Less critical | **Critical** |
| Data skills | Important | Important |

🧠 **Memory hook:** Data **Science = explore the UNKNOWN** (research-y, unbounded). Data **Engineering = build the KNOWN** (programming-y, bounded, linear).

---

## 1.8 Data Science Applications

| Domain | Examples |
|---|---|
| **Business Analytics** | Customer analytics, market segmentation, churn prediction |
| **Healthcare** | Disease prediction, drug discovery, healthcare management |
| **Social Media** | Behaviour analysis, sentiment analysis, personalization |
| **Finance** | Risk assessment, algorithmic trading, forecasting |
| **E-commerce** | Recommendation systems, supply-chain & price optimization |
| **Manufacturing** | Predictive maintenance, quality control |
| **Entertainment** | Content recommendation, box-office prediction, gaming |
| **Transportation** | Route optimization, demand forecasting, fleet management |

🧠 **Tip:** if asked, give **any 4–5 domains with one example each** — that's full marks.

---

## 1.9 Tools Used in the Course
- **Anaconda** → bundles Python 3 + IPython libraries + Jupyter notebook.
- Language **R**.
- You'll **write programs, generate images from data, and use real online datasets**.

---

## 1.10 Chapter 1 Takeaway Points (memorize verbatim)
1. **Big Data** gave rise to Data Science.
2. DS is rooted in **maths & statistics, computer science, domain knowledge** (the S-C-D pillars).
3. Data Scientist = an impressive profession.
4. **Not everything with data or science is Data Science!**
5. DS applications are compelling.

---
---

# 📗 CHAPTER 2 — Probability, Bayesian & Frequentist Approach

## 2.0 The One-Sentence Story of This Chapter

> **Probability = measuring uncertainty. Bayes' theorem = updating that uncertainty when new evidence arrives.**

The whole chapter builds one ladder:

```
Classical probability  →  Joint / Union / Marginal  →  Conditional
        →  Bayes' theorem  →  Bayesian updating (belief revision)
```

Every worked example is just *plugging numbers into one of ~6 formulas*. Learn the formula table in the cheat-sheet and you can do them all.

---

## 2.1 Classical Probability

> **Definition:** outcomes giving the **event**, divided by **total possible outcomes**.

$$ P(E) = \frac{n_e}{N} \qquad (N=\text{total outcomes},\; n_e=\text{outcomes in event }E) $$

Properties:
- Each outcome **equally likely**.
- Known **a priori** — *before* doing the experiment.
- For **games of chance** (dice, cards, coins).
- **Objective** — anyone using it correctly gets the *same* number.

**Example:** fair die → P(roll a 4) = 1/6; P(even) = 3/6 = 1/2.

🧠 **Memory hook:** classical probability = **"favourable ÷ total,"** and it's *a priori* (before you roll) and *objective* (everyone agrees).

---

## 2.2 Structure of Probability — Key Terms

| Term | Meaning |
|---|---|
| **Experiment** | A process with several possible outcomes (one outcome per trial) |
| **Trial** | One repetition of the process |
| **Event** | An outcome of an experiment |
| **Sample Space** | The set of **all** elementary events |
| **Union (A∪B)** | A **OR** B occurs |
| **Intersection (A∩B)** | A **AND** B both occur |
| **Mutually Exclusive** | **Cannot happen together** |
| **Independent** | One **does not affect** the other |
| **Collectively Exhaustive** | Together they **cover the whole sample space** |
| **Complementary (~A)** | "Not A" |

🧠 **Don't mix these up:** **Mutually exclusive** = can't both happen (e.g. a coin can't be Heads *and* Tails). **Independent** = one doesn't change the other's chance (e.g. two separate coin tosses). These are *different* ideas — a common exam trap.

### Worked Example — Tiny Town ✅ verified

**Experiment:** pick **two families without replacement** from Tiny Town.

| Family | Children? | Automobiles |
|---|---|---|
| A | Yes | 3 |
| B | Yes | 2 |
| C | No | 1 |
| D | Yes | 2 |

**Sample space** — ordered pairs, no family twice → **12 outcomes**:
```
(A,B) (A,C) (A,D)
(B,A) (B,C) (B,D)
(C,A) (C,B) (C,D)
(D,A) (D,B) (D,C)
```

- **"Both families have children"** → kids = A, B, D. Pairs: (A,B),(A,D),(B,A),(B,D),(D,A),(D,B) = **6** → $P = 6/12 = \mathbf{0.5}$.
- **"Total of four automobiles"** → sums to 4: B+D (2+2) and A+C (3+1). Ordered: (B,D),(D,B),(A,C),(C,A) = **4** → $P = 4/12 = \mathbf{1/3}$.

🧠 **Method:** *count favourable outcomes, divide by total* — that's just classical probability (§2.1) applied to a list.

---

## 2.3 Basic Probability Notation

| Notation | Meaning |
|---|---|
| **P(A)** | Probability of A |
| **P(B)** | Probability of B |
| **P(A,B)** | **Joint** — A AND B both occur |

```
   ___        ___
  /   \      /   \
 |  A  |░░░░|  B  |     ░░ = overlap = P(A,B)
  \___/      \___/
```

🧠 **Memory hook:** the **comma means AND**. $P(A,B)$ = "A and B together" = the *overlap*.

---

## 2.4 Probability Distributions

A **distribution** $P(X)$ = how probability is spread over the values of $X$.

### (a) Discrete → PMF (Probability Mass Function)
- $X$ takes **countable** values (1, 2, …, 100).
- Each value gets a probability; **they sum to 1**: $\sum_X PMF(X) = 1$.
- Picture: a stick at each value.

### (b) Continuous → PDF (Probability Density Function)
- $X$ takes values on a **continuous range** (e.g. height).
- **P(exact single value) = 0** (e.g. P(height = exactly 1.8 m) = 0).
- Probability = **area under the curve**: $P(a \le X \le b) = \int_a^b p(x)\,dx$.

```
PDF p(x)
   |        .-''-.
   |      .'  ██  '.      ██ = area between a and b = the probability
   |     /   ████   \
   |____/____████____\______ x
            a      b
```

🧠 **Memory hook:** **Discrete → SUM the PMF = 1. Continuous → INTEGRATE the PDF (area) = probability.** For continuous, a *single point* has probability **0** (area of a line = 0).

---

## 2.5 Union & Complement

### Union (A **OR** B)
$$ P(A \cup B) = P(A) + P(B) - P(A,B) $$
We subtract $P(A,B)$ so the **overlap isn't double-counted**.

### Complement (anything but A)
$$ P(\sim A) = 1 - P(A) $$

🧠 **Memory hook:** **"OR = add, but subtract the overlap once."** (If A and B are mutually exclusive, overlap = 0, so it's just $P(A)+P(B)$.)

---

## 2.6 Marginal Probability ✅ verified

Joint table (disease X vs symptom Y):

| | X=0 | X=1 |
|---|---|---|
| **Y=0** | 0.5 | 0.1 |
| **Y=1** | 0.1 | 0.3 |

All joint probabilities sum to 1: $0.5+0.1+0.1+0.3 = 1$ ✓.

**Marginal** = sum a whole row or column (collapse the other variable):

$$ P(X=x) = \sum_{y} P(X=x, Y=y) $$

- **Joint:** $P(X{=}0, Y{=}1) = \mathbf{0.1}$ (just read the cell).
- **Marginal:** $P(Y{=}1) = 0.1 + 0.3 = \mathbf{0.4}$ (sum the Y=1 **row**).
- **Marginal:** $P(X{=}0) = 0.5 + 0.1 = \mathbf{0.6}$ (sum the X=0 **column**).

🧠 **Memory hook:** a **marginal** lives in the **margin** of the table — you get it by **summing** (collapsing) one variable away.

---

## 2.7 Conditional Probability ✅ verified

> "Probability of A **given that** B already happened."

$$ \boxed{\;P(A \mid B) = \frac{P(A,B)}{P(B)}\;} $$

Same disease/symptom table. Given **Y=1** (row total $= 0.1 + 0.3 = 0.4$):

$$ P(X{=}1 \mid Y{=}1) = \frac{0.3}{0.4} = 0.75, \qquad P(X{=}0 \mid Y{=}1) = \frac{0.1}{0.4} = 0.25 $$

They sum to 1 (given Y=1, X is either 0 or 1).

🧠 **Think of it like:** "given B" means **B is now your whole world** — you shrink the sample space to just B, then ask how much of *that* is also A. That's why you divide by $P(B)$.

---

## 2.8 Conditional Probability — Bigger Example (Cancer test) ✅ verified

**Given:** $P(C)=\tfrac{1}{100}$, $P(NC)=\tfrac{99}{100}$, $P(+\mid C)=\tfrac{90}{100}$, $P(+\mid NC)=\tfrac{8}{100}$.
**Find:** $P(C \mid +)$.

**Step 1:** $P(C \mid +) = \dfrac{P(C,+)}{P(+)}$.

**Step 2 — $P(C,+)$** (product rule $P(C,+)=P(+\mid C)P(C)$):
$$ P(C,+) = \frac{90}{100}\times\frac{1}{100} = \frac{9}{1000} = 0.009 $$

**Step 3 — $P(NC,+)$:**
$$ P(NC,+) = \frac{8}{100}\times\frac{99}{100} = \frac{792}{10000} = 0.0792 $$

**Step 4 — $P(+)$** (every way to test positive):
$$ P(+) = 0.009 + 0.0792 = 0.0882 $$

**Step 5 — answer:**
$$ P(C \mid +) = \frac{0.009}{0.0882} \approx \mathbf{0.102 \;\;(\approx 10\%)} $$

> 🧠 **The big lesson — the base-rate effect:** the test is 90% accurate, yet a positive result means only **~10%** real chance of cancer — because cancer is **rare (1%)**. This counter-intuitive result is the *entire reason Bayes' theorem matters*. Expect a conceptual question on this.

---

## 2.9 Bayes' Theorem

From the conditional definition:
$$ P(A\cap B) = P(B \mid A)\,P(A) \quad\Rightarrow\quad \boxed{\,P(A \mid B) = \dfrac{P(B \mid A)\, P(A)}{P(B)}\,} $$

**Expanded denominator** (law of total probability):
$$ P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)} $$

🧠 **Memory hook — "flip the conditional":** Bayes lets you go from $P(B\mid A)$ (often easy to know) to $P(A\mid B)$ (what you actually want). Chant: *"posterior = likelihood × prior ÷ evidence."*

---

## 2.10 Bayes' — Example 1 (Liver disease / alcoholism) ✅ verified

> 10% have liver disease. 5% are alcoholics. Of those **with liver disease**, 7% are alcoholics. Find P(liver disease | alcoholic).

- $P(A)=P(\text{liver})=0.10$, $P(B)=P(\text{alcoholic})=0.05$, $P(B\mid A)=0.07$.

$$ P(A \mid B) = \frac{P(B\mid A)\,P(A)}{P(B)} = \frac{0.07 \times 0.10}{0.05} = \mathbf{0.14 \;(14\%)} $$

---

## 2.11 Bayes' — Example 2 (Diagnostic test) ✅ verified

> Disease in **0.5%** of people. Test positive in **99%** with disease, **5%** without (false positive). Person tests positive — chance they have it?

- $P(+\mid D)=0.99$, $P(D)=0.005$, $P(\sim D)=0.995$.

**Step 1 — $P(+)$ via total probability:**
$$ P(+) = (0.99)(0.005) + (0.05)(0.995) = 0.00495 + 0.04975 = \mathbf{0.0547} $$

**Step 2 — Bayes:**
$$ P(D \mid +) = \frac{0.99 \times 0.005}{0.0547} \approx \mathbf{0.0905 \;(\approx 9\%)} $$

> 🧠 **Same base-rate lesson as 2.8:** a 99%-accurate test → still only **~9%** real chance, because the disease is rare. *(Note: $P(+)$ is exactly 0.0547; the slides round to ≈0.05 and the answer to ≈9%.)*

**Notation reminder:** $P(D)$ = has disease; $P(\sim D)=1-P(D)$; $P(+\mid D)$ = positive given disease; $P(+\mid \sim D)$ = false positive.

---

## 2.12 Frequentist vs Bayesian Statistics

Both use $Y = X\theta + \varepsilon$ (Y output, X data, θ parameters, ε error).

| | **Frequentist** | **Bayesian** |
|---|---|---|
| Data X | **Random** | **Fixed** |
| Parameters θ | **Unknown but fixed** | **Random variables** |
| True model | One true parameter set; estimate it | No single truth; params more/less probable |
| Interested in | **Point estimate** of θ | **Distribution** of θ |

🧠 **Memory hook (the one-liner examiners want):**
- **Frequentist:** *"params FIXED, data RANDOM → give me a point estimate."*
- **Bayesian:** *"data FIXED, params RANDOM → give me a whole distribution."*

It's a clean **swap of what's random**. That's the whole distinction.

---

## 2.13 Bayesian Inference

- A **dynamic model** — belief is **constantly updated** as data arrives.
- Goal: the **posterior**, which is **∝ likelihood × prior**.

### Bayes' Rule (named parts)

$$ \underbrace{P(\theta \mid D)}_{\textbf{Posterior}} = \frac{\overbrace{P(D \mid \theta)}^{\textbf{Likelihood}} \times \overbrace{P(\theta)}^{\textbf{Prior}}}{\underbrace{P(D)}_{\textbf{Evidence}}} \;\propto\; P(D \mid \theta)\,P(\theta) $$

| Term | Meaning |
|---|---|
| **Prior** $P(\theta)$ | Belief **before** seeing data |
| **Likelihood** $P(D\mid\theta)$ | How well the params explain the observed data |
| **Evidence** $P(D)$ | Normalizing constant (prob. of the data) |
| **Posterior** $P(\theta\mid D)$ | **Updated** belief **after** data |

🧠 **Memory hook — "PLEP": Prior, Likelihood, Evidence, Posterior.** And remember the shape: *"Posterior ∝ Likelihood × Prior"* (evidence is just the scaler).

**Effects on the posterior:**
- **Stronger prior** → posterior pulled more toward the prior. (Flat Beta(1,1) barely influences; a peaked prior shifts a lot.)
- **More data** → **likelihood dominates** → posterior gets **narrower (more certain)**, centred near the data, *regardless of the prior*.

```
Posterior  ∝  Likelihood  ×  Prior
   ↑              ↑             ↑
 updated       what data     what we
 belief        tells us      believed before
```

---

## 2.14 Worked Example — Coin Flipping (Bayesian updating) ✅ verified

**Setup:** unknown coin, only see outcomes. Two hypotheses:
- **H1 fair:** 50/50 Heads → prior $P(\text{fair}) = 0.99$
- **H2 unfair:** always Heads → prior $P(\text{unfair}) = 0.01$

### 1st Flip → Heads
Want $P(\text{fair}\mid H)$. Knowns: $P(H\mid \text{fair})=0.5$, $P(H\mid \text{unfair})=1$.

**Evidence:** $P(H) = (0.5)(0.99) + (1)(0.01) = 0.5050$.

$$ P(\text{fair}\mid H) = \frac{0.5\times0.99}{0.5050} = \mathbf{0.9802} $$

Belief the coin is fair drops slightly: $0.99 \to 0.9802$.

### 2nd Flip → Heads again
> 🧠 **KEY IDEA: the posterior from step 1 becomes the new prior for step 2.** This *is* Bayesian updating.

New prior $P(\text{fair})=0.9802 \Rightarrow P(\text{unfair})=0.0198$.

New evidence: $P(H) = (0.5)(0.9802) + (1)(0.0198) = 0.5099$.

$$ P(\text{fair}\mid H) = \frac{0.5\times0.9802}{0.5099} = \mathbf{0.9612} $$

**Pattern:** each consecutive Heads makes us **less sure** it's fair: $0.99 \to 0.9802 \to 0.9612 \to \dots$ — belief revised step by step as evidence piles up.

🧠 **Think of it like:** a detective. Each new clue (flip) updates the suspicion. The unfair coin *always* gives Heads, so every Heads is weak evidence against "fair" — slowly, suspicion grows.

---

## 2.15 The Famous Bayesian Quote (worth quoting in an answer)

> *"A Bayesian is one who, vaguely expecting a horse, and catching a glimpse of a donkey, strongly believes he has seen a mule."*

**Meaning:** combine **prior (horse)** + **evidence (donkey)** → a compromise **conclusion (mule)**. Perfect one-line illustration of how the prior pulls the posterior.

---

## 2.16 References (from slides)
- *Bayesian statistics (a very brief introduction)* — Ken Rice
- statisticshowto.com/bayes-theorem-problems/
- Slides "Bayesian inference and generative models" — K.E. Stephan
- Intro slides to probabilistic & unsupervised learning — M. Sahani
- blog.stata.com/2016/11/01/introduction-to-bayesian-statistics-part-1-the-basic-concepts/

---
---

# 🎯 Quick Revision Cheat-Sheet (Chapters 1 & 2)

### Core Formulas

| Concept | Formula |
|---|---|
| Classical probability | $P(E) = n_e / N$ |
| Joint | $P(A,B)$ (comma = AND) |
| Union | $P(A\cup B) = P(A)+P(B)-P(A,B)$ |
| Complement | $P(\sim A) = 1 - P(A)$ |
| Marginal | $P(X=x) = \sum_y P(X=x, Y=y)$ |
| Conditional | $P(A\mid B) = \dfrac{P(A,B)}{P(B)}$ |
| Product rule | $P(A\cap B) = P(B\mid A)\,P(A)$ |
| **Bayes' theorem** | $P(A\mid B) = \dfrac{P(B\mid A)\,P(A)}{P(B)}$ |
| Total probability | $P(B) = P(B\mid A)P(A) + P(B\mid \neg A)P(\neg A)$ |
| Bayesian rule | Posterior ∝ Likelihood × Prior |

### One-Line Memory Hooks
- **3 pillars = "S-C-D"**: Statistics, Computing, Domain knowledge.
- **3 activities = "E-I-P / V-R-M"**: Explore→Visualize, Infer→Randomize, Predict→ML.
- **DS vs ML** → DS *explores & acts*; ML *proves & publishes*.
- **DS vs Data Engineering** → DS unbounded/exploratory; DE bounded/linear/programming.
- **Mutually exclusive ≠ Independent** (can't-both vs doesn't-affect).
- **Comma = AND** ($P(A,B)$ = overlap). **Union = add minus overlap.**
- **PMF sum = 1; PDF area = probability; point prob = 0** (continuous).
- **Marginal = sum a row/column** (collapse a variable).
- **Conditional**: "given B" shrinks the world to B → divide by $P(B)$.
- **Base-rate effect**: accurate test + rare condition → still low posterior (2.8, 2.11).
- **Frequentist** = params fixed, data random, point estimate. **Bayesian** = params random, data fixed, distribution.
- **PLEP** = Prior, Likelihood, Evidence, Posterior. *Posterior ∝ Likelihood × Prior.*
- **Bayesian updating** = yesterday's posterior is today's prior (coin example 2.14).

### Practice Questions (answers given)
1. State the three pillars of data science and why each matters. *(Stats=robust conclusions; Computing=large diverse data; Domain=right questions/interpretation.)*
2. Define experiment, trial, event, sample space. *(See §2.2.)*
3. A die is rolled. P(odd)? *(Ans: 3/6 = 0.5)*
4. From the disease/symptom table, compute $P(X{=}1 \mid Y{=}0)$. *(Ans: $0.1/(0.5+0.1) = 0.1/0.6 \approx 0.167$)*
5. Redo the 2.11 Bayes example with disease rate 1% instead of 0.5% — does the posterior go up or down? *(Up — higher base rate → higher posterior. Recompute: $P(+)=0.99(0.01)+0.05(0.99)=0.0594$, $P(D\mid+)=0.0099/0.0594\approx0.167$, i.e. ~17%, up from 9%.)*
6. Why does the coin's "fair" probability keep dropping with each Heads? *(The unfair coin always gives Heads, so each Heads is evidence against "fair"; posterior becomes the next prior.)*
7. State Bayes' theorem and label prior, likelihood, evidence, posterior. *(See §2.13 — PLEP.)*
8. Difference between frequentist and Bayesian treatment of θ? *(Frequentist: θ fixed/unknown, point estimate. Bayesian: θ random, full distribution.)*

---

*End of Chapters 1 & 2. Rewritten for clarity: intuition + memory hooks added, mutually-exclusive-vs-independent trap flagged, all worked examples re-verified numerically (no number errors found here). Chapters 3 & 4 are in their own files.*
