# Foundation of Data Science (CS G320) — Study Guide

### Chapters 1 & 2: Introduction to FDS + Probability, Bayesian & Frequentist Approach

---

# 📘 CHAPTER 1 — Introduction to Foundation of Data Science

## 1.1 What is Data Science?

**Data Science** is an area that **manages, manipulates, extracts, and interprets *knowledge* from a tremendous amount of data**.

Key points:
- It is a **multidisciplinary field** whose goal is to address the challenges in **big data**.
- Data science principles apply to **all data — big *and* small** (not just huge datasets).

### The 3 Pillars of Data Science (the Venn Diagram)

Data Science sits at the **intersection of three fields**. Picture three overlapping circles:

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
        │   (the centre = all │         │   Data Structures,   │
        │    three overlap)   │         │   Programming)       │
        └──────────┬──────────┘         └──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │  DOMAIN KNOWLEDGE   │
        │  (Business, Medicine,│
        │   Engineering, Law)  │
        └─────────────────────┘
```

| Pillar | Why it is central |
|---|---|
| **Statistics** | Studies how to make **robust conclusions with incomplete information**. |
| **Computing** | Programming lets us apply analysis techniques to **large & diverse datasets** — not just numbers but text, images, videos, sensor readings. |
| **Domain Knowledge** | Understanding a domain lets data scientists **ask the right questions** and **correctly interpret answers**. |

> **Key line to remember:** *Data science is all of these things, but it is more than the sum of its parts because of the applications.*

---

## 1.2 What does "Foundation of Data Science" mean?

> **Definition:** *Drawing **useful** conclusions from data using computation.*

It has **three core activities**:

| Activity | What it does | Tool used |
|---|---|---|
| **Exploration** | Identifying patterns in information | **Visualizations** |
| **Inference** | Quantifying whether those patterns are **reliable** (degree of certainty — will patterns reappear in new data? how accurate are predictions?) | **Randomization** |
| **Prediction** | Making informed guesses | **Machine Learning** |

**Mnemonic:** **E → I → P** = Explore (visualize), Infer (randomize), Predict (ML).

---

## 1.3 What a Foundation in DS Requires

- Not just *understanding* statistical and computational techniques, **but recognizing how they apply to real scenarios**.
- Whatever we study (weather, markets, polls), collected data gives an **incomplete description** of the subject.
- **Central challenge of DS:** make **reliable conclusions using partial information**.

### Two Essential Tools

| Tool | What it lets us do | Example |
|---|---|---|
| **Computation** | Use *all* available information to draw conclusions | Instead of just the *average* temperature, consider the **whole range** of temperatures for a nuanced analysis |
| **Randomness** | Consider the many ways incomplete information might be "completed" | Instead of assuming temperatures vary a fixed way, use randomness to **imagine many possible scenarios** consistent with the data |

---

## 1.4 Statistical Techniques

- Statistics has long addressed the **same fundamental challenge** as data science: *how to draw robust conclusions about the world using incomplete information.*
- An important contribution of statistics: a **consistent and precise vocabulary** for describing the relationship between **observations and conclusions**.

The course focuses on **three core inferential problems**:
1. **Testing hypotheses**
2. **Estimating confidence**
3. **Predicting unknown quantities**

---

## 1.5 Data Science Goes *Beyond* Statistics

Data science **extends statistics** by taking full advantage of:
- Computing
- Data visualization
- Machine learning
- Optimization
- Access to information

The combination of **fast computers + the Internet** lets anyone access vast datasets: millions of news articles, full encyclopedias, domain databases, massive music/photo/video repositories.

**Challenge:** Real data often **does not follow regular patterns** or match standard equations.

**Warning:** Interesting variation in real data **can be lost** if you focus too much on **simplistic summaries (like averages)**.

---

## 1.6 Data Science vs Machine Learning

| **Machine Learning** | **Data Science** |
|---|---|
| Develop new (individual) models | Explore many models, build & tune hybrids |
| **Prove** mathematical properties of models | **Understand empirical** properties of models |
| Improve/validate on a few, relatively clean, small datasets | Develop/use tools that handle **massive** datasets |
| Publish a paper 🙂 | **Take action!** |

> Memory hook: ML = *prove & publish*. DS = *explore & act*.

---

## 1.7 Data Science vs Data Engineering

| Aspect | **Data Science** | **Data Engineering** |
|---|---|---|
| Approach | Scientific (Exploration) | Engineering (Development) |
| Problems | **Unbounded** | **Bounded** |
| Path to solution | Iterative, exploratory, nonlinear | Mostly linear |
| Education | More is better (PhDs common) | BS and/or self-trained |
| Presentation skills | Important | Not as important |
| Research experience | Important | Not as important |
| Programming skills | Not as important | **Important** |
| Data skills | Important | Important |

> Hook: Data **Science = explore the unknown** (research-y). Data **Engineering = build the known** (programming-y).

---

## 1.8 Data Science Applications

| Domain | Example applications |
|---|---|
| **Business Analytics** | Customer analytics, market segmentation, churn prediction |
| **Healthcare** | Disease prediction, drug discovery, healthcare management |
| **Social Media** | User behaviour analysis, sentiment analysis, content personalization |
| **Finance** | Risk assessment, algorithmic trading, financial forecasting |
| **E-commerce** | Recommendation systems, supply chain optimization, price optimization |
| **Manufacturing** | Predictive maintenance, quality control, supply chain management |
| **Entertainment** | Content recommendation, box office prediction, gaming |
| **Transportation** | Route optimization, demand forecasting, fleet management |

---

## 1.9 Tools Used in the Course
- **Anaconda** distribution → packages Python 3 interpreter + IPython libraries + Jupyter notebook environment.
- Programming language **R**.
- You'll learn to **write programs, generate images from data, and work with real-world online datasets**.

---

## 1.10 Chapter 1 Takeaway Points (memorize)
1. **Big Data** has given rise to Data Science.
2. Data science is rooted in solid foundations of **mathematics & statistics, computer science, and domain knowledge**.
3. Data Scientist = an impressive profession.
4. **Not everything with data or science is Data Science!**
5. The applications of Data Science are compelling.

---
---

# 📗 CHAPTER 2 — Probability, Bayesian & Frequentist Approach

## 2.1 Classical Probability

> **Definition:** Number of outcomes leading to the **event**, divided by the **total number of outcomes possible**.

$$ P(E) = \frac{n_e}{N} $$

Where:
- **N** = total number of outcomes
- **nₑ** = number of outcomes in event E

Properties of classical probability:
- Each outcome is **equally likely**.
- Determined **a priori** — *before* performing the experiment.
- Applicable to **games of chance** (dice, cards, coins).
- **Objective** — everyone correctly using the method assigns an *identical* probability.

**Example:** Rolling a fair die, P(rolling a 4) = 1/6 (one favourable outcome ÷ six total outcomes). P(even number) = 3/6 = 1/2.

---

## 2.2 Structure of Probability — Key Terms

| Term | Meaning |
|---|---|
| **Experiment** | A process that produces outcomes (more than one possible outcome; only one outcome per trial) |
| **Trial** | One repetition of the process |
| **Event** | An outcome of an experiment |
| **Sample Space** | The set of **all elementary events** for an experiment |
| **Union (A∪B)** | A **OR** B occurs |
| **Intersection (A∩B)** | A **AND** B both occur |
| **Mutually Exclusive** | Events that **cannot happen together** |
| **Independent** | One event's occurrence **does not affect** the other |
| **Collectively Exhaustive** | The events **cover the entire sample space** |
| **Complementary (~A)** | "Not A" — everything other than A |

### Worked Example — Tiny Town

**Experiment:** Randomly select, **without replacement**, two families from Tiny Town.

| Family | Children in Household | Number of Automobiles |
|---|---|---|
| A | Yes | 3 |
| B | Yes | 2 |
| C | No | 1 |
| D | Yes | 2 |

**Sample space** (ordered pairs, no family chosen twice — 12 outcomes):
```
(A,B) (A,C) (A,D)
(B,A) (B,C) (B,D)
(C,A) (C,B) (C,D)
(D,A) (D,B) (D,C)
```

- **Event: "each family in the sample has children"** → families with children = A, B, D. Favourable pairs: (A,B),(A,D),(B,A),(B,D),(D,A),(D,B) → 6 outcomes → P = 6/12 = **0.5**.
- **Event: "sample families own a total of four automobiles"** → pairs whose autos sum to 4: B+D = 2+2 = 4 → (B,D),(D,B); also A+C = 3+1 = 4 → (A,C),(C,A). So 4 outcomes → P = 4/12 = **1/3**.

---

## 2.3 Basic Probability Notation

| Notation | Meaning |
|---|---|
| **P(A)** | Probability of A occurring |
| **P(B)** | Probability of B occurring |
| **P(A,B)** | **Joint probability** — A AND B both occur |

Venn picture of joint probability: two overlapping circles; the **overlap** is P(A,B).

```
   ___        ___
  /   \      /   \
 |  A  |░░░░|  B  |     ░░ = overlap = P(A,B)
  \___/      \___/
```

---

## 2.4 Probability Distributions

A **probability distribution** P(X) describes how probability is spread over the possible values of a variable X. Two types:

### (a) Discrete → PMF (Probability Mass Function)
- X takes **countable** distinct values (e.g. 1, 2, …, 100).
- Each value gets a probability (e.g. each = 1/100).
- **Rule:** the probabilities **sum to 1**:

$$ \sum_{X} PMF(X) = 1 $$

Picture: a bar/stick at each value, each of height 1/100.

### (b) Continuous → PDF (Probability Density Function)
- X takes values on a **continuous range** (e.g. height of UAE population).
- The probability of an **exact single value is 0** (e.g. P(height = exactly 1.8 m) = 0).
- Probability is given by the **area under the curve** over an interval:

$$ P(a \le X \le b) = \int_{a}^{b} p(x)\,dx $$

Example: P(1.75 ≤ X ≤ 1.85) = shaded area under the bell curve between 1.75 and 1.85.

```
PDF p(x)
   |        .-''-.
   |      .'  ██  '.      ██ = area between a and b = the probability
   |     /   ████   \
   |____/____████____\______ x
            a      b
```

**Memory hook:** Discrete → **sum** the PMF = 1. Continuous → **integrate** the PDF (area) = the probability.

---

## 2.5 Union & Complement

### Union Probability (A **OR** B)

$$ P(A \cup B) = P(A) + P(B) - P(A,B) $$

(We subtract P(A,B) so the overlap isn't counted twice.)

### Complement (anything other than A)

$$ P(\sim A) = 1 - P(A) $$

```
 ┌──────────────────────┐
 │   Sample Space        │
 │      ___              │
 │     / A \   ← P(A)    │
 │     \___/             │
 │  everything else =    │
 │      ~A = 1 - P(A)    │
 └──────────────────────┘
```

---

## 2.6 Marginal Probability

Given a **joint probability table** (disease X vs symptoms Y):

| | X=0 | X=1 |
|---|---|---|
| **Y=0** | 0.5 | 0.1 |
| **Y=1** | 0.1 | 0.3 |

**Rule:** all joint probabilities sum to 1:

$$ \sum_{x,y} P(X=x, Y=y) = 1 \quad (0.5+0.1+0.1+0.3 = 1) $$

**Marginal probability** = sum a row or column to "marginalize out" the other variable:

$$ P(X=x) = \sum_{y} P(X=x, Y=y) $$

Worked values:
- **Joint:** P(X=0, Y=1) = **0.1** (just read the cell)
- **Marginal:** P(Y=1) = 0.1 + 0.3 = **0.4** (sum the Y=1 row)
- **Marginal:** P(X=0) = 0.1 + 0.5 = **0.6** (sum the X=0 column)

> Mnemonic: a **marginal** lives in the "margin" of the table — you collapse one variable by **summing**.

---

## 2.7 Conditional Probability

> "What is the probability of A occurring, **given that** B has occurred?"

$$ P(A \mid B) = \frac{P(A,B)}{P(B)} $$

### Worked Example — same disease/symptom table

| | X=0 | X=1 |
|---|---|---|
| **Y=0** | 0.5 | 0.1 |
| **Y=1** | 0.1 | 0.3 |

General form: $P(X \mid Y) = \dfrac{P(X=x, Y=y)}{P(Y=y)}$

Given symptoms Y=1 (row total = 0.1 + 0.3 = 0.4):

$$ P(X=1 \mid Y=1) = \frac{0.3}{0.1 + 0.3} = \frac{0.3}{0.4} = 0.75 $$

$$ P(X=0 \mid Y=1) = \frac{0.1}{0.1 + 0.3} = \frac{0.1}{0.4} = 0.25 $$

(They sum to 1 because, given Y=1, X must be either 0 or 1.)

---

## 2.8 Conditional Probability — Bigger Worked Example (Cancer test)

**Given:**
- P(C) = probability of cancer = 1/100
- P(NC) = probability of no cancer = 99/100
- P(+ | C) = probability of positive test given cancer = 90/100
- P(+ | NC) = probability of positive test given no cancer (false positive) = 8/100

**Find:** P(C | +) — probability of cancer given a positive test.

**Step 1 — Use the formula:**
$$ P(C \mid +) = \frac{P(C, +)}{P(+)} $$

**Step 2 — Find P(C, +)** using the product rule  P(C,+) = P(+|C) × P(C):
$$ P(C,+) = \frac{90}{100} \times \frac{1}{100} = \frac{9}{1000} $$

**Step 3 — Find P(NC, +):**
$$ P(NC,+) = P(+\mid NC) \times P(NC) = \frac{8}{100} \times \frac{99}{100} = \frac{792}{10000} $$

**Step 4 — Find P(+)** by summing over all ways a positive test can happen:
$$ P(+) = P(C,+) + P(NC,+) = \frac{9}{1000} + \frac{792}{10000} $$

**Step 5 — Final answer:**
$$ P(C \mid +) = \frac{\frac{9}{1000}}{\frac{9}{1000} + \frac{792}{10000}} \approx 0.1 $$

> **Key insight:** Even though the test is 90% accurate for cancer patients, because cancer itself is rare (1%), a positive test only means ~**10%** chance of actually having cancer. This is the famous **base-rate effect** and is the whole motivation for Bayes' theorem.

---

## 2.9 Bayes' Theorem

Built from the definition of conditional probability:

$$ P(B \mid A) = \frac{P(B \cap A)}{P(A)} \quad\Rightarrow\quad P(A \cap B) = P(B \mid A) \times P(A) $$

Substitute into  $P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}$  to get **Bayes' theorem**:

$$ \boxed{\,P(A \mid B) = \dfrac{P(B \mid A) \times P(A)}{P(B)}\,} $$

### Alternative form (expand the denominator)

$$ P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)} $$

(This is just P(B) written using the **law of total probability**.)

---

## 2.10 Bayes' Theorem — Example 1 (Liver disease / alcoholism)

> 10% of patients have liver disease. 5% are alcoholics. Among those **with liver disease**, 7% are alcoholics. Find the probability a patient has liver disease **given** he is an alcoholic.

**Define:**
- P(A) = P(liver disease) = 0.10
- P(B) = P(alcoholism) = 0.05
- P(B | A) = P(alcoholic given liver disease) = 0.07
- P(A | B) = ?

**Apply Bayes:**
$$ P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)} = \frac{0.07 \times 0.10}{0.05} = 0.14 $$

**Answer:** If the patient is an alcoholic, his chance of having liver disease is **0.14 (14%)**.

---

## 2.11 Bayes' Theorem — Example 2 (Diagnostic test)

> A disease occurs in **0.5%** of the population. A test is positive in **99%** of people **with** the disease, and **5%** of people **without** it (false positive). A person tests positive — what's the probability they have the disease?

**Step 1 — Bayes' formula:**
$$ P(\text{disease} \mid +) = \frac{P(+ \mid \text{disease}) \times P(\text{disease})}{P(+)} $$

**Known:**
- P(+ | disease) = 0.99
- P(disease) = 0.005  → so P(~disease) = 1 − 0.005 = 0.995
- P(+) = ??? (must compute)

**Step 2 — Compute P(+) via law of total probability:**
$$ P(+) = P(+\mid D)\,P(D) + P(+\mid \sim D)\,P(\sim D) $$
$$ = (0.99 \times 0.005) + (0.05 \times 0.995) = 0.00495 + 0.04975 \approx 0.05 $$

**Step 3 — Final answer:**
$$ P(\text{disease} \mid +) = \frac{0.99 \times 0.005}{0.05} = 0.09 \quad \textbf{(i.e. 9%)} $$

> Again the **base-rate lesson**: a positive result on a 99%-accurate test still gives only a **9%** chance of the disease, because the disease is very rare.

**Notation reminder:**
- P(D) = chance of having the disease
- P(~D) = chance of *not* having it, and P(~D) = 1 − P(D)
- P(PT|D) = chance of positive test **given disease present**
- P(PT|~D) = chance of positive test **given disease absent** (false positive)

---

## 2.12 Frequentist vs Bayesian Statistics

Both use the model:  $Y = X\theta + \varepsilon$  (Y = output, X = data, θ = parameters, ε = error)

| | **Frequentist** | **Bayesian** |
|---|---|---|
| Data X | **Random** variable | **Fixed** |
| Parameters θ | **Unknown but fixed** | **Random variables** |
| True model | There **is** a single true set of parameters; we want the best estimate | There is **no single** true model — parameters are more or less probable |
| Interested in | **Point estimates** of parameters given data | **Distribution** of parameters given data |

> One-line hook: **Frequentist** = "parameters are fixed, data is random → give me a point estimate." **Bayesian** = "data is fixed, parameters are random → give me a whole distribution."

---

## 2.13 Bayesian Inference

- Provides a **dynamic model** — our belief is **constantly updated** as more data arrives.
- Ultimate goal: calculate the **posterior probability density**, which is **proportional to** the **likelihood** × **prior**.
- Can model the brain ("Bayesian brain"), history, human behaviour.

### Bayes' Rule (with named parts)

$$ \underbrace{P(\theta \mid D)}_{\textbf{Posterior}} = \frac{\overbrace{P(D \mid \theta)}^{\textbf{Likelihood}} \times \overbrace{P(\theta)}^{\textbf{Prior}}}{\underbrace{P(D)}_{\textbf{Evidence}}} \;\propto\; P(D \mid \theta) \times P(\theta) $$

| Term | Meaning |
|---|---|
| **Prior** P(θ) | What we believed about parameters **before** seeing data |
| **Likelihood** P(D\|θ) | How well the parameters explain the observed data ("how good are our parameters given the data?") |
| **Evidence** P(D) | Normalizing constant (probability of the data) |
| **Posterior** P(θ\|D) | Our **updated** belief about parameters **after** seeing data |

### Principles of Bayesian Inference
1. **Formulation of a model** → produces the **Likelihood function P(D|θ)** and the **Prior distribution P(θ)**.
2. **Observation of data** → Measurement → data **D**.

### Effects on the Posterior
- **More informative prior** → the prior pulls the posterior more strongly toward the prior's belief (a flat Beta(1,1) prior barely influences it; a peaked prior shifts it a lot).
- **Larger sample size** → the **likelihood dominates**; the posterior becomes **narrower (more certain)** and centered near the data, regardless of the prior.

```
Posterior  ∝  Likelihood  ×  Prior
   ↑              ↑             ↑
 updated       what data     what we
 belief        tells us      believed before
```

---

## 2.14 Worked Example — Coin Flipping Model (Bayesian updating)

**Setup:** Someone flips a coin. We don't know if it's fair. We only see the outcomes.

**Two hypotheses:**
- **H1 (fair coin):** 50% Heads / 50% Tails → prior **P(A = fair) = 0.99**
- **H2 (unfair coin):** both sides Heads, 100% Heads → prior **P(A = unfair) = 0.01**

### 1st Flip → Heads

We want **P(A = fair | B = Heads)**:

$$ P(A=fair \mid B=Heads) = \frac{P(B=Heads \mid A=fair) \times P(A=fair)}{P(B=Heads)} $$

**Knowns:**
- P(A = fair) = 0.99
- P(B = Heads | A = fair) = 0.5
- P(B = Heads | A = unfair) = 1 (unfair coin is always Heads)

**Compute the evidence P(B = Heads)** using total probability:
$$ P(B=Heads) = P(B|A)P(A) + P(B|\bar{A})P(\bar{A}) = (0.5 \times 0.99) + (1 \times 0.01) = 0.5050 $$

**Posterior:**
$$ P(A=fair \mid B=Heads) = \frac{0.5 \times 0.99}{0.5050} = \mathbf{0.9802} $$

So after one Heads, belief that the coin is fair drops slightly: 0.99 → **0.9802**.

### 2nd Flip → Heads again

> **Key idea: the posterior from the previous step becomes the new prior!**

**New prior:** P(A = fair) = 0.9802  → so P(A = unfair) = 1 − 0.9802 = 0.0198

**New evidence:**
$$ P(B=H) = P(B|A)P(A) + P(B|\bar A)P(\bar A) = (0.5 \times 0.9802) + (1 \times 0.0198) = 0.5099 $$

**New posterior:**
$$ P(A=fair \mid B=H) = \frac{0.5 \times 0.9802}{0.5099} = \mathbf{0.9612} $$

**Pattern:** Each consecutive Heads makes us **less confident** the coin is fair (0.99 → 0.9802 → 0.9612 → …). This is **Bayesian updating** — belief is revised step by step as evidence accumulates.

---

## 2.15 The Famous Bayesian Quote (worth remembering)

> *"A Bayesian is one who, vaguely expecting a horse, and catching a glimpse of a donkey, strongly believes he has seen a mule."*

Meaning: a Bayesian **combines prior belief (horse) with new evidence (donkey)** to reach a compromise conclusion (mule) — illustrating how the prior pulls the posterior.

---

## 2.16 References (from slides)
- *Bayesian statistics (a very brief introduction)* — Ken Rice
- statisticshowto.com/bayes-theorem-problems/
- Slides "Bayesian inference and generative models" — K.E. Stephan
- Intro slides to probabilistic & unsupervised learning — M. Sahani
- Animation: blog.stata.com/2016/11/01/introduction-to-bayesian-statistics-part-1-the-basic-concepts/

---
---

# 🎯 Quick Revision Cheat-Sheet

### Core Formulas

| Concept | Formula |
|---|---|
| Classical probability | $P(E) = n_e / N$ |
| Joint | $P(A,B)$ |
| Union | $P(A\cup B) = P(A)+P(B)-P(A,B)$ |
| Complement | $P(\sim A) = 1 - P(A)$ |
| Marginal | $P(X=x) = \sum_y P(X=x, Y=y)$ |
| Conditional | $P(A\mid B) = \dfrac{P(A,B)}{P(B)}$ |
| Product rule | $P(A\cap B) = P(B\mid A)\,P(A)$ |
| Bayes' theorem | $P(A\mid B) = \dfrac{P(B\mid A)\,P(A)}{P(B)}$ |
| Total probability | $P(B) = P(B\mid A)P(A) + P(B\mid \neg A)P(\neg A)$ |
| Bayesian rule | Posterior ∝ Likelihood × Prior |

### Key Distinctions to Memorize
- **Exploration / Inference / Prediction** → Visualization / Randomization / Machine Learning
- **DS vs ML** → DS explores & acts; ML proves & publishes
- **DS vs Data Engineering** → DS = unbounded/exploratory; DE = bounded/linear/programming
- **Frequentist vs Bayesian** → Freq: params fixed, data random, point estimate. Bayes: params random, data fixed, distribution.
- **PMF vs PDF** → discrete (sum = 1) vs continuous (area = probability, point prob = 0)
- **Base-rate effect** → a very accurate test on a rare condition still gives a low posterior (Examples in 2.8 & 2.11).

### Practice Questions to Test Yourself
1. State the three pillars of data science and why each is central.
2. Define experiment, trial, event, sample space.
3. A die is rolled. P(odd number)? *(Ans: 3/6 = 0.5)*
4. Using the disease/symptom table, compute P(X=1 | Y=0). *(Ans: 0.1/(0.5+0.1) = 0.167)*
5. Redo the diagnostic-test Bayes example with disease rate 1% instead of 0.5% — does the posterior go up or down? *(Up — higher base rate → higher posterior.)*
6. Explain why the coin's "fair" probability keeps dropping with each Heads.
7. State Bayes' theorem and label prior, likelihood, evidence, posterior.
8. Difference between frequentist and Bayesian treatment of parameters θ?

---

*End of Chapters 1 & 2. Chapters 3 & 4 will go in a separate file.*
