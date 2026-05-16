# Foundation of Data Science (CS G320) — Study Guide

### Chapter 3: Probability Distributions

*(Continuation of Chapters 1 & 2. Read alongside `FDS_Chapter_1_and_2_Study_Guide.md`.)*

---

# 📙 CHAPTER 3 — Probability Distributions

## 🧭 3.0 Read This First — The Whole Chapter in Plain Words

Before any formula, here is the **entire story** of Chapter 3 in everyday language. Come back to this box whenever you feel lost.

> **The one question this chapter answers:** *"I have some data. What kind of randomness produced it, and how do I describe that randomness with a formula?"*

A **probability distribution** is just a **rule that says how likely each possible outcome is**. That's it. Every distribution below is the *same idea* dressed for a different situation:

| If your situation is… | You use… | Real-life example |
|---|---|---|
| One yes/no trial | **Bernoulli** | Will this one coin land heads? |
| Many yes/no trials, count the yeses | **Binomial** | How many heads in 8 tosses? |
| One trial, but more than 2 outcomes | **Multinomial** | Which face does a die show? |
| A measured number that clusters around an average | **Gaussian (bell curve)** | People's heights, IQ scores |
| Comparing values from different scales | **Standard Normal (z)** | Is a 130 IQ "more extreme" than a 6 ft height? |
| You want to express a *belief* about a probability | **Beta / Dirichlet** | "I think the coin is probably fair, but I'm not sure" |
| You only know min, max, most-likely | **Triangular** | Estimating commute time with no data |

**The single learning recipe — use it for every distribution in this chapter:**

```
   ① WHEN do I use it?   (the situation in plain words)
            ↓
   ② WHAT is the formula? (and what each symbol means)
            ↓
   ③ HOW do I compute?    (a fully worked number example)
            ↓
   ④ HOW spread out?      (mean & variance — the "centre" and "width")
```

> **Mantra:** *A distribution = a formula + a few knobs (parameters). Learn the situation first, the formula second.*

---

## 🗺️ 3.0b Roadmap — How the Distributions Connect

These aren't 7 random formulas. They're **one family tree**. Each builds on the previous:

```
        ONE yes/no trial
        ┌──────────────┐
        │  BERNOULLI    │  "one coin toss"
        └──────┬────────┘
               │  repeat n times, count successes
               ▼
        ┌──────────────┐
        │   BINOMIAL    │  "8 coin tosses, how many heads?"
        └──────┬────────┘
               │  allow K outcomes instead of 2
               ▼
        ┌──────────────┐
        │ MULTINOMIAL   │  "roll a die (6 outcomes)"
        └──────┬────────┘
               │  let the value be a continuous measurement
               ▼
        ┌──────────────┐
        │  GAUSSIAN     │  "heights — the bell curve"
        └──────┬────────┘
               │  recenter to mean 0, width 1
               ▼
        ┌──────────────┐
        │ STANDARD NORM │  "z-scores, one table for everything"
        └──────────────┘

   Separately, the "BELIEF" distributions (priors):
   BETA  → expresses belief about a Bernoulli/Binomial probability
   DIRICHLET → the same idea for Multinomial (Beta's big sibling)
   TRIANGULAR → rough model when you only know min / mode / max
```

> **You are here →** we walk this tree top to bottom. Every section says where you are on it.

---

## 3.1 The Exponential Family (the "club" all these belong to)

**① In plain words:** Most of the distributions in this chapter — Bernoulli, binomial, multinomial, Poisson, Gaussian, Dirichlet — secretly share the *same mathematical skeleton*. We call that shared club the **exponential family**.

> A **large family of useful distributions that share common properties.**

**Why you should care (not just trivia):** because they share a skeleton, any clever trick proved for the *family* (like "there's always a tidy summary statistic" or "there's always a matching conjugate prior") automatically works for *every member*. Learn the family, get the members for free.

Key facts:
- It is **not** a single distribution — it is a *family* (a "club" of distributions).
- The variable involved can be **discrete *or* continuous**.
- Shared form ⇒ shared results: sufficient statistics, conjugate priors, etc. apply to **all** members.

> **Memory hook:** "Exponential **family**" = a *club*. Bernoulli, Binomial, Multinomial, Poisson, Gaussian are all *members of the club*.

---

## 3.2 Three Ways to Find a Distribution's Knobs (parameters)

**① In plain words:** A distribution has knobs ($\theta$) — e.g. the coin's bias $\mu$. Given data, *how do we set the knobs?* Three philosophies (this directly extends the Frequentist-vs-Bayesian split from Ch 2):

| Method | Idea in one line | Uses |
|---|---|---|
| **Density Estimation** | "Just look at the data and guess the shape it came from." | the raw data only |
| **Frequentist (MLE)** | "Pick the knob settings that make the data we saw **most likely**." | likelihood only |
| **Bayesian (MAP / posterior)** | "Start with a **belief** about the knobs, then let the data update it." | prior **+** likelihood |

> **Connects to Ch 2:** Frequentist = a single best guess (point estimate) of $\theta$. Bayesian = a *whole distribution* over $\theta$. A **conjugate prior** is a prior whose **posterior comes out in the same form** as the prior — that's *why* Beta pairs with Bernoulli, and Dirichlet with Multinomial (we'll cash this out in §3.23–3.25).

---

# ── PART A: DISCRETE DISTRIBUTIONS ──

> **You are here:** top of the family tree — yes/no and counting situations.

## 3.3 Bernoulli Distribution — *One* Yes/No Trial

**① WHEN:** You have a **single trial** with exactly two outcomes — *one* coin toss, *one* click/no-click, *one* pass/fail. Nothing repeated yet.

**② FORMULA:** Let $x \in \{0, 1\}$, and let the one knob be $\mu = P(x = 1)$ (probability of "success").

The naive way to say it:

$$ P(x = 1) = \mu, \qquad P(x = 0) = 1 - \mu $$

The compact one-line trick (the **Bernoulli distribution**) — one formula that gives *both* cases:

$$ \boxed{\;\mathrm{Bern}(x \mid \mu) = \mu^{x}\,(1-\mu)^{\,1-x}\;} $$

*Why it works (sanity check):* the exponents act as on/off switches. Put $x=1 \Rightarrow \mu^1(1-\mu)^0 = \mu$. Put $x=0 \Rightarrow \mu^0(1-\mu)^1 = 1-\mu$. ✓ Both cases fall out of the one formula.

**③ HOW (estimate the knob — MLE):** Given $N$ independent trials with $m$ successes (so $\sum x_i = m$), the most-likely value of $\mu$ is just the **observed success rate**:

$$ \mu_{\text{MLE}} = \frac{m}{N} = \frac{\text{number of observations of } x = 1}{N} $$

> **Worked number:** $N = 5$ tosses, $m = 3$ heads → $\mu_{\text{MLE}} = 3/5 = 0.6$. (Exactly what your gut says — the formula just confirms intuition.)

**④ SPREAD (mean & variance):** $\mathbb{E}[x] = \mu$, $\;\mathrm{Var}[x] = \mu(1-\mu)$. (Variance is biggest at $\mu = 0.5$ — a fair coin is the most "unpredictable".)

> **Caution (the reason Bayesian methods exist):** MLE can **overfit** tiny samples. See 3 heads in 3 tosses? MLE screams $\mu = 1$ — "this coin *never* lands tails." Obviously silly from 3 flips. The fix: add a **prior belief** → that's the Beta distribution (§3.23).

---

## 3.4 Multinomial / Categorical — One Trial, *K* Outcomes

**① WHEN:** Same as Bernoulli but the trial has **more than two** outcomes — rolling a die ($K=6$), picking a category, classifying into one of $K$ labels. *Bernoulli is just the $K=2$ case.*

**② FORMULA:** Encode the outcome as a **one-hot vector** $x = (x_1, \dots, x_K)$ — exactly one entry is 1 (the outcome that happened), the rest are 0. Knobs $\mu = (\mu_1,\dots,\mu_K)$ with $\mu_k = P(x_k=1)$ and $\sum_k \mu_k = 1$.

$$ p(x \mid \mu) = \prod_{k=1}^{K} \mu_k^{\,x_k} $$

*(Same on/off-switch trick as Bernoulli: only the $\mu_k$ for the outcome that occurred survives, the rest become $\mu^0 = 1$.)*

For $N$ independent observations the **likelihood** multiplies them all:

$$ p(\mathcal{D}\mid\mu) = \prod_{n=1}^{N}\prod_{k=1}^{K}\mu_k^{\,x_{nk}} = \prod_{k=1}^{K}\mu_k^{\left(\sum_n x_{nk}\right)} = \prod_{k=1}^{K}\mu_k^{\,m_k} $$

where $m_k = \sum_n x_{nk}$ = how many times outcome $k$ happened.

**③ HOW (MLE):** again just the observed fractions:

$$ \mu_k^{\text{MLE}} = \frac{m_k}{N} \quad(\text{fraction of observations in state } k) $$

> **Memory hook:** Bernoulli is the **K = 2** special case of the multinomial (rolling a die = $K=6$).

---

## 3.5 Binomial Distribution — *n* Yes/No Trials, Count the Successes

**① WHEN:** You repeat a Bernoulli trial **$n$ independent times** and ask *"how many successes total?"* — e.g. "8 coin tosses, what's the chance of exactly 5 heads?" This is the workhorse of the chapter; most worked examples below use it.

**② FORMULA:**

$$ \boxed{\;\mathrm{Bin}(m \mid n, \mu) = \binom{n}{m}\,\mu^{m}\,(1-\mu)^{\,n-m}\;} $$

**Read it as three pieces — this is the key intuition:**
- $\mu^{m}$ → probability the $m$ successes happen,
- $(1-\mu)^{n-m}$ → probability the other $n-m$ failures happen,
- $\binom{n}{m}$ → **how many different orders** those successes could appear in.

The **binomial coefficient** (the "number of orderings"):

$$ \binom{n}{m} = \frac{n!}{m!\,(n-m)!} $$

| Symbol | Meaning |
|---|---|
| $n$ | number of trials |
| $m$ | number of successes |
| $\mu$ | probability of success on one trial |
| $\binom{n}{m}$ | number of ways to arrange $m$ successes among $n$ trials |

**④ SPREAD:** $\mathbb{E}[m] = n\mu$, $\;\mathrm{Var}[m] = n\mu(1-\mu)$. *(Makes sense: 8 tosses of a fair coin → expect $8\times0.5 = 4$ heads.)*

```
Bin(m | n=10, μ=0.25)   ← typical right-skewed bar shape
P |        ▁ █ █
  |      ▁ █ █ █ ▆
  |    ▁ █ █ █ █ █ ▃
  |____▁_█_█_█_█_█_█_▃_▁_▁_▁__ m
      0 1 2 3 4 5 6 7 8 9 10
```

> **Memory hook:** Bernoulli = **1** trial. Binomial = **n** Bernoulli trials, count the successes. $\binom{n}{m}$ counts the *orderings*; $\mu^m(1-\mu)^{n-m}$ is the probability of *one* such ordering.

---

# 🧮 PART A — WORKED EXAMPLES (Bernoulli & Binomial)

> **You are here:** still on discrete distributions — now we *practise* the formulas above. Each example tags which formula it uses.

## 3.6 First: What Makes Something a "Bernoulli Trial"?

**① In plain words:** Before you can use Binomial, you must check the situation actually qualifies. The three-part checklist:

A process is a sequence of **Bernoulli trials** when:
- The **n trials are independent**.
- Each trial has **exactly two outcomes**: success or failure.
- The **probability of success $p$ is constant** — does **not** drift trial to trial.

Coin tosses: Heads/Tails, independent, $p$ fixed → ✓ Bernoulli trials.

---

## 3.7 Worked Example — With vs Without Replacement *(the classic trap)*

**Setup:** Eight balls drawn from a bag of **10 white + 8 black**. Predict whether each is black.

**(a) With replacement** — Bernoulli trial?

> **Yes.** $P(\text{black}) = \dfrac{8}{18} = \dfrac{4}{9}$, **constant** every draw (ball goes back), draws independent → **Bernoulli trials**. ✓

**(b) Without replacement** — Bernoulli trial?

> **No.** $P(\text{black})$ **changes** after each draw (first $8/18$, then it depends on what was removed), and draws are **dependent** → **NOT Bernoulli**. ✗

> **Key rule (memorize — examiners love this):** *With replacement → Bernoulli (p constant, independent). Without replacement → NOT Bernoulli (p changes, dependent).*

---

## 3.8 Worked Example — Biased Coin, 8 Tosses *(uses Binomial §3.5)*

**Problem:** $P(\text{head}) = \tfrac{2}{3}$, tossed **8 times**. Find **(a) at least 7 heads**, **(b) at most 7 heads**.

**Plug into the formula:** $n = 8$, $\mu = \tfrac{2}{3}$, $1-\mu = \tfrac{1}{3}$, so $\mathrm{Bin}(m\mid 8,\tfrac23)=\binom{8}{m}\left(\tfrac23\right)^m\left(\tfrac13\right)^{8-m}$.

**(a) "At least 7" = P(m=7) + P(m=8)** — there's no shortcut, just two terms:

$$ P(7) = \binom{8}{7}\left(\tfrac{2}{3}\right)^{7}\left(\tfrac{1}{3}\right)^{1} = 8\cdot\frac{128}{2187}\cdot\frac{1}{3} = \frac{1024}{6561} $$

$$ P(8) = \binom{8}{8}\left(\tfrac{2}{3}\right)^{8}\left(\tfrac{1}{3}\right)^{0} = 1\cdot\frac{256}{6561} = \frac{256}{6561} $$

$$ P(m \ge 7) = \frac{1024}{6561} + \frac{256}{6561} = \frac{1280}{6561} \approx \mathbf{0.195} $$

**(b) "At most 7" = 1 − P(m=8)** — here the **complement** saves you summing 8 terms:

$$ P(m \le 7) = 1 - P(8) = 1 - \frac{256}{6561} = \frac{6305}{6561} \approx \mathbf{0.961} $$

> **Trick:** "at most 7 out of 8" = everything except "all 8" → use the **complement**, don't sum 8 terms.

---

## 3.9 Worked Example — Three Rolls of a Die, Count the Sixes *(Binomial)*

**Problem:** Distribution of getting a **six** in **three** rolls of a fair die.

**Set the knobs:** Success = "roll a 6" → $\mu = \tfrac{1}{6}$, $1-\mu = \tfrac{5}{6}$, $n = 3$.

$$ P(m) = \binom{3}{m}\left(\tfrac{1}{6}\right)^{m}\left(\tfrac{5}{6}\right)^{3-m} $$

Build the whole table by sweeping $m = 0,1,2,3$:

| $m$ (number of sixes) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $\binom{3}{m}$ | 1 | 3 | 3 | 1 |
| $P(m)$ | $\frac{125}{216}$ | $\frac{75}{216}$ | $\frac{15}{216}$ | $\frac{1}{216}$ |
| decimal | 0.5787 | 0.3472 | 0.0694 | 0.0046 |

**Always sanity-check:** $125+75+15+1 = 216$ ✓ (probabilities sum to 1).

---

## 3.10 Worked Example — Doublets in 4 Throws of Two Dice *(Binomial)*

**Problem:** Distribution of the number of **doublets** (both dice equal) in **4 throws** of a pair.

**Set the knobs:** doublet = {(1,1)…(6,6)} = 6 of 36 → $\mu = \tfrac{6}{36} = \tfrac{1}{6}$, $n = 4$. *(Same $\mu$ as §3.9, just $n=4$ now — notice the pattern.)*

$$ P(m) = \binom{4}{m}\left(\tfrac{1}{6}\right)^{m}\left(\tfrac{5}{6}\right)^{4-m} $$

| $m$ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $P(m)$ | $\frac{625}{1296}$ | $\frac{500}{1296}$ | $\frac{150}{1296}$ | $\frac{20}{1296}$ | $\frac{1}{1296}$ |
| decimal | 0.4823 | 0.3858 | 0.1157 | 0.0154 | 0.0008 |

Check: $625+500+150+20+1 = 1296 = 6^4$ ✓.

---

## 3.11 Worked Example — At Least 5 Heads in 8 Tosses *(Binomial, fair coin)*

**Problem:** P(**at least 5 heads**) in **8 fair tosses**.

**Set the knobs:** $n = 8$, $\mu = \tfrac{1}{2}$. Fair coin simplifies things — every term shares $\left(\tfrac12\right)^8 = \tfrac{1}{256}$, so only the $\binom{8}{m}$ counts vary: $\mathrm{Bin}(m\mid 8,\tfrac12) = \dfrac{\binom{8}{m}}{256}$.

$$ P(m \ge 5) = \frac{\binom{8}{5}+\binom{8}{6}+\binom{8}{7}+\binom{8}{8}}{256} = \frac{56+28+8+1}{256} = \frac{93}{256} \approx \mathbf{0.363} $$

---

## 3.12 Worked Example — Telephone Lines Busy *(Binomial, exactly m)*

**Problem:** 1 in 10 phones is busy. Pick **6** at random. P(**exactly 4 busy**)?

**Set the knobs:** Success = "busy" → $\mu = 0.1$, $1-\mu = 0.9$, $n = 6$, want $m = 4$.

$$ P(4) = \binom{6}{4}(0.1)^{4}(0.9)^{2} = 15 \times 0.0001 \times 0.81 = \mathbf{0.001215} $$

So ≈ **0.12 %** chance exactly four of six are busy. *(Tiny — because "busy" is rare and we're asking for a lot of them.)*

---

## 3.13 Worked Example — Green & Red Balls, With Replacement *(Binomial)*

**Problem:** Bag = **5 green + 1 red**. Draw **3 with replacement**. Distribution of #green.

**Set the knobs:** with replacement → binomial OK. Success = "green" → $\mu = \tfrac{5}{6}$, $1-\mu = \tfrac{1}{6}$, $n = 3$.

$$ P(m) = \binom{3}{m}\left(\tfrac{5}{6}\right)^{m}\left(\tfrac{1}{6}\right)^{3-m} $$

| $m$ green | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $P(m)$ | $\frac{1}{216}$ | $\frac{15}{216}$ | $\frac{75}{216}$ | $\frac{125}{216}$ |
| decimal | 0.0046 | 0.0694 | 0.3472 | 0.5787 |

> **Notice:** this is the **mirror image** of the §3.9 die table — same numbers, flipped, because here success (green) is the *likely* outcome instead of the rare one. Same formula, opposite skew.

---

## 3.14 Standard Deviation and Variance *(the "how spread out" toolkit)*

**① In plain words:** Mean tells you the *centre*. **Variance/SD tell you the width** — how far data typically sits from that centre. You need this for every distribution from here on.

| Quantity | **Population** (you have everyone) | **Sample** (you have a subset) |
|---|---|---|
| **Variance** | $\sigma^{2} = \dfrac{\sum (x_i - \mu)^{2}}{N}$ | $s^{2} = \dfrac{\sum (x_i - \bar{x})^{2}}{n-1}$ |
| **Standard deviation** | $\sigma = \sqrt{\dfrac{\sum (x_i - \mu)^{2}}{N}}$ | $s = \sqrt{\dfrac{\sum (x_i - \bar{x})^{2}}{n-1}}$ |

- **Variance** = average *squared* distance from the mean (squared so + and − don't cancel).
- **SD** = $\sqrt{\text{variance}}$ → back in the data's own units (easier to read).
- Sample versions divide by $n-1$ (**Bessel's correction**) — a small fix so a subset doesn't *underestimate* the true spread.

### Worked Example — variance & SD of a die

**Problem:** Fair die rolled; find variance and SD of the outcome.

Sample space $\{1,2,3,4,5,6\}$, each prob $\tfrac16$.

$$ \mu = \mathbb{E}[X] = \frac{1+2+3+4+5+6}{6} = 3.5 $$

$$ \mathbb{E}[X^2] = \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.17 $$

$$ \sigma^{2} = \mathbb{E}[X^2] - \mu^{2} = 15.17 - 12.25 = \mathbf{2.92} $$

$$ \sigma = \sqrt{2.92} \approx \mathbf{1.71} $$

> **Memory hook (the shortcut formula):** $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ — *"mean of the squares minus the square of the mean."* Almost always faster than the definition.

---

# ── PART B: CONTINUOUS DISTRIBUTIONS ──

> **You are here:** we leave counting behind. The variable is now a **measured number** (height, time, score) that can take any value on a range.

## 3.15 Gaussian (Normal) Distribution — *the* Bell Curve

**① WHEN:** Your data is a continuous measurement that **clusters around an average**, with fewer values far away — heights, weights, exam scores, measurement errors. The single most important continuous distribution.

**② FORMULA:**

$$ \boxed{\;\mathcal{N}(x \mid \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi\sigma^{2}}}\,\exp\!\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)\;} $$

**Don't be scared of it — read it as a story:** the $(x-\mu)^2$ part measures *how far x is from the centre*; the minus sign and $\exp$ turn "far away" into "very unlikely". The fraction out front is just a scaler so the total area = 1. Only **two knobs**:

| Parameter | Name | Role |
|---|---|---|
| $\mu$ | **mean** | where the peak sits (centre of the bell) |
| $\sigma^{2}$ | **variance** ($\sigma$ = SD) | how wide/flat the bell is |

```
        .-''''-.        ← peak at x = μ
      .'   |    '.
     /     |      \
    /      |       \      width controlled by σ
___/_______|________\___ x
   μ-2σ    μ        μ+2σ
```

**④ The rule worth memorizing — Empirical (68–95–99.7) rule:** for *any* bell curve,

| Within | Coverage |
|---|---|
| $\mu \pm 1\sigma$ | ≈ **68 %** |
| $\mu \pm 2\sigma$ | ≈ **95 %** |
| $\mu \pm 3\sigma$ | ≈ **99.7 %** |

> *Plain reading:* almost everything (99.7 %) is within 3 SDs of the mean. Anything beyond is genuinely rare.

---

## 3.16 Expectation of a Function — "Probability-Weighted Average"

**① In plain words:** The **expectation** $\mathbb{E}[f]$ is just a *weighted average* of $f(x)$, where each value is weighted by how likely it is. The "average outcome you'd get in the long run."

**Discrete** (sum over outcomes):

$$ \mathbb{E}[f] = \sum_{x} p(x)\,f(x) $$

**Continuous** (integrate instead of sum — same idea, smooth):

$$ \mathbb{E}[f] = \int p(x)\,f(x)\,dx $$

If you only have $N$ data points, approximate it by the plain sample average:

$$ \mathbb{E}[f] \approx \frac{1}{N}\sum_{n=1}^{N} f(x_n) $$

> **One idea, two notations:** discrete → $\sum$, continuous → $\int$. Both mean "average, weighted by probability."

---

## 3.17 Expectation under the Gaussian (why μ and σ² *are* mean and variance)

**① In plain words:** We *claimed* in §3.15 that $\mu$ is the mean and $\sigma^2$ the variance. Here's the proof — plug the Gaussian into the §3.16 definition:

$$ \mathbb{E}[x] = \int \mathcal{N}(x\mid\mu,\sigma^2)\,x\,dx = \mu $$

$$ \mathbb{E}[x^{2}] = \int \mathcal{N}(x\mid\mu,\sigma^2)\,x^{2}\,dx = \mu^{2} + \sigma^{2} $$

Apply the §3.14 shortcut $\mathrm{Var} = \mathbb{E}[x^2]-\mathbb{E}[x]^2$:

$$ \mathrm{Var}[x] = (\mu^{2}+\sigma^{2}) - \mu^{2} = \sigma^{2} $$

> Confirms: for the Gaussian, the knob $\mu$ **is** the mean and $\sigma^2$ **is** the variance — they were named honestly.

---

## 3.18 Gaussian in Many Dimensions (multivariate)

**① WHEN:** Same bell-curve idea but $x$ is a **vector** (e.g. height *and* weight together). The single number $\sigma^2$ becomes a **covariance matrix** $\Sigma$ that also captures how dimensions move together.

$$ \mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{D/2}}\,\frac{1}{|\Sigma|^{1/2}}\,\exp\!\left(-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right) $$

| Symbol | Meaning |
|---|---|
| $\mu$ | $D$-dimensional **mean vector** |
| $\Sigma$ | $D \times D$ **covariance matrix** (spread + correlations) |
| $|\Sigma|$ | determinant of $\Sigma$ |
| $\Sigma^{-1}$ | inverse (precision) matrix |

> Don't memorize the matrix algebra — just know: the scalar Gaussian (§3.15) is the special case $D = 1$, $\Sigma = \sigma^2$. Same shape, more dimensions.

---

## 3.19 Standard Normal & z-score — *One Ruler for Everything*

**① WHEN:** You want to compare values from **different scales** (Is a 130 IQ more unusual than a 200 cm height?), or use a single lookup table for any bell curve. Trick: re-express every value as *"how many SDs from the mean am I?"*

**② FORMULA — the z-score:**

$$ \boxed{\; z = \frac{x - \mu}{\sigma} \;} $$

| z-score | Plain meaning |
|---|---|
| $z > 0$ | above the mean (by $z$ SDs) |
| $z < 0$ | below the mean |
| $z = 0$ | exactly the mean |

The **standard normal** is just a Gaussian with mean 0, SD 1 — its PDF:

$$ \varphi(z) = \frac{1}{\sqrt{2\pi}}\,e^{-z^{2}/2} $$

> **Why bother?** Convert *anything* to a z-score → use **one** z-table → done. Heights, IQs, test scores all become comparable.

**Finding a probability between two values:** convert both endpoints to z, then read the area between them off the z-table.

```
 φ(z)        .-''-.
           .' ████ '.        ████ = P(a ≤ X ≤ b)
          /  ██████  \             = Φ(z_b) − Φ(z_a)
_________/___██████___\________ z
        z_a         z_b
```

---

## 3.20 Worked Example — z-scores from a Small Data Set *(uses §3.14 + §3.19)*

**Problem:** Data **26, 33, 60, 28, 34, 33, 25, 44, 50, 36, 26, 37, 43, 62, 35, 38, 45, 32, 26, 34** — find mean & SD, then express some values as z-scores.

**Step 1 — mean.** Sum $= 760$, $N = 20$:

$$ \mu = \frac{760}{20} = 38 $$

**Step 2 — SD** (population form, §3.14): the spread works out to

$$ \sigma \approx \mathbf{11.2} $$

**Step 3 — convert with $z = \dfrac{x-\mu}{\sigma}$** ($\mu=38$, $\sigma=11.2$):

| Original value | Calculation | Standard score (z) |
|---|---|---|
| 26 | $(26-38)/11.2$ | $-1.07$ |
| 33 | $(33-38)/11.2$ | $-0.45$ |
| 60 | $(60-38)/11.2$ | $+1.96$ |
| 62 | $(62-38)/11.2$ | $+2.14$ |

> **Reading it:** 60 sits ≈ **2 SDs above** the mean (unusually high); 26 ≈ **1 SD below**.

---

## 3.21 Worked Example — IQ Above 130 *(value → z → probability)*

**Problem:** IQ ~ Normal, $\mu = 100$, $\sigma = 15$. Fraction with IQ **above 130**?

**Step 1 — convert to z:**

$$ z = \frac{130 - 100}{15} = \frac{30}{15} = 2.0 $$

**Step 2 — z-table:** $P(Z \le 2.0) = 0.9772$. We want *above*, so take the complement:

$$ P(X > 130) = 1 - 0.9772 = \mathbf{0.0228} \approx 2.28\% $$

> Matches the 68–95–99.7 rule: beyond $+2\sigma$ ≈ 2.5 % in one tail. ✓

---

## 3.22 Worked Example — Heights & the Shortest 10 % *(both directions of z)*

**Problem:** Women's heights ~ Normal, $\mu = 65$ in, $\sigma = 3.5$ in. (a) P(shorter than 60 in)? (b) Height cutting off the **shortest 10 %**?

**(a) value → z → probability** (forward direction):

$z = \dfrac{60 - 65}{3.5} = \dfrac{-5}{3.5} \approx -1.43$. From the z-table, $P(Z < -1.43) \approx \mathbf{0.0764}$ (≈ 7.6 %).

**(b) probability → z → value** (backward direction): the 10th percentile of the standard normal is $z \approx -1.28$. Convert back with $x = \mu + z\sigma$:

$$ x = 65 + (-1.28)(3.5) \approx 65 - 4.48 = \mathbf{60.5 \text{ in}} $$

So the shortest 10 % are below ≈ 60.5 in.

> **The two z-problem directions (know both):** (i) value → z → probability (z-table *forward*); (ii) probability → z → value via $x = \mu + z\sigma$ (z-table *backward*).

---

# ── PART C: "BELIEF" DISTRIBUTIONS (Bayesian Priors) ──

> **You are here:** these don't model data directly — they model *your belief about a probability*. This is where Ch 2's Bayesian thread comes back.

## 3.23 Beta Distribution — A Belief About a Probability

**① WHEN:** In Bayesian work you need a prior for the Bernoulli/binomial knob $\mu$. The Beta is the natural choice — it lives on $[0,1]$ (exactly where a probability lives) and is the **conjugate prior** (posterior comes out Beta again — clean updating).

**② FORMULA:**

$$ \boxed{\;\mathrm{Beta}(\mu \mid a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\,\Gamma(b)}\;\mu^{\,a-1}\,(1-\mu)^{\,b-1}\;} $$

**Read it intuitively:**
- Defined on $\mu \in [0,1]$ — perfect for a *probability*.
- Think of $a, b$ as **pseudo-counts**: $a$ = imagined prior successes, $b$ = imagined prior failures.
- $\Gamma(\cdot)$ = **gamma function** = factorial generalised ($\Gamma(n) = (n-1)!$). The whole front fraction is just a **normaliser** so the area = 1 — *don't memorize it, understand its job.*

**④ SPREAD:**

$$ \mathbb{E}[\mu] = \frac{a}{a+b}, \qquad \mathrm{Var}[\mu] = \frac{ab}{(a+b)^{2}(a+b+1)} $$

*(Mean = fraction of pseudo-successes — exactly the intuition.)*

```
Shapes of Beta(a,b):
 a=b=1  → flat (uniform, "no opinion") ▁▁▁▁▁▁▁▁
 a=b=0.5→ U-shaped                     █▁    ▁█
 a=b>1  → bell, peak at 0.5             ▁▃█▃▁
 a>b    → skewed toward 1                 ▁▃▆█
 a<b    → skewed toward 0              █▆▃▁
```

> **Conjugate magic (the payoff, ties to Ch 2):** Prior $\mathrm{Beta}(a,b)$ + binomial data with $m$ successes & $l$ failures → Posterior $\mathrm{Beta}(a+m,\;b+l)$. **Same family in, same family out.** Updating belief = just *add your counts to a and b.* That's the entire point of "conjugate prior."

---

## 3.24 Dirichlet Distribution — Beta's Multi-Outcome Sibling

**① WHEN:** Exactly the Beta idea, but the prior is for a **multinomial** (K outcomes) instead of binary. Dirichlet : Multinomial :: Beta : Bernoulli.

$$ \mathrm{Dir}(\mu \mid \alpha) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)}\,\prod_{k=1}^{K}\mu_k^{\,\alpha_k - 1}, \qquad \alpha_0 = \sum_{k=1}^{K}\alpha_k $$

- $\mu = (\mu_1,\dots,\mu_K)$, $\mu_k \ge 0$, $\sum_k \mu_k = 1$ (a probability *vector*).
- $\alpha = (\alpha_1,\dots,\alpha_K)$ = **concentration hyperparameters** = prior pseudo-counts per category (same role as Beta's $a, b$).

> **Memory hook:** Beta : Bernoulli/Binomial :: Dirichlet : Multinomial. ($K = 2$ Dirichlet **is** literally a Beta.)

---

## 3.25 Full Bayesian Example — Probability a Coin Lands Heads

**① The story:** unknown coin bias $\mu = P(\text{Heads})$. We only see flips. Walk the Bayesian loop end-to-end (this is §2.14's coin example, now with a *continuous* parameter):

1. **Prior (before data):** no opinion → $\mu$ uniform on $[0,1]$, i.e. $\mathrm{Beta}(a{=}1, b{=}1)$ (flat).

2. **Data:** flip the coin. Say $a' = 1$ head and $b' = 1$ tail in 2 flips.

3. **Posterior (after data):** by conjugacy, with $h$ heads and $t$ tails,

$$ \mathrm{Beta}\big(\mu \mid a + h,\; b + t\big) = \frac{\Gamma(a+h+b+t)}{\Gamma(a+h)\,\Gamma(b+t)}\;\mu^{\,a+h-1}(1-\mu)^{\,b+t-1} $$

> **What actually happened:** the posterior is **Beta again** — its peak slides toward the observed head-fraction, and it gets **narrower (more certain)** with every extra flip. This is **Bayesian updating** from Ch 2 (§2.13–2.14), but now over a *continuous* $\mu$ instead of two discrete hypotheses. Same idea, smoother.

---

## 3.26 Triangular Distribution — When You Have *No Data*, Only Estimates

**① WHEN:** You have no dataset — just an expert's guess of **minimum $a$**, **maximum $b$**, and **most-likely (mode) $c$**. Common in project-time / commute estimation. Shape = a triangle.

**② FORMULA** (piecewise — rising slope then falling slope):

$$ f(x) = \begin{cases} \dfrac{2(x-a)}{(b-a)(c-a)}, & a \le x \le c \\[2mm] \dfrac{2(b-x)}{(b-a)(b-c)}, & c < x \le b \\[1mm] 0, & \text{otherwise} \end{cases} $$

```
 f(x)
   |        /\
   |       /  \
   |      /    \          area under triangle = 1
   |     /      \
   |____/________\____ x
       a    c     b
```

**④ SPREAD:** $\mathbb{E}[X] = \dfrac{a+b+c}{3}$ (just the average of the three numbers), and $\mathrm{Var}[X] = \dfrac{a^2+b^2+c^2-ab-ac-bc}{18}$.

> **Use:** when you only know **min, max, and most likely** — no data collection needed. The pragmatic estimator's distribution.

---

## 3.27 Worked Example — Commute Time (Triangular)

**Problem:** Commute (hours) $X \sim$ Triangular with $a = 0$, $b = 2$, $c = 1$ (a symmetric triangle). Find **(a)** the PDF, **(b)** $P(X > 1.5)$, **(c)** $\mathbb{E}[X]$.

**(a) PDF** — substitute $b-a=2$, $c-a=1$, $b-c=1$ into the §3.26 formula:

$$ f(x) = \begin{cases} \dfrac{2(x-0)}{2\cdot 1} = x, & 0 \le x \le 1 \\[2mm] \dfrac{2(2-x)}{2\cdot 1} = 2 - x, & 1 < x \le 2 \\[1mm] 0, & \text{otherwise} \end{cases} $$

**(b) P(X > 1.5)** — integrate the falling piece from 1.5 to 2:

$$ P(X > 1.5) = \int_{1.5}^{2}(2-x)\,dx = \left[2x - \tfrac{x^2}{2}\right]_{1.5}^{2} = (4-2) - (3 - 1.125) = 2 - 1.875 = \mathbf{0.125} $$

> **Shortcut check (no calculus):** it's just a little triangle of base 0.5 and height 0.5 → area $=\tfrac12(0.5)(0.5)=0.125$. ✓ Same answer, picture it.

**(c) Expected value** — average of the three numbers:

$$ \mathbb{E}[X] = \frac{a+b+c}{3} = \frac{0+2+1}{3} = \mathbf{1 \text{ hour}} $$

---

## 3.28 Worked Example — Expectation of a Function (ties §3.16 + Triangular)

**Problem:** $X \sim \text{Triangular}(a{=}0, b{=}3, c{=}2)$ h. Find $\mathbb{E}[X]$ and $\mathbb{E}\big[\tfrac{4}{X+1} + \tfrac{1}{2}X^2 + 3\big]$.

**Mean of X:**

$$ \mathbb{E}[X] = \frac{0+3+2}{3} = \frac{5}{3} \approx 1.67 \text{ h} $$

Evaluating the function (per §3.16) gives approximately:

$$ \mathbb{E}\!\left[\frac{4}{X+1} + \frac{1}{2}X^2 + 3\right] \approx 0.6 + 1.5 + 3 \approx \mathbf{5.1} $$

> **The exam point is the *process*, not the decimal:** expectation of a function = integrate $f(x)\,p(x)$ over the support (§3.16). And **linearity** lets you split sums and pull out constants: $\mathbb{E}[a\,g(X)+c] = a\,\mathbb{E}[g(X)] + c$. Memorize *that*, not the number.

---

# 🎯 Quick Revision Cheat-Sheet (Chapter 3)

### The Family Tree (recall the connections)

```
Bernoulli (1 trial) → Binomial (n trials, count) → Multinomial (K outcomes)
Gaussian (bell) → Standard Normal (z-scores)
Beta → prior for Bernoulli/Binomial      Dirichlet → prior for Multinomial
Triangular → only min/mode/max known
```

### Core Distributions

| Distribution | Formula | Mean | Variance | Use when |
|---|---|---|---|---|
| **Bernoulli** | $\mu^x(1-\mu)^{1-x}$ | $\mu$ | $\mu(1-\mu)$ | 1 trial, 2 outcomes |
| **Binomial** | $\binom{n}{m}\mu^m(1-\mu)^{n-m}$ | $n\mu$ | $n\mu(1-\mu)$ | # successes in $n$ trials |
| **Multinomial** | $\prod_k \mu_k^{x_k}$ | — | — | $K$-category outcome |
| **Gaussian** | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | continuous bell-shaped data |
| **Standard Normal** | $\frac{1}{\sqrt{2\pi}}e^{-z^2/2}$, $z=\frac{x-\mu}{\sigma}$ | 0 | 1 | standardize / z-table |
| **Beta** | $\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$ | $\frac{a}{a+b}$ | $\frac{ab}{(a+b)^2(a+b+1)}$ | prior for $\mu$ (Bernoulli/Binomial) |
| **Dirichlet** | $\frac{\Gamma(\alpha_0)}{\prod\Gamma(\alpha_k)}\prod\mu_k^{\alpha_k-1}$ | — | — | prior for multinomial |
| **Triangular** | piecewise linear (a, c, b) | $\frac{a+b+c}{3}$ | $\frac{a^2+b^2+c^2-ab-ac-bc}{18}$ | only min/mode/max known |

### Key Distinctions to Memorize
- **Bernoulli (1 trial) → Binomial (n trials) → Multinomial (K outcomes).**
- **With replacement → Bernoulli trials. Without replacement → NOT (p changes, dependent).**
- **MLE** of $\mu$ = (#successes)/(N); can **overfit** small samples → use a **Beta prior**.
- **Variance** $= \mathbb{E}[X^2] - (\mathbb{E}[X])^2$; SD $=\sqrt{\text{Var}}$; sample divides by $n-1$.
- **z-score** $= (x-\mu)/\sigma$; 68–95–99.7 rule for $\pm1,2,3\sigma$.
- **Conjugate priors:** Beta ↔ Bernoulli/Binomial; Dirichlet ↔ Multinomial → posterior stays same family (Bayesian updating, links to Ch 2 §2.13–2.14).
- **Exponential family** = the club containing Bernoulli, Binomial, Multinomial, Poisson, Gaussian, Dirichlet…

### Practice Questions to Test Yourself
1. Write the Bernoulli PMF and verify it for $x=0$ and $x=1$.
2. A fair coin is tossed 10 times. P(exactly 6 heads)? *(Ans: $\binom{10}{6}/1024 = 210/1024 \approx 0.205$)*
3. Why are draws **without replacement** not Bernoulli trials?
4. Give mean and variance of $\mathrm{Bin}(n,\mu)$.
5. Convert $x=85$ to a z-score if $\mu=70,\sigma=10$. *(Ans: $z=1.5$)*
6. State the 68–95–99.7 rule.
7. What makes the Beta a **conjugate** prior for the binomial? Give the posterior parameters.
8. Triangular with $a=0,c=4,b=10$: find $\mathbb{E}[X]$. *(Ans: $14/3 \approx 4.67$)*
9. Show $\mathrm{Var}(X)=\mathbb{E}[X^2]-\mathbb{E}[X]^2$ for a fair die. *(Ans: $\approx 2.92$)*
10. How does the Dirichlet relate to the Beta?

---

*End of Chapter 3. Restructured as a learn-path: every section follows ① when → ② formula → ③ worked example → ④ spread, with a family-tree thread connecting all distributions. No content removed — all formulas, tables, and worked examples preserved.*
