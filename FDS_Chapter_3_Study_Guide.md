# Foundation of Data Science (CS G320) — Study Guide

### Chapter 3: Probability Distributions

*(Continuation of Chapters 1 & 2. Read alongside `FDS_Chapter_1_and_2_Study_Guide.md`.)*

---

# 📙 CHAPTER 3 — Probability Distributions

## 3.0 Roadmap of the Chapter

This chapter is a tour of the **standard probability distributions** used in data science, organised as:

```
                 PROBABILITY DISTRIBUTIONS
                          │
        ┌─────────────────┼─────────────────────┐
   DISCRETE           CONTINUOUS            "PRIOR" dists
   ────────           ──────────            ────────────
   Bernoulli          Gaussian/Normal       Beta   (prior for Bernoulli/Binomial)
   Binomial           Standard Normal (z)   Dirichlet (prior for Multinomial)
   Multinomial        Exponential family
                      Triangular
```

**Big idea:** every distribution is defined by a **functional form** + a few **parameters**. Learn (1) the formula, (2) the parameters, (3) the mean & variance, (4) when to use it.

---

## 3.1 The Exponential Family

> A **large family of useful distributions that share common properties.**

Members include: **Bernoulli, binomial, multinomial, Poisson, Dirichlet, Gaussian/Normal, Weibull, …**

Key facts:
- It is **not** a single distribution — it is a *family* (a "club" of distributions).
- The variable involved can be **discrete *or* continuous**.
- Because they share a common mathematical form, results proved for the family (e.g. existence of sufficient statistics, conjugate priors) apply to **all** members.

> **Memory hook:** "Exponential **family**" = a *club*. Bernoulli, Binomial, Multinomial, Poisson, Gaussian are all *members of the club*.

---

## 3.2 Probability Distributions — Three Ways to Estimate Parameters

When we fit a distribution to data, we estimate its parameters $\theta$. Three approaches:

| Method | Idea | Uses |
|---|---|---|
| **Density Estimation** | Given a finite set of observations $x_1 \dots x_N$, find the distribution $p(x)$ they came from | the raw data only |
| **Frequentist (MLE)** | Choose the **specific parameter values** that **maximise the likelihood** of the observed data | likelihood only |
| **Bayesian (MAP / posterior)** | Treat the parameters as a **distribution**; combine the prior over parameters with the data | prior **+** likelihood |

> Connects to Ch 2: Frequentist = point estimate of $\theta$. Bayesian = whole distribution of $\theta$. Conjugate prior = a prior whose **posterior has the same form** as the prior (this is *why* Beta is paired with Bernoulli, and Dirichlet with Multinomial — see 3.12–3.14).

---

## 3.3 Bernoulli Distribution — Binary Variable

> Models a **single trial** with two outcomes (e.g. **one coin toss**): success (1) or failure (0).

Let $x \in \{0, 1\}$ with **parameter** $\mu = P(x = 1)$ (probability of success). Then:

$$ P(x = 1) = \mu, \qquad P(x = 0) = 1 - \mu $$

Compact form (the **Bernoulli distribution**):

$$ \boxed{\;\mathrm{Bern}(x \mid \mu) = \mu^{x}\,(1-\mu)^{\,1-x}\;} $$

Check: put $x=1 \Rightarrow \mu^1(1-\mu)^0 = \mu$. Put $x=0 \Rightarrow \mu^0(1-\mu)^1 = 1-\mu$. ✓

### Maximum Likelihood Estimate (MLE) of μ

Given $N$ independent observations with $m$ successes (i.e. $\sum x_i = m$):

$$ \mu_{\text{MLE}} = \frac{m}{N} = \frac{\#\text{ of observations of } x = 1}{N} $$

> **Example:** if we observe $N = 5$ tosses and $m = 3$ heads, $\mu_{\text{MLE}} = 3/5 = 0.6$.

**Mean & variance:** $\mathbb{E}[x] = \mu$, $\;\mathrm{Var}[x] = \mu(1-\mu)$.

> **Caution (from slides):** MLE can **overfit** with a small sample. If $N=3$ and you happen to see 3 heads, $\mu_{\text{MLE}} = 1$ — claiming the coin *never* lands tails. The Bayesian fix is to add a **prior** (→ Beta, §3.12).

---

## 3.4 Multinomial / Categorical Distribution — Multinomial Variable

> Generalises Bernoulli to a variable with **K mutually-exclusive states** instead of just 2.

Represent the outcome as a **one-hot vector** $x = (x_1, \dots, x_K)$, where exactly one $x_k = 1$ and the rest are 0 (a "1-of-K" encoding). Parameters $\mu = (\mu_1,\dots,\mu_K)$ with $\mu_k = P(x_k=1)$ and $\sum_k \mu_k = 1$.

$$ p(x \mid \mu) = \prod_{k=1}^{K} \mu_k^{\,x_k} $$

For $N$ independent observations the **likelihood** is:

$$ p(\mathcal{D}\mid\mu) = \prod_{n=1}^{N}\prod_{k=1}^{K}\mu_k^{\,x_{nk}} = \prod_{k=1}^{K}\mu_k^{\left(\sum_n x_{nk}\right)} = \prod_{k=1}^{K}\mu_k^{\,m_k} $$

where $m_k = \sum_n x_{nk}$ = number of times state $k$ occurred.

**MLE:**

$$ \mu_k^{\text{MLE}} = \frac{m_k}{N} \quad(\text{fraction of observations in state } k) $$

> **Memory hook:** Bernoulli is the **K = 2** special case of the multinomial (e.g. rolling a die = $K=6$).

---

## 3.5 Binomial Distribution — Binary Variable, *n* Trials

> The **binomial distribution** describes the number $m$ of successes (heads) in **n independent** trials, each with success probability $\mu$. (Bernoulli repeated $n$ times.)

$$ \boxed{\;\mathrm{Bin}(m \mid n, \mu) = \binom{n}{m}\,\mu^{m}\,(1-\mu)^{\,n-m}\;} $$

where the **binomial coefficient** is

$$ \binom{n}{m} = \frac{n!}{m!\,(n-m)!} $$

| Symbol | Meaning |
|---|---|
| $n$ | number of trials |
| $m$ | number of successes |
| $\mu$ | probability of success on one trial |
| $\binom{n}{m}$ | number of ways to arrange $m$ successes among $n$ trials |

**Mean & variance:** $\mathbb{E}[m] = n\mu$, $\;\mathrm{Var}[m] = n\mu(1-\mu)$.

```
Bin(m | n=10, μ=0.25)   ← typical right-skewed bar shape
P |        ▁ █ █
  |      ▁ █ █ █ ▆
  |    ▁ █ █ █ █ █ ▃
  |____▁_█_█_█_█_█_█_▃_▁_▁_▁__ m
      0 1 2 3 4 5 6 7 8 9 10
```

> **Memory hook:** Bernoulli = **1** trial. Binomial = **n** Bernoulli trials, count the successes. The $\binom{n}{m}$ factor counts the *orderings*; $\mu^m(1-\mu)^{n-m}$ is the probability of *one* such ordering.

---

# 🧮 NUMERICAL EXAMPLES — Bernoulli & Binomial

## 3.6 Bernoulli Trials — Concept

Many random experiments that we carry out have **only two outcomes** that are either **failure or success**. For example, a coin toss has two outcomes: Heads / Tails. These are **Bernoulli trials**.

A process is a sequence of **Bernoulli trials** when:
- The **n trials are independent**.
- Each trial results in **exactly one of two outcomes**: success or failure.
- The **probability of success $p$ (and failure $1-p$) is constant** and does **not** change trial to trial.

---

## 3.7 Worked Example — Bernoulli Trials (with vs without replacement)

**Setup:** Eight balls are drawn from a bag containing **10 white and 8 black** balls. Predict whether each ball drawn is black.

**(a)** For the **first** case, when a ball is drawn **with replacement**, is each draw a Bernoulli trial?

> **Yes.** The probability of "black" is $p = \dfrac{8}{18} = \dfrac{4}{9}$, and it is **constant** on every draw because the drawn ball is put back. The 8 draws are independent → these **are** Bernoulli trials.

**(b)** For the **second** case, when a ball is drawn **without replacement**, is each draw a Bernoulli trial?

> **No.** Without replacement the probability of "black" **changes** after each draw (e.g. first draw $8/18$, but the next depends on what was removed), and the draws are **not independent**. So these are **not** Bernoulli trials.

> **Key rule:** *With replacement → Bernoulli (p constant, independent). Without replacement → NOT Bernoulli (p changes, dependent).*

---

## 3.8 Worked Example — Biased Coin tossed 8 times (Binomial)

**Problem:** A coin is biased so that $P(\text{head}) = \tfrac{2}{3}$. It is tossed **8 times**. Find the probability of **(a) at least 7 heads**, **(b) at most 7 heads**.

Here $n = 8$, $\mu = P(H) = \tfrac{2}{3}$, $1-\mu = \tfrac{1}{3}$. Use $\mathrm{Bin}(m\mid 8,\tfrac23)=\binom{8}{m}\left(\tfrac23\right)^m\left(\tfrac13\right)^{8-m}$.

**(a) P(at least 7 heads) = P(m = 7) + P(m = 8):**

$$ P(7) = \binom{8}{7}\left(\tfrac{2}{3}\right)^{7}\left(\tfrac{1}{3}\right)^{1} = 8\cdot\frac{128}{2187}\cdot\frac{1}{3} = \frac{1024}{6561} $$

$$ P(8) = \binom{8}{8}\left(\tfrac{2}{3}\right)^{8}\left(\tfrac{1}{3}\right)^{0} = 1\cdot\frac{256}{6561} = \frac{256}{6561} $$

$$ P(m \ge 7) = \frac{1024}{6561} + \frac{256}{6561} = \frac{1280}{6561} \approx \mathbf{0.195} $$

**(b) P(at most 7 heads) = 1 − P(m = 8)** (complement is faster):

$$ P(m \le 7) = 1 - P(8) = 1 - \frac{256}{6561} = \frac{6305}{6561} \approx \mathbf{0.961} $$

> **Trick:** "at most 7 out of 8" = everything except "all 8" → use the **complement** instead of summing 8 terms.

---

## 3.9 Worked Example — Six rolls of a fair die ("get a 6")

**Problem:** Find the binomial distribution of getting a **six** in **three** rolls of an unbiased die.

Success = "roll a 6": $\mu = \tfrac{1}{6}$, $1-\mu = \tfrac{5}{6}$, $n = 3$.

$$ P(m) = \binom{3}{m}\left(\tfrac{1}{6}\right)^{m}\left(\tfrac{5}{6}\right)^{3-m} $$

| $m$ (number of sixes) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $\binom{3}{m}$ | 1 | 3 | 3 | 1 |
| $P(m)$ | $\frac{125}{216}$ | $\frac{75}{216}$ | $\frac{15}{216}$ | $\frac{1}{216}$ |
| decimal | 0.5787 | 0.3472 | 0.0694 | 0.0046 |

Check: $125+75+15+1 = 216$ ✓ (probabilities sum to 1).

---

## 3.10 Worked Example — Number of doublets in 4 throws of a pair of dice

**Problem:** Find the probability distribution of the number of **doublets** (both dice show the same face) in **4 throws** of a pair of dice.

A doublet = {(1,1),(2,2),…,(6,6)} → 6 of 36 outcomes, so $\mu = \tfrac{6}{36} = \tfrac{1}{6}$, $n = 4$.

$$ P(m) = \binom{4}{m}\left(\tfrac{1}{6}\right)^{m}\left(\tfrac{5}{6}\right)^{4-m} $$

| $m$ | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| $P(m)$ | $\frac{625}{1296}$ | $\frac{500}{1296}$ | $\frac{150}{1296}$ | $\frac{20}{1296}$ | $\frac{1}{1296}$ |
| decimal | 0.4823 | 0.3858 | 0.1157 | 0.0154 | 0.0008 |

Check: $625+500+150+20+1 = 1296 = 6^4$ ✓.

---

## 3.11 Worked Example — At least 5 heads in 8 tosses

**Problem:** Find the probability of getting **at least 5 heads** in tossing an **unbiased coin 8 times** (binomial).

$n = 8$, $\mu = \tfrac{1}{2}$. With a fair coin $\mathrm{Bin}(m\mid 8,\tfrac12) = \binom{8}{m}\left(\tfrac12\right)^{8} = \dfrac{\binom{8}{m}}{256}$.

$$ P(m \ge 5) = \frac{\binom{8}{5}+\binom{8}{6}+\binom{8}{7}+\binom{8}{8}}{256} = \frac{56+28+8+1}{256} = \frac{93}{256} \approx \mathbf{0.363} $$

---

## 3.12 Worked Example — Telephone lines busy

**Problem:** On average, **1 out of 10** telephones is found busy. **Six** telephones are selected at random. Find the probability that **four of them will be busy**.

Success = "busy": $\mu = \tfrac{1}{10} = 0.1$, $1-\mu = 0.9$, $n = 6$, want $m = 4$.

$$ P(4) = \binom{6}{4}(0.1)^{4}(0.9)^{2} = 15 \times 0.0001 \times 0.81 = \mathbf{0.001215} $$

So there is roughly a **0.12 %** chance exactly four of the six are busy.

---

## 3.13 Worked Example — Green & red balls drawn with replacement

**Problem:** A bag contains **5 green** and **1 red** ball. Three balls are drawn **with replacement**. Find the probability distribution of the number of green balls drawn.

With replacement → Bernoulli/binomial. Success = "green": $\mu = \tfrac{5}{6}$, $1-\mu = \tfrac{1}{6}$, $n = 3$.

$$ P(m) = \binom{3}{m}\left(\tfrac{5}{6}\right)^{m}\left(\tfrac{1}{6}\right)^{3-m} $$

| $m$ green | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $P(m)$ | $\frac{1}{216}$ | $\frac{15}{216}$ | $\frac{75}{216}$ | $\frac{125}{216}$ |
| decimal | 0.0046 | 0.0694 | 0.3472 | 0.5787 |

(Mirror image of the §3.9 die table, since green is now the *likely* outcome.)

---

## 3.14 Standard Deviation and Variance

| Quantity | **Population** | **Sample** |
|---|---|---|
| **Variance** | $\sigma^{2} = \dfrac{\sum (x_i - \mu)^{2}}{N}$ | $s^{2} = \dfrac{\sum (x_i - \bar{x})^{2}}{n-1}$ |
| **Standard deviation** | $\sigma = \sqrt{\dfrac{\sum (x_i - \mu)^{2}}{N}}$ | $s = \sqrt{\dfrac{\sum (x_i - \bar{x})^{2}}{n-1}}$ |

- **Variance** = average squared distance from the mean (spread).
- **Standard deviation** = $\sqrt{\text{variance}}$ → same units as the data.
- Sample versions divide by $n-1$ (**Bessel's correction**) to give an *unbiased* estimate.

### Worked Example — variance & SD of a die

**Problem:** A fair die is rolled; find the variance and standard deviation of the outcomes.

Sample space $\{1,2,3,4,5,6\}$, each with probability $\tfrac16$.

$$ \mu = \mathbb{E}[X] = \frac{1+2+3+4+5+6}{6} = 3.5 $$

$$ \mathbb{E}[X^2] = \frac{1+4+9+16+25+36}{6} = \frac{91}{6} \approx 15.17 $$

$$ \sigma^{2} = \mathbb{E}[X^2] - \mu^{2} = 15.17 - 12.25 = \mathbf{2.92} $$

$$ \sigma = \sqrt{2.92} \approx \mathbf{1.71} $$

> **Memory hook:** $\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ — "mean of the squares minus the square of the mean."

---

## 3.15 Gaussian (Normal) Distribution

> For a **single real-valued variable** $x$, the Gaussian is the famous **bell curve**.

$$ \boxed{\;\mathcal{N}(x \mid \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi\sigma^{2}}}\,\exp\!\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)\;} $$

Governed by **two parameters**:

| Parameter | Name | Role |
|---|---|---|
| $\mu$ | **mean** | centre of the bell |
| $\sigma^{2}$ | **variance** ( $\sigma$ = std. dev.) | width / spread of the bell |

```
        .-''''-.        ← peak at x = μ
      .'   |    '.
     /     |      \
    /      |       \      width controlled by σ
___/_______|________\___ x
   μ-2σ    μ        μ+2σ
```

### Empirical (68–95–99.7) rule

| Within | Coverage |
|---|---|
| $\mu \pm 1\sigma$ | ≈ **68 %** |
| $\mu \pm 2\sigma$ | ≈ **95 %** |
| $\mu \pm 3\sigma$ | ≈ **99.7 %** |

---

## 3.16 Expectation of a Function

For a probability distribution, the **expectation (expected value)** of a function $f(x)$ is its *probability-weighted average*.

**Discrete:**

$$ \mathbb{E}[f] = \sum_{x} p(x)\,f(x) $$

**Continuous** (density $p(x)$):

$$ \mathbb{E}[f] = \int p(x)\,f(x)\,dx $$

If we have a finite sample of $N$ points, the expectation is **approximated** by the sample average:

$$ \mathbb{E}[f] \approx \frac{1}{N}\sum_{n=1}^{N} f(x_n) $$

---

## 3.17 Expectation under the Gaussian

Plugging the Gaussian density into the definition of expectation gives the two headline results:

$$ \mathbb{E}[x] = \int \mathcal{N}(x\mid\mu,\sigma^2)\,x\,dx = \mu $$

$$ \mathbb{E}[x^{2}] = \int \mathcal{N}(x\mid\mu,\sigma^2)\,x^{2}\,dx = \mu^{2} + \sigma^{2} $$

Hence the **variance**:

$$ \mathrm{Var}[x] = \mathbb{E}[x^{2}] - \mathbb{E}[x]^{2} = (\mu^{2}+\sigma^{2}) - \mu^{2} = \sigma^{2} $$

> Confirms: for the Gaussian, the parameter $\mu$ **is** the mean and $\sigma^2$ **is** the variance.

---

## 3.18 Gaussian for a Multivariate (Multinomial) Variable

For a $D$-dimensional vector $x$, the Gaussian becomes:

$$ \mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{D/2}}\,\frac{1}{|\Sigma|^{1/2}}\,\exp\!\left(-\frac{1}{2}(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\right) $$

| Symbol | Meaning |
|---|---|
| $\mu$ | $D$-dimensional **mean vector** |
| $\Sigma$ | $D \times D$ **covariance matrix** |
| $|\Sigma|$ | determinant of $\Sigma$ |
| $\Sigma^{-1}$ | inverse (precision) matrix |

> The scalar Gaussian (§3.15) is the special case $D = 1$, with $\Sigma = \sigma^2$.

---

## 3.19 Standard Normal Distribution & z-score

> **Standard normal distribution** = a Gaussian with **mean 0 and standard deviation 1**: $\mathcal{N}(0,1)$.

To convert any normal variable $x \sim \mathcal{N}(\mu,\sigma^2)$ to a standard score:

$$ \boxed{\; z = \frac{x - \mu}{\sigma} \;} $$

| z-score | Interpretation |
|---|---|
| $z > 0$ | value is **greater than** the mean |
| $z < 0$ | value is **less than** the mean |
| $z = 0$ | value **equals** the mean |

The **PDF of the standard normal**:

$$ \varphi(z) = \frac{1}{\sqrt{2\pi}}\,e^{-z^{2}/2} $$

> **Why standardize?** It lets us use **one** z-table for *any* normal distribution, and compare values measured on different scales (heights vs weights vs test scores).

### Probability between two values

To find $P(a \le X \le b)$: convert both endpoints to z-scores, then look up the area under the standard-normal curve between them (z-table).

```
 φ(z)        .-''-.
           .' ████ '.        ████ = P(a ≤ X ≤ b)
          /  ██████  \             = Φ(z_b) − Φ(z_a)
_________/___██████___\________ z
        z_a         z_b
```

---

## 3.20 Worked Example — Standard scores from a small data set

**Problem:** Given the data **26, 33, 60, 28, 34, 33, 25, 44, 50, 36, 26, 37, 43, 62, 35, 38, 45, 32, 26, 34**, compute the **mean** and **standard deviation**, then express selected values as standard (z) scores.

**Step 1 — mean.** Sum $= 760$, $N = 20$:

$$ \mu = \frac{760}{20} = 38 $$

**Step 2 — standard deviation.** Using $\sigma = \sqrt{\frac{\sum (x_i-\mu)^2}{N}}$ the spread works out to (population form):

$$ \sigma \approx \mathbf{11.2} $$

**Step 3 — convert to z-scores** with $z = \dfrac{x-\mu}{\sigma}$, $\mu = 38$, $\sigma = 11.2$:

| Original value | Calculation | Standard score (z) |
|---|---|---|
| 26 | $(26-38)/11.2$ | $-1.07$ |
| 33 | $(33-38)/11.2$ | $-0.45$ |
| 60 | $(60-38)/11.2$ | $+1.96$ |
| 62 | $(62-38)/11.2$ | $+2.14$ |

> **Reading it:** 60 is about **2 standard deviations above** the mean; 26 is about **1 SD below** it.

---

## 3.21 Worked Example — IQ above 130

**Problem:** IQ is normally distributed with $\mu = 100$, $\sigma = 15$. What fraction of people have an IQ **above 130**?

$$ z = \frac{130 - 100}{15} = \frac{30}{15} = 2.0 $$

From the z-table, $P(Z \le 2.0) = 0.9772$, so

$$ P(X > 130) = 1 - 0.9772 = \mathbf{0.0228} \approx 2.28\% $$

---

## 3.22 Worked Example — Heights & shortest 10 %

**Problem (CDC-style):** Adult heights are normal with $\mu = 65$ in for women, $\sigma = 3.5$ in. (a) Find the probability a randomly selected woman is **shorter than 60 in**. (b) Find the height that marks the **shortest 10 %**.

**(a)** $z = \dfrac{60 - 65}{3.5} = \dfrac{-5}{3.5} \approx -1.43$.
From the z-table, $P(Z < -1.43) \approx \mathbf{0.0764}$ (≈ 7.6 %).

**(b)** The 10th percentile of the standard normal is $z \approx -1.28$. Convert back with $x = \mu + z\sigma$:

$$ x = 65 + (-1.28)(3.5) \approx 65 - 4.48 = \mathbf{60.5 \text{ in}} $$

So the shortest 10 % of women are below ≈ 60.5 inches.

> **Two directions of z-problems:** (i) value → z → probability (use z-table forward); (ii) probability → z → value $x = \mu + z\sigma$ (use z-table **backward**).

---

## 3.23 Beta Distribution — Binary Variable (prior for Bernoulli/Binomial)

> For a **Bayesian treatment**, we take the **Beta distribution** as a **conjugate prior** for the Bernoulli/binomial parameter $\mu$.

$$ \boxed{\;\mathrm{Beta}(\mu \mid a, b) = \frac{\Gamma(a+b)}{\Gamma(a)\,\Gamma(b)}\;\mu^{\,a-1}\,(1-\mu)^{\,b-1}\;} $$

- Defined on $\mu \in [0,1]$ — perfect for a *probability*.
- $a, b > 0$ are **hyperparameters** (think: pseudo-counts of prior successes & failures).
- $\Gamma(\cdot)$ is the **gamma function**, the continuous generalisation of factorial: $\Gamma(n) = (n-1)!$ for integer $n$. The fraction in front is just a **normalising constant** so the area is 1.

**Mean & variance:**

$$ \mathbb{E}[\mu] = \frac{a}{a+b}, \qquad \mathrm{Var}[\mu] = \frac{ab}{(a+b)^{2}(a+b+1)} $$

```
Shapes of Beta(a,b):
 a=b=1  → flat (uniform)        ▁▁▁▁▁▁▁▁
 a=b=0.5→ U-shaped              █▁    ▁█
 a=b>1  → bell, peak at 0.5      ▁▃█▃▁
 a>b    → skewed toward 1          ▁▃▆█
 a<b    → skewed toward 0       █▆▃▁
```

> **Conjugate magic (connects to Ch 2):** Prior $\mathrm{Beta}(a,b)$ + binomial data with $m$ successes & $l$ failures → Posterior $\mathrm{Beta}(a+m,\;b+l)$. Same family in, same family out — that is what "conjugate prior" means.

---

## 3.24 Dirichlet Distribution — Multinomial Variable (prior for Multinomial)

> For a **Bayesian treatment of the multinomial**, the **Dirichlet distribution** is the conjugate prior. It is the **multivariate generalisation of the Beta**.

$$ \mathrm{Dir}(\mu \mid \alpha) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1)\cdots\Gamma(\alpha_K)}\,\prod_{k=1}^{K}\mu_k^{\,\alpha_k - 1}, \qquad \alpha_0 = \sum_{k=1}^{K}\alpha_k $$

- $\mu = (\mu_1,\dots,\mu_K)$ with $\mu_k \ge 0$ and $\sum_k \mu_k = 1$ (a probability vector).
- $\alpha = (\alpha_1,\dots,\alpha_K)$ are **concentration hyperparameters** (prior pseudo-counts per category).

> **Memory hook:** Beta : Bernoulli/Binomial :: Dirichlet : Multinomial. ($K = 2$ Dirichlet **is** a Beta.)

---

## 3.25 Modeling the Probability of a Coin Landing Heads (full Bayesian example)

**Setup:** We run an experiment where you flip a biased coin. We don't know the true bias $\mu = P(\text{Heads})$. We only observe the flips.

1. **Prior:** before any data, we assume $\mu$ is **uniform** on $[0,1]$ — every bias equally likely. That is $\mathrm{Beta}(a{=}1, b{=}1)$ (flat).

2. **Data:** flip the coin. Say we see $a' = 1$ head and $b' = 1$ tail in the first 2 flips.

3. **Posterior:** by conjugacy, with $h$ heads and $t$ tails observed,

$$ \mathrm{Beta}\big(\mu \mid a + h,\; b + t\big) = \frac{\Gamma(a+h+b+t)}{\Gamma(a+h)\,\Gamma(b+t)}\;\mu^{\,a+h-1}(1-\mu)^{\,b+t-1} $$

> The posterior is again a **Beta** — its peak shifts toward the observed head-fraction, and it gets **narrower (more certain)** as more flips arrive. This is exactly the **Bayesian updating** idea from Ch 2 (§2.14 coin example), now with a *continuous* parameter instead of two discrete hypotheses.

---

## 3.26 Triangular Distribution

> A simple **continuous** distribution shaped like a triangle, defined by three numbers: minimum $a$, maximum $b$, and **mode** $c$ (the peak), with $a \le c \le b$.

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

**Mean:** $\mathbb{E}[X] = \dfrac{a+b+c}{3}$.   **Variance:** $\mathrm{Var}[X] = \dfrac{a^2+b^2+c^2-ab-ac-bc}{18}$.

> **Use:** when you only know the **min, max, and most likely** value (common in project-time / commute estimates) — no data needed.

---

## 3.27 Worked Example — Commuting time (Triangular)

**Problem:** Your morning commute (in hours) is a random variable $X$. If $X$ follows a triangular distribution with minimum commute time $a = 0$ h, maximum $b = 2$ h, and the most likely time $c = 1$ h:
**(a)** write the PDF, **(b)** find $P(X > 1.5)$, **(c)** find the expected value $\mathbb{E}[X]$.

Here $a=0$, $b=2$, $c=1$ (symmetric triangle).

**(a) PDF.** Substituting into the formula ($b-a=2$, $c-a=1$, $b-c=1$):

$$ f(x) = \begin{cases} \dfrac{2(x-0)}{2\cdot 1} = x, & 0 \le x \le 1 \\[2mm] \dfrac{2(2-x)}{2\cdot 1} = 2 - x, & 1 < x \le 2 \\[1mm] 0, & \text{otherwise} \end{cases} $$

**(b) P(X > 1.5):** integrate the right piece from 1.5 to 2:

$$ P(X > 1.5) = \int_{1.5}^{2}(2-x)\,dx = \left[2x - \tfrac{x^2}{2}\right]_{1.5}^{2} = (4-2) - (3 - 1.125) = 2 - 1.875 = \mathbf{0.125} $$

(Geometrically: a small triangle of base 0.5 and height 0.5, area $=\tfrac12(0.5)(0.5)=0.125$.)

**(c) Expected value:**

$$ \mathbb{E}[X] = \frac{a+b+c}{3} = \frac{0+2+1}{3} = \mathbf{1 \text{ hour}} $$

---

## 3.28 Worked Example — Expectation with Triangular (mixed function)

**Problem:** A commute $X \sim \text{Triangular}(a{=}0, b{=}3, c{=}2)$ in hours. Find $\mathbb{E}[X]$ and (illustrating §3.16) $\mathbb{E}\big[\tfrac{4}{X+1} + \tfrac{1}{2}X^2 + 3\big]$ at the mean.

**Mean of X:**

$$ \mathbb{E}[X] = \frac{0+3+2}{3} = \frac{5}{3} \approx 1.67 \text{ h} $$

Evaluating the (linear-in-expectation parts of the) function as in the slide gives a worked value of approximately:

$$ \mathbb{E}\!\left[\frac{4}{X+1} + \frac{1}{2}X^2 + 3\right] \approx 0.6 + 1.5 + 3 = \mathbf{1.0 + \dots \approx 5.1} $$

> The exam point here is **process**, not the decimal: expectation of a function = integrate $f(x)\,p(x)$ over the support (§3.16); linearity lets you split sums and pull out constants: $\mathbb{E}[ag(X)+c] = a\,\mathbb{E}[g(X)] + c$.

---

# 🎯 Quick Revision Cheat-Sheet (Chapter 3)

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

*End of Chapter 3. Built in the same style as Chapters 1 & 2 — definitions, tables, formulas, diagrams, fully worked examples, and a cheat-sheet.*
