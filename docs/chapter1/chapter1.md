# Introduction to Reinforcement Learning

Author: [Omar Abu Shanab](https://www.linkedin.com/in/omar-abu-shanab-46b164131/)


---

## Chapter Overview & Agenda

Welcome to the introduction to Reinforcement Learning (RL). This chapter establishes the foundational concepts, intuition, and vocabulary necessary before diving into the formal mathematics of RL systems. 

Here is what you can expect to learn:
* **The Core Concept of RL:** Understanding how an agent learns through interaction, driven by delayed consequences and the absence of an external supervisor.
* **RL vs. Other Paradigms:** How Reinforcement Learning distinguishes itself from Supervised and Unsupervised Learning as a distinct, third paradigm of Machine Learning.
* **The Exploration-Exploitation Trade-off:** The fundamental dilemma of balancing the discovery of new actions with the execution of known, rewarding ones.
* **The n-Armed Bandit Problem:** A simplified, stateless scenario used to isolate and study this core trade-off using action-value methods.
* **The Four Pillars of an RL System:** A detailed breakdown of the essential components that dictate an RL agent's behavior: the Policy, the Reward Signal, the Value Function, and (optionally) a Model of the environment.
---

## What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a branch of Machine Learning that focuses on how an agent should take actions in an environment to maximize the reward it receives over time.

Rather than learning from labeled examples or discovering hidden patterns in data, an RL agent learns by *doing* — interacting with its world, observing the consequences, and gradually improving its behavior. RL refers simultaneously to a class of problems, a family of solution methods that work well on those problems, and the field of study that examines both.

### The Three Distinguishing Features of an RL Problem

| Feature | Description |
|---|---|
| **Closed-loop** | The agent's actions influence its future inputs — there is no separation between actor and environment |
| **No supervisor** | There is no teacher telling the agent which action to take; it must discover what works |
| **Delayed consequences** | The effects of actions play out over extended time periods, not just immediately |

<div align="center">
  <img src="RL feedback loop.png" alt="Closed Loop" width="60%">
  <p style="font-size: 0.9em; margin-bottom: 0;"><em>Figure 1.1 — The agent–environment closed-loop: the agent observes state and reward, selects an action, and the environment transitions accordingly.</em></p>
  <span style="color: gray; font-size: 0.9em;"> (Source: Sutton & Barto, 2018) </span>
</div>

> **Example — Learning to Drive:** When you first learn to drive, no instructor controls the steering wheel for you. You observe the road (state), decide when to turn or brake (action), and feel the outcome — staying in lane or drifting off (reward signal). Over many attempts, you improve. This is the RL loop in everyday life.

---

## Formulation of an RL Problem

Every RL problem requires three ingredients:

1. **Sensation** — the agent must be able to sense the state of the environment to some extent
2. **Action** — the agent must be able to affect the state through its choices
3. **Goal** — the agent must have an explicit objective tied to the environment's state

Any method well-suited to problems with these three aspects can be considered an RL solution method.

---

## Comparison with Other Learning Paradigms

### Supervised Learning

<div align="center">
  <img src="supervised-machine-learning.webp" alt="Supervised Learning" width="60%">
  <p style="font-size: 0.9em; margin-bottom: 0;"><em>Figure 1.2: Supervised Learning</em></p>
  <span style="color: gray; font-size: 0.9em;">(Source: GeeksforGeeks)</span>
</div>


Supervised learning uses a training set of labeled examples provided by a knowledgeable external supervisor. Each example pairs a situation with the correct action or label. The goal is for the agent to **generalize** — to act correctly in situations not seen during training.

This approach falls short in interactive settings. In real-world interaction, it is often impossible to obtain examples that are both *correct* and *representative of every situation* the agent might face. In uncharted territory, we want the agent to learn from its own experience — not from a fixed dataset.

### Unsupervised Learning

<div align="center">
  <img src="unsupervised learning.png" alt="Unsupervised Learning" width="60%">
  <p style="font-size=0.9em; margin-bottom: 0;"><em>Figure 1.3 — Unsupervised learning seeks hidden structure in unlabeled data.
  </em></p>
  <span style="color: gray; font-size: 0.9em;"> (Source: Hosseini et al., 2019) </span>
</div>

Unsupervised learning finds hidden structure in collections of unlabeled data. While RL also receives no labeled examples, it is *not* unsupervised learning — its goal is to maximize a reward signal, not to uncover structure. This distinction makes RL a **third, distinct ML paradigm**.

### Summary Comparison

| | Supervised | Unsupervised | Reinforcement |
|---|---|---|---|
| **Labels / Feedback** | Labeled examples | No labels | Reward signal |
| **Goal** | Generalize to new inputs | Find hidden structure | Maximize cumulative reward |
| **Learns from** | A fixed dataset | A fixed dataset | Ongoing interaction |
| **Supervisor** | Yes — a teacher | No | No — must explore |

### Unique Challenges in RL

The most distinctive challenge in RL is the **exploration–exploitation trade-off**:

- To earn reward, the agent must *exploit* actions it already knows to be effective.
- To find better actions, it must *explore* unfamiliar options.

Pursuing either strategy exclusively guarantees failure, as the agent will fall into one of two traps:

* **The Trap of Pure Exploitation (Suboptimal Stagnation):** If an agent only exploits, it acts greedily based solely on its limited initial experience. It will quickly lock onto the first action that yields a positive reward and repeat it forever, getting stuck in a "local optimum." It survives, but it leaves massive potential rewards on the table because it refuses to try anything new.
* **The Trap of Pure Exploration (Random Thrashing):** If an agent only explores, it effectively acts at random forever. It never capitalizes on the knowledge it has gathered to actually harvest rewards. The agent wastes time, energy, and potential score by continuously testing known bad actions simply because they are available.

Neither exploration nor exploitation can dominate. The agent must try a variety of actions and progressively favor those that appear best, shifting its balance as it learns about its environment.

> **Example — A New Restaurant in Town:**
> * **Pure Exploitation:** You eat at your usual, decent restaurant every single night. You never risk a bad meal, but you permanently miss out on discovering the incredible 5-star place that just opened next door.
> * **Pure Exploration:** You force yourself to try a new place every single night, regardless of reviews. You might discover the 5-star place, but you never return to it, and you inevitably eat at places that give you food poisoning. 
> 
> The optimal diner balances both: occasionally taking a risk on new options to update their knowledge (exploration) while predominantly returning to proven favorites to actually enjoy their dinner (exploitation).

---
## Examples of RL Problems

<div align="center">
  <img src="gazelle calf.jpeg" alt="Gazelle Calf learning to run" width="60%">
  <p style="font-size:0.9em; margin-bottom: 0;"><em>Figure 1.4 — A gazelle calf struggles to its feet minutes after birth; within half an hour it runs at speed. No instructor guided it — only the consequences of each attempt.</em></p>
  <span style="color: gray; font-size:0.9em;"> (Source: www.oregonlive.com) </span>
</div>

RL problems share a common structure: an agent with an explicit goal interacts with an uncertain environment, where correct choices require accounting for indirect, delayed consequences. Consider these examples from Sutton & Barto, 2018:

- A **chess player** chooses moves informed by both explicit planning and intuitive judgment about position desirability
- A **petroleum refinery controller** adjusts parameters in real time to optimize yield, cost, and quality trade-offs
- A **mobile robot** decides whether to seek more trash or return to recharge, based on battery level and past experience
- A **gazelle calf** learns to run within minutes of birth, with no teacher — only the feedback of falling or staying upright

All of these involve interaction, uncertainty, explicit goals, and consequences that unfold over time.

---

## The n-Armed Bandit Problem

Before tackling the full RL problem, it helps to study a simpler version that isolates the exploration–exploitation challenge. This is the **n-armed bandit problem**.

### Setup

Imagine you face *n* slot machines (the "arms"), each paying out rewards drawn from an unknown probability distribution. At each step, you pull one lever and receive a reward. Your objective: **maximize total reward over time**.

You don't know which arm is best — you must *learn* it through repeated pulls. This is the bandit problem: pure evaluative feedback with no teacher.

> **Why "bandit"?** The name comes from "one-armed bandit" — a slot machine. With *n* levers instead of one, you face the dilemma of which to pull.

### The Core Dilemma

If you always pull the arm with the highest *estimated* value (greedy), you may never discover that another arm is actually better. If you explore constantly, you waste pulls on suboptimal arms. The tension is unavoidable.

**Action-value methods** address this by maintaining estimates of each arm's expected reward:

$$Q_t(a) = \frac{\text{sum of rewards from arm } a \text{ so far}}{\text{number of times arm } a \text{ was pulled}}$$

**ε-greedy selection** offers a simple balance:
- With probability **1 − ε**: pull the arm with the highest estimated value *(exploit)*
- With probability **ε**: pull a random arm *(explore)*

Even small ε (e.g., 0.1) dramatically outperforms pure greedy play in the long run, because exploration ensures the agent eventually identifies the true best arm.

> **Example — Clinical Trials:** A doctor testing *n* experimental treatments faces the same dilemma. Allocating all patients to the current best estimate ignores potentially superior treatments. Spreading patients randomly wastes lives. Adaptive trial designs (like Thompson Sampling) mirror RL bandit solutions.

### Why It Matters for RL

The bandit problem is the simplest instance of the broader RL challenge. It has no states, no delayed consequences — only the immediate exploration–exploitation trade-off. Mastering it builds intuition for everything that follows.

---

## The Four Main Subelements of an RL System

A complete RL system has four components beyond the agent and environment themselves.

---

### 1 — Policy

A **policy** defines the agent's way of behaving at a given time. It is a mapping from perceived states to actions:

$$\pi: \text{state} \rightarrow \text{action}$$

The policy may be a simple lookup table, a mathematical function, or a complex search process. It is the **core** of an RL agent — alone, it is sufficient to determine behavior.

> **Example — A Chess Opening Book:** A chess player's opening policy might be: *"If the opponent plays 1.d4, respond with the King's Indian Defense."* This maps board positions (states) to moves (actions) without any computation in the moment. The policy guides behavior; how good the policy *is* depends on what comes next.

---

### 2 — Reward Signal

A **reward signal** defines the goal. At each time step, the environment sends the agent a single number — the reward. The agent's sole objective is to maximize the **total reward over the long run**.

The reward signal defines what is good and bad in the *immediate* sense. The agent cannot alter the reward process directly — only through its actions, which affect the environment's state and thus future rewards.

> **Example — The Beginner Chess Player:** A beginner might feel rewarded every time they capture an opponent's piece (immediate reward). But capturing a pawn while walking into a checkmate trap is a terrible long-term decision. The reward signal (capturing material) is easy to sense; using it wisely requires understanding long-term value.

---

### 3 — Value Function

Whereas reward signals indicate what is good *right now*, the **value function** specifies what is good *in the long run*.

The **value of a state** is the total amount of reward an agent can expect to accumulate starting from that state, under a given policy:

$$V^\pi(s) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t R_t \;\middle|\; S_0 = s, \pi \right]$$

To understand what this means, let's break down the notation piece by piece:

* **$V^\pi(s)$**: The expected **Value** of being in state **$s$**, assuming the agent strictly follows policy **$\pi$**.
* **$\mathbb{E}$**: The **Expected Value**. Because environments can be unpredictable, we calculate the mathematical average of all possible future outcomes.
* **$\sum_{t=0}^{\infty}$**: The sum of all future steps, from right now (**$t=0$**) until the end of time (**$\infty$**).
* **$\gamma$** (gamma): The **Discount Factor** (a number between 0 and 1). It penalizes delayed rewards, representing the idea that a reward received right now is worth more than a reward received much later.
* **$R_t$**: The immediate **Reward** received at time step **$t$**.
* **$\mid$**: "Given that." It establishes the starting conditions for the calculation.
* **$S_0 = s, \pi$**: The starting conditions: we begin at time zero in our specific state (**$s$**), and all future actions are chosen by our policy (**$\pi$**).

In plain English: *The value of a state is the expected sum of all future discounted rewards, given that we start in that state and follow our current policy.*

> **Example — The Queen's Gambit (Chess):** In the Queen's Gambit opening, White offers a pawn (1.d4 d5 2.c4). Black can capture it, gaining immediate material reward. But White's *value* calculation tells a different story: accepting the gambit hands White rapid development and central control, which translates into a decisive long-term advantage. The captured pawn has high *immediate reward*; the resulting position has low *value*. A strong player sacrifices the immediate reward to pursue states of higher value.

This distinction is critical:

| | Reward | Value |
|---|---|---|
| **Measures** | Immediate goodness | Long-term desirability |
| **Source** | Given directly by the environment | Must be *estimated* from experience |
| **Difficulty** | Easy to observe | Hard to compute accurately |

Rewards are foundational — without them, there could be no values. But **action choices are made based on value judgments**, not immediate rewards alone. Estimating values efficiently is arguably the most important problem in all of RL.

> **Another Example — n-Armed Bandit:** In the bandit setting, value is simply the expected reward of each arm. A greedy agent exploits the arm with highest *estimated* value. But the estimated value is only as good as the agent's exploration history — arms pulled few times have unreliable estimates. The bandit's challenge is precisely this: values are uncertain and must be refined through interaction.

---

### 4 — Model of the Environment *(optional)*

A **model** mimics the environment's behavior, allowing the agent to make inferences about what will happen next given a state and action. Models are used for **planning** — deciding on a course of action by simulating possible futures before they are experienced.

- **Model-based methods** use a model to plan ahead
- **Model-free methods** learn directly from trial-and-error, without an internal model

> **Example — Chess Engine vs. Chess Intuition:** A chess engine like Stockfish uses a model of the game (complete knowledge of rules) to search millions of positions ahead — classic model-based planning. A human grandmaster, by contrast, relies on pattern recognition and intuition to evaluate positions without exhaustive search — closer to model-free RL. Both can play at a high level, through very different mechanisms.

---

## Putting It All Together

With these four subelements — **policy, reward signal, value function, and model** — we have the vocabulary to formalize RL mathematically. The next chapter introduces Markov Decision Processes (MDPs), which provide the rigorous mathematical framework for everything that follows.

The central insight of this chapter is worth stating plainly:

> *Reinforcement learning is distinguished from other computational approaches by its emphasis on learning by an agent from direct interaction with its environment, without relying on exemplary supervision or complete models of the environment.* — Sutton & Barto, 2018

The exploration–exploitation trade-off, the distinction between reward and value, and the role of policy as the behavioral core of an agent are themes that will recur throughout this seminar.

---



## Citations
* Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). MIT Press.
* Hess, Shervin. "Speke's Gazelle Juliet Runs in the Africa Savanna Habitat." The Oregonian/OregonLive, Oregon Zoo, <https://www.oregonlive.com/living/2016/04/baby_gazelle_that_nearly_died.html>
* GeeksforGeeks. "Supervised Machine Learning" GeeksforGeeks, 14 Apr. 2026, <https://www.geeksforgeeks.org/machine-learning/supervised-machine-learning/>.
* Hosseini, Parsa, et al. "Multimodal Analysis in Biomedicine." 28 Jan. 2019.
---
To cite this, please use the following bibtex:

```bibtex
@misc{Abushanab_2026_ReinforcementLearning,
  author       = {Omar Abu Shanab},
  title        = {Reinforcement Learning: A Gentle Introduction, Chapter 1},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}}
}
