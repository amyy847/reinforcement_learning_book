# Chapter 3: Dynamic Programming
#### Author: Mohammad Tarek Wahby

---
## Table of Contents
1. [Introduction to Dynamic Programming in Reinforcement Learning](#Section-31-Introduction-to-Dynamic-Programming-in-Reinforcement-Learning)
   1. [What is Dynamic Programming?](#311-What-is-Dynamic-Programming)
   2. [What is Reinforcement Learning?](#312-What-is-Reinforcement-Learning)
   3. [The RL Problem as a Markov Decision Process](#313-The-RL-Problem-as-a-Markov-Decision-Process)
   4. [Where Does Dynamic Programming Come In?](#314-Where-Does-Dynamic-Programming-Come-In)
2. [Value Functions and the Bellman Equation](#Section-32-Value-Functions-and-the-Bellman-Equation)
   1. [How Does an Agent Know If It Is Doing Well?](#321-How-Does-an-Agent-Know-If-It-Is-Doing-Well)
   2. [The State-Value Function V(s)](#322-The-State-Value-Function-Vs)
   3. [The Action-Value Function Q(s, a)](#323-The-Action-Value-Function-Qs-a)
   4. [The Bellman Equation](#324-The-Bellman-Equation) 
   5. [Bootstrapping — Learning From Your Own Estimates](#325-Bootstrapping--Learning-From-Your-Own-Estimates)
   6. [From Values to Decisions — The Road Ahead](#326-From-Values-to-Decisions--The-Road-Ahead)
3. [The Core DP Algorithms](#Section-3-The-Core-DP-Algorithms) 
   1. [Putting the Bellman Equation to Work](#331-Putting-the-Bellman-Equation-to-Work)
   2. [Policy Evaluation](#332-policy-evaluation)
   3. [Policy Improvement](#333-Policy-Improvement)
   4. [Policy Iteration](#334-Policy-Iteration)
   5. [Value Iteration](#335-Value-Iteration)
   6. [Key Concepts and Limitations of DP](#336-key-concepts-and-limitations-of-dp)

4. [Summary and Key Takeaways](#Section-4-Summary-and-Key-Takeaways)
   1. [Summary](#341-Summary)
   2. [The Four Algorithms in One Picture](#342-The-Four-Algorithms-in-One-Picture)
   3. [What DP Cannot Do](#343-What-DP-Cannot-Do)
   4. [Fundamental Takeaway](#344-Fundamental-Takeaway)
5. [References](#References)
6. [Summary of Key Terms](#Summary-of-Key-Terms)
7. [Citiation](#Citiation)

---
# Section 3.1: Introduction to Dynamic Programming in Reinforcement Learning

## 3.1.1 What is Dynamic Programming?

Dynamic Programming (DP) is an algorithmic technique used for solving complex problems
by breaking them into smaller, overlapping, simpler subproblems. Instead of solving the
same subproblem over and over again, DP solves it once, stores the result, and reuses
it whenever needed

Think of it like planning a road trip. Instead of recalculating the best route from
every single city from scratch, you remember the best route from each city you have
already figured out and build on that knowledge. DP does exactly the same thing —
it builds solutions step by step, using what it has already learned.

![Dynamic Programming.png](Dynamic%20Programming.png)

**Figure 3.1.1**-*A simple representation of Dynamic Programming.
The problem is represented by the blue color and the result stored is represented by the red color.*

## 3.1.2 What is Reinforcement Learning?

Before we can understand how DP fits into Reinforcement Learning, we need to understand
what Reinforcement Learning (RL) actually is.

In RL, we have two main characters:

- **The Agent** — the decision-maker. This could be a robot, a game-playing AI, or
  any system that takes actions.
- **The Environment** — everything the agent interacts with. The world around it.

At every moment in time, the agent looks at the world and sees the current **state** —
a description of the situation it finds itself in. Based on that state, the agent
chooses an **action**. The environment then responds by giving the agent a **reward**
— a simple score that tells the agent how well it did — and moves into a new state.

The agent's goal is simple: **collect as much reward as possible over time.**

![mdp-cycle.png](mdp-cycle.png)
**Figure 3.1.2**- *The agent and its interaction with the environment*

The agent's strategy for choosing actions is called a **policy** ($\pi$). A policy is simply
a rule that says: *"When I am in this state, I will take this action."* The ultimate
goal of RL is to find the **optimal policy** — the strategy that earns the most
reward in the long run.

## 3.1.3 The RL Problem as a Markov Decision Process

For DP to work within RL, we need to describe the RL problem in a precise, mathematical
way. This is where the **Markov Decision Process (MDP)** comes in.

An MDP is simply a formal way of saying: *"The world has states, the agent takes
actions, and those actions lead to new states and rewards."* It relies on one
key assumption known as **the Markov Property**:

> *"The future depends only on where you are right now — not on how you got there."*

In other words, the current state contains all the information the agent will ever
need to make a good decision. The full history does not matter.

An MDP is made up of five core components:

| Component | Symbol | What it means |
|---|---|---|
| States | S | All possible situations the agent can be in |
| Actions | A | Everything the agent can do |
| Transition Dynamics | P(s'\|s,a) | The probability of moving to a new state |
| Reward Function | R(s,a,s') | The reward received after each transition |
| Discount Factor | γ (gamma) | How much the agent values future rewards vs. immediate ones |

The **discount factor γ** deserves a special mention. A value of γ close to 0 means
the agent is short-sighted — it only cares about immediate reward. A value close to
1 means the agent is patient — it values future rewards almost as much as immediate
ones.

## 3.1.4 Where Does Dynamic Programming Come In?

Now that we understand the RL problem, we can understand exactly where DP enters
the picture.

Dynamic Programming in RL refers to a collection of algorithms that can compute
**optimal policies** — but only when given a **perfect model of the environment**
in the form of an MDP. In other words, DP assumes we already know:

- Every possible state
- Every possible action
- The exact probabilities of transitioning between states
- The exact reward at each step

This is a strong assumption. In the real world, we rarely have this level of perfect
knowledge. However, this is also precisely what makes DP so valuable as a **foundation**:
it shows us exactly what optimal behaviour looks like when we have all the information
we could ever want. Every more advanced RL method that comes after DP is essentially
trying to achieve the same result **without** needing that perfect model.

![Dynamic Programming Comparison.jpg](Dynamic%20Programming%20Comparison.jpg)

**Figure 3.1.3**-*Dynamic Programming Assumption vs Real Life Scenario*

The way DP searches for the optimal policy is through **value functions** — estimates
of how good it is to be in a particular state. By applying the **Bellman Equation**
repeatedly, DP refines these estimates until they converge on the true optimal values.
We will explore value functions and the Bellman Equation in detail in the next section.

# Section 3.2: Value Functions and the Bellman Equation
## 3.2.1 How Does an Agent Know If It Is Doing Well?

Recall our road trip analogy from Section 1. The agent moves through the world,
collecting rewards along the way. But here is an important question: how does the
agent decide which state is *worth being in*?

Imagine you are playing chess. You could win $100 if you win the game. At some point
mid-game, you have two possible moves. One move feels safe but slow. The other is
risky but puts you in a dominant position. Which do you choose?

To answer that, you are not just thinking about the immediate move — you are thinking
about **how good your overall position is from here.** You are estimating the value
of being in a certain state.

This is exactly what a **value function** does.

> A **value function** tells the agent how good it is to be in a particular state —
> not just right now, but considering all the future rewards it can expect from that
> point onward.


---

## 3.2.2 The State-Value Function V(s)

Formally, the **state-value function**, written as **V(s)**, answers the question:

$$V_\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'}
P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_\pi(s') \right]$$


> *"If I am in state s and I follow my current policy π from here, how much total
> reward can I expect to collect?"*

The key word here is **expected** — because the environment may be uncertain, we
think in terms of averages. We also factor in the discount factor γ from Section 1,
meaning rewards collected sooner are worth more than rewards collected far into
the future.

So V(s) is not just about the next reward. It is the **sum of all future rewards**,
discounted over time, that the agent expects to receive by following its policy
from state s onward.


---
## 3.2.3 The Action-Value Function Q(s, a)

There is a closely related function that turns out to be just as important — the
**action-value function**, written as **Q(s, a)**.

While V(s) asks *"how good is this state?"*, Q(s, a) asks a more specific question:

$$Q_\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') +
\gamma \sum_{a'} \pi(a' \mid s') Q_\pi(s', a') \right]$$


> *"If I am in state s, I take action a right now, and then follow my policy
> from there — how much total reward can I expect?"*

The difference is subtle but powerful. Q(s, a) lets the agent compare its options
directly. Instead of just knowing that a state is good, the agent can ask: *"Which
specific action leads to the best outcome from here?"*

Think of it this way — V(s) tells you how good a city is to be in. Q(s, a) tells
you how good each road out of that city is. If you know Q(s, a) for every action,
choosing the best move becomes trivial: just pick the action with the highest Q value.

---

## 3.2.4 The Bellman Equation

Now we arrive at the idea that makes Dynamic Programming possible. We know that
value functions measure long-term reward — but how do we actually *calculate* them?

Here is the key insight:

> **The value of being in a state right now is equal to the immediate reward you
> receive, plus the discounted value of wherever you end up next.**

This recursive relationship is called the **Bellman Equation**, named after
mathematician Richard Bellman who formalised it in the 1950s.

Written simply, it says:

V(s) = (Immediate Reward) + γ × V(next state)

This is the bridge that connects the present to the future. Rather than trying to
calculate the total reward of an entire journey all at once, the Bellman Equation
lets us break the problem into one step at a time — which is precisely the
*"breaking complex problems into simpler subproblems"* idea.

The Bellman Equation applies not just to V(s), but to Q(s, a) as well. In both
cases, the core idea is the same: **the value of where you are now is anchored to
the value of where you are going.*


---
## 3.2.5 Bootstrapping — Learning From Your Own Estimates

There is one more concept worth naming here, because it defines what makes DP
distinctly powerful — and distinctly limited.

Notice that the Bellman Equation computes V(s) using V(next state). In other words,
it updates an estimate using *another estimate*. This technique is called
**bootstrapping**.

It is a bit like pulling yourself up by your own bootstraps — you are using what
you already believe to be true to refine what you believe. It sounds circular, but
it works remarkably well in practice. As long as the agent keeps applying the
Bellman Equation repeatedly across all states, the estimates gradually converge
toward the true values.

This is the computational heartbeat of Dynamic Programming. Every DP algorithm
you will encounter is, at its core, repeatedly applying the Bellman Equation until
the value estimates stop changing.

## 3.2.6 From Values to Decisions — The Road Ahead

We now have all the ingredients we need:

- A way to **measure how good a state is** → the value function V(s)
- A way to **measure how good an action is** → the action-value function Q(s, a)
- A way to **compute those values recursively** → the Bellman Equation
- A way to **refine estimates using other estimates** → bootstrapping

There is one question left: how does the agent actually *use* all of this to
find the best policy?

The answer comes in three tightly connected algorithms. 

The first asks:
*"Given a fixed policy, how good is it?"* — that is **Policy Evaluation.**


The second asks: *"Given those values, can we do better?"* — that is
**Policy Improvement.** Put them together in a loop and you get
**Policy Iteration.**

And if you want to skip straight to the answer as
efficiently as possible, you get **Value Iteration.**

Each of these is a direct application of everything covered in this section.
The Bellman Equation is not just a formula — it is the engine that drives all
four of them. In the next section, we take that engine and put it to work.

---
# Section 3: The Core DP Algorithms

## 3.3.1 Putting the Bellman Equation to Work

In Section 2, we established that the Bellman Equation gives us a recursive way to
measure how good a state is. But knowing the equation and actually *using* it to
find the best policy are two different things.

This section introduces the four core algorithms of Dynamic Programming. Each one
is a direct application of the Bellman Equation, and each one answers a progressively
more ambitious question:

- **Policy Evaluation** — *"How good is my current policy?"*
- **Policy Improvement** — *"Can I do better?"*
- **Policy Iteration** — *"How do I find the best policy by combining the two?"*
- **Value Iteration** — *"How do I find the best policy as efficiently as possible?"*

Think of these four algorithms as a natural progression. You start by measuring,
then improving, then combining, then optimizing. By the end of this section, you
will see how they all connect — and why their limitations point the way toward
every major RL algorithm that comes after them.

## 3.3.2 Policy Evaluation

Before you can improve a policy, you need to know how good it currently is.
**Policy Evaluation** is exactly that measurement process.

Formally, a policy *π(a|s)* is the probability of taking action *a* in state *s*.
Policy Evaluation solves the Bellman Expectation Equation for every state under
that fixed policy:

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V_\pi(s')\right]$$

>For every state, we look at every action the policy might take,
>and every state that action might lead to, and compute the average expected return
>from there. This gives us the true value of every state under the current policy.

### How It Works in Practice — Iterative Policy Evaluation

In most real problems, we cannot solve this equation in one shot. Instead, we use
**Iterative Policy Evaluation** — we start with an arbitrary guess for V(s) (often
just zero for all states), and repeatedly apply the Bellman Equation as an update
rule across every state. With each sweep through the state space, our estimates
get closer to the true values. This process is guaranteed to converge on the
correct V(s) as long as the MDP is finite.

![Figure 3.1 — Policy Evaluation](Policy%20Evaluation.png)
**Figure 3.1**- *Iterative Policy Evaluation sweeps through all states s, updating
each state's value based on the current policy and the values of its successor
states. The process repeats until the values stabilise. Adapted from
Sutton & Barto (2018).*


---

## 3.3.3 Policy Improvement
Once we know how good our current policy is, the natural next question is:
*can we do better?*

**Policy Improvement** answers this by using the value function we just computed
to construct a *greedier* policy — one that always picks the action with the
highest expected return from each state.

The logic is simple. Suppose we are in state *s* and our current policy says to
take action *a*. We can check: is there another action *a'* that leads to a higher
Q(s, a') value? If yes, we update the policy to take *a'* instead. We do this
for every state. The result is a new policy that is guaranteed to be at least as
good as the original — and usually strictly better.

This is formalised by the **Policy Improvement Theorem**: if the new greedy policy
is at least as good as the old one at every state, then it is at least as good
overall. The only time this process stops improving is when the policy is already
optimal — at which point the value function and the policy are perfectly consistent
with each other.

![Figure 3.2 — Policy Improvement](Policy%20Improvement.png)

**Figure 3.2**- *Starting from an arbitrary policy at k=0, Policy Improvement updates
the policy at each iteration. The value ratios across states gradually stabilise,
indicating convergence toward the optimal policy. Adapted from Sutton & Barto (2018).*

---

## 3.3.4 Policy Iteration

Policy Evaluation tells us how good a policy is. Policy Improvement gives us
a better one. The natural question is: why not do both, repeatedly, until we
cannot improve any further?

That is precisely what **Policy Iteration** does.

The algorithm runs as follows:

1. **Initialise** — Start with an arbitrary value function and an arbitrary policy
   for all states.
2. **Evaluate** — Run Policy Evaluation to compute V(s) for the current policy.
3. **Improve** — Run Policy Improvement to produce a new, greedier policy.
4. **Repeat** — Go back to step 2 with the new policy.
5. **Terminate** — Stop when the policy no longer changes between iterations.

![Figure 3.3 — Policy Iteration Steps](Policy%20Iteration%20steps.png)

**Figure 3.3**- *The Policy Iteration loop alternates between full Policy Evaluation
and Policy Improvement until the policy stabilises. Each cycle is guaranteed to
produce a policy no worse than the one before it. Adapted from Sutton & Barto (2018).*

Because the MDP is finite, there are only a finite number of distinct policies.
Since each iteration produces a strictly better policy (or confirms we have
reached optimal), Policy Iteration is **guaranteed to terminate** in a finite
number of steps.

There is, however, one edge case worth mentioning. If two or more policies produce
equally good results, the algorithm may oscillate between them indefinitely rather
than converging. In practice, this is handled by adding a tie-breaking rule — for
instance, keeping the old action if the new one is no better rather than
switching. This small adjustment ensures the algorithm always terminates cleanly.
---

## 3.3.5 Value Iteration

Policy Iteration works, but it carries a hidden cost. Each iteration requires
a **full Policy Evaluation** — meaning many sweeps through the entire state space
just to compute accurate values for one policy, before we can even consider
improving it. In large problems, this is expensive.

There is a smarter way. Instead of waiting for the value function to fully converge
before improving the policy, what if we combined evaluation and improvement into
a single step?

This is the idea behind **Value Iteration**. Rather than averaging over what the
policy would do, Value Iteration simply takes the **best possible action at every
state** in each update:

$$V(s) \leftarrow \max_{a} \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma V(s')\right]$$

The key difference from Policy Evaluation is the **max** operator. Instead of
following a fixed policy, Value Iteration assumes the agent always acts optimally.
Each sweep through the state space is simultaneously an evaluation step and an
improvement step rolled into one.

Value Iteration terminates once the value function changes by only a negligibly
small amount between sweeps — meaning the values have effectively converged.
At that point, the optimal policy can be extracted directly by acting greedily
with respect to the final value function.

![Figure 3.4 — Value Iteration](Value%20Iteration.png)

**Figure 3.4**- *Value Iteration applies a single combined sweep of evaluation and
improvement at each step. The equation selects the highest expected return across
all available actions, moving directly toward the optimal value function. The
algorithm outputs a deterministic policy π(s) upon convergence.
Adapted from Sutton & Barto (2018).*

A useful way to think about the relationship between the two algorithms:

> Policy Iteration = many evaluation sweeps, then one improvement step, repeated.
> 
> Value Iteration = one combined sweep per iteration, straight to convergence.

Value Iteration is generally faster in practice, though both are guaranteed to
find the optimal policy given a perfect model.

---

## 3.3.6 Key Concepts and Limitations of DP

Having covered all four core algorithms, we can now step back and identify what
makes Dynamic Programming powerful as a framework — and where it fundamentally
breaks down.

The use case for the Dynamic Programming methods explained so far is only setback by the size of the entire state set of the MDP. In a very large state set, regular Dynamic Programming may not be efficient computationally. *Asynchronous* DP algorithm helps mitigate this problem. They don't necessarily improve computation, but it creates an algorithm that doesn't get locked in a very long sweep before it can improve the policy.

---
### What Makes DP Work

**Bootstrapping** runs through every DP algorithm we have seen. Rather than
waiting to observe the actual outcome of an entire episode, DP updates its value
estimates using other estimates — the Bellman Equation always references V(s'),
the value of the *next* state, which is itself an estimate. This is why DP can
learn efficiently: it does not wait for the end of the road to update its beliefs.
It updates at every single step.

**Full backups** are another defining trait of DP. At every sweep, every state in
the entire state space is updated using the complete transition model P(s'|s,a).
This is what gives DP its accuracy — no state is ignored, no transition is
approximated. But it is also what makes DP expensive.

### Where DP Breaks Down

**The perfect model assumption** is the first and most fundamental limitation.
Every algorithm in this section assumes we know P(s'|s,a) and R(s,a,s') exactly.
In the real world — a robot navigating a new environment, an algorithm trading in
a live market — this kind of perfect knowledge simply does not exist. DP cannot
be applied directly when the model is unknown.

**The Curse of Dimensionality** is the second, and arguably more severe, limitation.
The computational cost of DP grows exponentially with the number of state variables.
A grid world with 10 cells is trivial. A robot with 10 joints, each at 100 possible
angles, has 100¹⁰ possible states. Performing full backups across a state space
that large is computationally intractable — no amount of hardware can save it.

This is illustrated in the table below:

| State Space Size | Approximate DP Cost |
|---|---|
| 100 states | Trivial |
| 10,000 states | Manageable |
| 1,000,000 states | Slow but feasible |
| 10¹⁰ states and beyond | Computationally intractable |

![Curse of Dimensionality.png](Curse%20of%20Dimensionality.png)

**Figure 3.5**- * Simple visual representation of the Curse of Dimensionality*

### Why DP Still Matters

Given these limitations, one might ask: why study DP at all?

The answer is that DP is not merely a historical curiosity — it is the **theoretical
backbone** of modern reinforcement learning. Every major algorithm that comes after
it, from TD-Learning to Q-Learning to Actor-Critic methods, is essentially an
attempt to achieve what DP achieves, but without requiring a perfect model and
without sweeping the entire state space.

Those methods replace full backups with **sample-based updates** — instead of
considering every possible next state, they observe one transition at a time from
real experience. They replace the known model with **direct interaction** with the
environment. But the objective remains exactly the same: find the value function,
apply something like the Bellman Equation, extract the optimal policy.

Understanding DP deeply means understanding *why* those methods are built the way
they are. The limitations of DP are not dead ends — they are the precise problems
that the rest of reinforcement learning was built to solve.

---
# Section 4: Summary and Key Takeaways
## 3.4.1 Summary

We started with an agent — a decision-maker dropped into an uncertain world,
collecting rewards, trying to figure out the best way to behave. We gave that world
a precise mathematical description through the MDP framework, and we established
that the agent's goal is to find the optimal policy — the strategy that maximises
long-term reward.

We then asked: how does an agent even *measure* how good a situation is? That
question gave us value functions. How does it *compute* those values efficiently?
That gave us the Bellman Equation. And how does it *use* those values to find the
best possible behavior? That gave us the four core algorithms of DP.

---

## 3.4.2 The Four Algorithms in One Picture

Each algorithm answered a progressively sharper question:

| Algorithm | The Question It Answers |
|---|---|
| Policy Evaluation | How good is my current policy? |
| Policy Improvement | Can I construct something better? |
| Policy Iteration | How do I find the optimal policy by repeating both? |
| Value Iteration | How do I get there as efficiently as possible? |

Together, they form a complete toolkit for solving any RL problem — provided you
have a perfect model of the environment. That last condition is both DP's greatest
strength and its defining limitation.

---

## 3.4.3 What DP Cannot Do

Dynamic Programming is theoretically perfect. Given a complete MDP, it will find
the optimal policy. Every time. Without fail.

But the real world does not hand you a perfect model. A self-driving car does not
know the exact probability of every other driver's next move. A trading algorithm
does not know the precise dynamics of a live market. And even if the model were
known, the state space in most real problems is so vast that sweeping through
every single state is computationally impossible — the Curse of Dimensionality
ensures that.

These are not minor inconveniences. They are fundamental barriers. And they are
exactly why the field of reinforcement learning did not stop at Dynamic Programming.

## 3.4.4 Fundamental Takeaway

Here is the most important takeaway of everything you have read:

> Every major reinforcement learning algorithm that exists beyond DP —
> TD-Learning, Q-Learning, SARSA, Actor-Critic — is an answer to the question:
> *"How do we achieve what DP achieves, without its limitations?"*

Those methods replace the perfect model with real experience. They replace full
sweeps of the state space with single sampled transitions. They replace guaranteed
convergence on small problems with practical convergence on large ones. But at
their core, they are all still doing the same thing DP does — estimating value,
applying something like the Bellman Equation, and using that to improve behaviour.

If you understand DP, you do not just understand four algorithms. You understand
the *logic* that underlies the entire field. When you encounter Q-Learning and
see the update rule, you will recognise the Bellman Equation. When you see
bootstrapping in TD-Learning, you will know exactly where that idea came from.
When you understand why model-free methods exist, it is because you understand
what the perfect model assumption costs.


---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. Available at: <http://incompleteideas.net/book/RLbook2020.pdf>
2. van Otterlo, M., & Wiering, M. (2012). Reinforcement Learning and Markov Decision Processes. University of Groningen. Available at: <https://www.ai.rug.nl/~mwiering/Intro_RLBOOK.pdf>

---
### Summary of Key Terms

| Term | Simple Definition |
|---|---|
| Agent | The decision-maker in RL |
| Environment | The world the agent interacts with |
| State (S) | A description of the current situation |
| Action (A) | A choice the agent can make |
| Reward (R) | Feedback signal telling the agent how well it did |
| Policy (π) | The agent's strategy for choosing actions |
| MDP | The mathematical framework that formalises the RL problem |
| Markov Property | The future depends only on the current state |
| Discount Factor (γ) | How much the agent values future vs. immediate rewards |
| Dynamic Programming | Algorithms that compute optimal policies given a perfect model |
| Value Function V(s) | How much total future reward the agent expects from state s |
| Action-Value Function Q(s,a) | How much total future reward the agent expects from taking action a in state s |
| Bellman Equation | The recursive formula linking a state's value to its successor's value |
| Bootstrapping | Updating estimates using other estimates rather than final outcomes |
| Policy Evaluation | Computing the value function for a fixed policy |
| Policy Improvement | Constructing a better policy using the current value function |
| Policy Iteration | Alternating between evaluation and improvement until convergence |
| Value Iteration | Combining evaluation and improvement into a single sweep per iteration |
| Full Backup | Updating every state using the complete transition model |
| Bootstrapping | Updating value estimates using other value estimates |
| Curse of Dimensionality | Exponential growth in computation as the state space grows |
| Perfect Model Assumption | DP requires complete knowledge of transitions and rewards |
---
## Citiation

To cite this, please use the following bibtex:

```bibtex
@misc{yourlastname_2026_ReinforcementLearning,
  author       = {Mohammad Tarek Wahby},
  title        = {Reinforcement Learning: A Gentle Introduction, Chapter #3: Dynamic Programming},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}},
  note         = {Accessed: April 30, 2026}
}