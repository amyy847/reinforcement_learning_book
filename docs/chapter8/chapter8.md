# Chapter 8.1 - Introduction to Deep Reinforcement Learning

**Author:** [Abdelrahman Hamada](https://www.linkedin.com/in/abdelrahman-hamada-1202a71b3/)

---

In the previous chapters we were using tabular methods to store our expected rewards, either if it was for individual states or for every state-action pairs. Now imagine you are trying to play a video game, where every frame of pixels on your screen is a state. The number of possible frames is astronomically large, far beyond what any table could store. Even simpler environments, like a robot navigating a continuous 3D space, have infinitely many possible states. A table would need an infinite number of rows.

This is where tabular methods break down completely, they cannot scale to problems where the state space is large or continuous. What we need instead is something that can generalize, a function that, given a state it has never seen before, can still produce a reasonable estimate of its value. This is exactly what function approximation gives us, and non-linear function approximators like neural networks are our most powerful tool for it.

We will start with value-based methods:

## 8.1.1 Deep Q Learning

<!-- TODO: NOT DONE and written really badly -->

A quick recap on Q-learning, so in Q-learning we try to learn $Q^*$, the optimal quality function, and if we visit every state-action pairs infinite many times, eventually we will converge to $Q^*$.

$$Q(s, a)_{k+1} = Q_k(s, a) + \alpha[r_{t+1} + \gamma \max_{a'} Q(s', a') - Q_k(s, a)]$$

Q-learning is an off-policy learning algorithm since we use $\epsilon$-greedy methods for data generation while the policy that we will be using after training will be:

$$\pi(s) = \arg\max_a Q(s, a)$$

<!-- TODO: NOT DONE and maybe talk about a known game, not just numbers -->

As the state space grows, tabular Q-learning becomes computationally impossible. Consider a state space of $10^6$ states with 10 actions, the table alone requires $10^7$ entries, and convergence requires every one of those entries to be visited infinite many times. In practice, most states are never visited at all, making no convergence guarantees.

<!-- TODO: NOT DONE and be more precise about are we changing the values of s and a, not s only  -->
<!-- TODO: put a link to the last chapter maybe ?-->

The problem with tabular Q-learning is that we are learning each state individually, and modifying the value of a state-action pair $s, a$ would never change the value any other state-action pair $s', a'$ in any way. When using function approximators changing a parameter in the function changes the function landscape, leading to the modification of multiple state values. In the last chapter we approximated our $Q$ function using linear approximators, but still linear approximators encode the states using feature vectors that are designed by us rather than learned from experience.

<p align="center">
    <img src="../assets/continuous_q_function.png" alt="cq" width="400"/>
    <img src="../assets/tabular_q_function.png" alt="tq" width="400"/>
</p>

As it can be seen in the above figure, improving a single state creates a bump in the landspace of the continuous function, but improving a single state in the tabular function only improves the state itself and does not touch any neighbouring states.

The usage of deep neural nets here is our best option, and that is exactly how in deep Q-learning we approximate our Q function, deep neural learns the feature representation of a state by its own,

### 8.1.1.1 Playing Atari with Deep RL

Consider an example where we are trying to train an agent to play the atari game breakout, our state here is the current image of the game, or to be more precise, our state is last 4 frames of the games. The reason for selecting the last 4 states of the game is to let our state have the markovian property, we want to decide our next action from our current state only, and using a single image would not give us enough information on how to act.

Since we are using the last 4 frames, our state spacee would be extremely huge, so using tabular methods is just impossible, instead we will be using a deep neural network, so now our $Q$ function will be parameterized by some weights $\theta$. Our goal becomes making our $Q$ function eventually learn the optimal $Q$ function.

$$Q(s, a; \theta) \approx Q^*(s, a)$$

<!-- TODO: proof read, this is badly written -->
<!-- TODO: try to link with the linear approximators chapers and use their symbols -->

<p align="center">
    <img src="../assets/cnn.png" alt="cnn"/>
</p>

In the early layers we will be using a convolutional neural network, this network extracts the important features from our current state, so instead of passing the whole complete image of $84 * 84$ pixels for the last 4 frames as an input for the fully connected layers, the CNN learns to extract the important features needed from that image into a meaningful vector representation that the later layers can make the best use of. Another way to look at the convolutional layers is that they are learning the correct state vector representation for the state analogous to the feature vectors we hand-crafted in linear function approximators. This feature vector is then passed to multiple fully connected layers that have RELU activation function between them and finally in the end we compute the value for every possible action.

Our correct target is:

$$y = \mathbb{E}_{s, r \sim p}[r_{t+1} + \gamma \max_a Q(s, a;\theta^-)]$$

We won't be calculating the complete expectation over the next states, instead we will be using a noisy single point estimate, this will work because this point estimate is unbiased, so after vising this target for many times, the noise will eventually get averaged out. Our final target will be:

$$ y = r_{t+1} + \gamma \max_a Q(s, a;\theta^-) $$

<!-- Write about how does normal ML differ from Deep RL in the idea of the target  -->

The main difference from supervised ML is that here our target is not fixed, we don't have a true value that we are trying to converge, instead we have a moving target that is changing every update of the weight, and this happens because in our target, we are using the same neural network for TD bootstrapping and the network itself changes during training leading to the change of the target itself, so what we are training towards is always changing.

Our objective function becomes:

$$L(\theta) = \mathbb{E}[(y - Q(s, a; \theta))^2]$$

Since we have only experience points from playing the game, the expectation can be approximated by averaging over all data points

$$
\begin{align}
L(\theta) &= \mathbb{E}[(y - Q(s, a; \theta))^2] \\
&= \frac{1}{N}\sum^{N}_{i=1}(y_i - Q(s_i, a_i; \theta))^2
\end{align}
$$

And we can note that just the mean squared error from machine learning. The complete expectation is expensive to compute, so instead we will be optimizing with stochastic gradient descent.

### 8.1.1.2 The Moving Target Problem

Note that in the target we were using $\theta^-$ and in the online network we were just using $\theta$, the presence of two different parameters is to just to freeze our evaluation network for some steps during training. If we used the same network paramters for evaluation, we would never converge, because it will be always changing.

Imagine in the last iteration we changed our network parameters to move $Q(s, a)$  towards a target, but as we know changing the parameters of the network also affects other states, which may affect $Q(s', a')$ that was used in computing that target. So the next time we train on $(s, a)$, the target has already shifted, not because the environment changed, but because *we* changed it by updating the network. Unlike supervised learning where labels are fixed, here the label is generated by the same network we are updating, so every step we take moves the ground beneath our feet. It feels like you are always running after something, or as I like to call it, _you are running after your shadow_.

So we decide to freeze our evaluation network for some $C$ steps and use it as a fixed target, after the $C$ steps we set $\theta^- = \theta_i$ and freeze again for $C$ steps.

> Think about it: if we set $\theta^- = \theta_{i-1}$ where $\theta_i$ are the weights of the current iteration, what would this algorithm reduce to ?

### 8.1.1.3 Replay Buffer

The agent is not trained on sequential steps in the same episode, this causes the network to overfit to the plays that is optimal to what it has been learning in the current episode, rather than generalizing across the broader state-action space. Consecutive transitions share the same properties so gradient updates become highly correlated and push the network weights in a narrow direction.

Another way to look at this is that to image you are training a normal neural network that does classification for cat and dog images, but instead of shuffling your mini-batch, you just select full batch of cat samples, here your network will just optimize for outputting cats whatever the sample, and your training won't converge and you will be forgetting your previous training. The same thing could happen in DQN, the network will just overfit to the current path, forgetting what it has learned before.

During training we sample mini-batches from this replay buffer, we consider the replay buffer as a probability distribution over $(s, a, r, s')$ tuples that were experienced during game playing. In normal DQN the probability distribution is a uniform distribution, this is not always the case though as we will see later in prioritized experience replay

### 8.1.1.4 The Final Full Algorithm
```
────────────────────────────────────────────────
INITIALIZE:
  D        ← replay memory with capacity N
  Q        ← action-value network with random weights θ
  Q_hat    ← target network with weights θ⁻ = θ

────────────────────────────────────────────────
FOR episode = 1 to M DO:

  s₁ ← initial sequence {x₁}
  φ₁ ← preprocess(s₁)

  FOR t = 1 to T DO:

    WITH probability ε:
      aₜ ← random action
    OTHERWISE:
      aₜ ← argmax_a Q(φ(sₜ), a; θ)

    // --- Environment Step ---
    Execute aₜ
    Observe reward rₜ and next image xₜ₊₁
    sₜ₊₁ ← {sₜ, aₜ, xₜ₊₁}
    φₜ₊₁ ← preprocess(sₜ₊₁)
    Store (φₜ, aₜ, rₜ, φₜ₊₁) in D

    Sample random batch {(φⱼ, aⱼ, rⱼ, φⱼ₊₁)} from D
    FOR each sampled transition j:
      IF episode terminates at step j+1:
        yⱼ ← rⱼ
      ELSE:
        yⱼ ← rⱼ + γ · max_a' Q(φⱼ₊₁, a'; θ⁻)

    Minimize loss: L = (yⱼ - Q(φⱼ, aⱼ; θ))²
    Update θ via gradient descent on L
    EVERY C steps:
      θ⁻ ← θ

  END FOR
END FOR
```
> Note: $\phi(s)$ here means applying preprocessing to our state frames, the original frame in the image is $210 * 160$ RGB image, in this algorithm it scaled down to $84 * 84$ as we mentioned in DQN part and converted to a grayscale image.

## 8.1.2 Double DQN

<!--  note that this can be proved by jensens inequality but i decided not to mention it here because seeing it for the first time might be confusing -->

In DQN, we were using in our target $\max_a Q(s', a)$ to select the best action that could be taken in $s'$ and evaluate it, so we are using the same network for selection and evaluation. The problem is that our network is not free from noise, so it may select action that is overestimated due to noise, which causes the value of the action that was selected in the current state to be pushed towards an overestimated value, and this overestimated value will cause more overestimation in later evuations and so on.


<p align="center">
    <img src="../assets/ddqn.drawio.png" width="230"/>
</p>

So instead of using the same network to select the maximum action and evalute it we can decouple action selection and evaluation. We will use the current network we are training on to select, which is done using $\arg \max_a Q(s', a)$ and then we pass this action to be evalatued by another network. The other network is usually the same network but with weights freezed from the last 100 steps for example.

So finally our target becomes:

$$y = \mathbb{E}_{s, r \sim p}[r_{t+1} + \gamma Q(s, \arg \max_a Q(s, a; \theta);\theta^-)]$$

A different way to look at this that the two networks have _"different opinions"_, both are not correct, but selecting the best according to one network and evaluating with another averages the bias out and the network converges better.

## 8.1.3 Dueling Networks

Sometimes it does not matter what action you take in some state, you will always get the same value. This could be seen in racing games states where you are driving in a wide straight track, moving a bit right or a bit left won't matter, you will always get the same value. This value is in the state itself, not in the action taken.

For a normal deep Q-network learn this, it will have to see that every action possible for this state gives the same value, it will have to see a lot of samples.

A better way to do this is to first, define an advantage function, and it is defines as how much value does taking this action add to the value of the current state. It is defined mathematically as:

$$A(s, a) = Q(s, a) - V(s)$$

And we are going to change our neural network architecture to the following: start with a same CNN used in the DQN to exract our state features, but instead of passing all the features to a single fully-connected layer that predicts $Q(s, a)$, the network will be splitted into seperated heads, one will predict our advantage function $A(s, a)$ and the other will predict the value of our current state $V(s)$. The outputs of both is then passed to an aggregation node that calculates $Q(s, a) = A(s, a) + V(s)$

### 8.1.3.1 The Identifiability Problem

The last aggregation node does not just calculate $Q(s, a) = A(s, a) + V(s)$, because there is nothing in this equation constraints the advantage network to learn the advantage function only and the value network to learn the value function only. What could happen is that the advantage network for example learns $A(s, a) + k$ and the value learns $V(s) - k$ and adding them together would give us 

$$
\begin{align}
Q(s, a) &= A(s, a) + k + V(s) - k \\
&= A(s, a) + V(s)
\end{align}
$$

And we can see that we got $Q(s, a)$ but the two networks did not learn their correct functions.

To combat this, we make our last node calculate:

$$ Q(s, a) = V(s) + A(s, a) - \frac{1}{|A|}\sum_a A(s, a) $$

This forces the value network to learn the correct $V(s)$ and not learn it with an offset, because now if it learned the advantage funcion with an offset and the advantage network tried to compensate, then our advantage addition becomes

$$
\begin{align}
&A(s, a) - k - \frac{1}{|A|}\sum_{a'} [A(s, a) - k] \\
&A(s, a) - k - \frac{1}{|A|}\sum_{a'} A(s, a') + k \\
&Which\ gets\ us\ back\ to \\
&A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')
\end{align}
$$

We can see that the k can't hide in our A anymore (the mean subtraction removes it), the only place an offset can exist is in $V(s)$. And since $V(s)$ is directly supervised by the Bellman targets during training, any offset in it gets corrected by the loss. So the network is forced to learn a meaningful $V(s)$.

## 8.1.4 Prioritized replay

Instead of sampling uniformally form the experience buffer, we will give some experiences more importance than the other during sampling. The importance is given by the TD error, the higher the error the more importance the sample has, i.e. more priority goes to it to be learned

Each tuple $(s, a, r, s')$ will be assigned an error of:

$$ \delta =  r_{t+1} + Q(s', a') - Q(s, a)$$

Then we assign the tuple a probility that is defined by:

$$ P(i) = \frac{p^\alpha_i}{\sum p^\alpha_j} $$

The $\alpha$ parameter lets choose how close is this distribution to a uniform distribution, the lower the $\alpha$ the lower it will look look like a uniform distribution with $\alpha = 0$ collapses to a fully uniform distribution.

and $p$ is defined as:

$$ p_i = |\delta| + \epsilon $$

The $\epsilon$ is added to still train on experiences that are already learned and to prevent forgetting them.

Lastly, we sample from the experience using the above probabilities.

The main goal of Prioritized replay and dueling networks is to acheieve sampling efficieny, which means that we get more information or more learning from less data. And we can see that in dueling networks because here $V(s)$ gets updated in for every action, while in normal deep Q-learning only the specific $Q(s, a)$ gets updated

## 8.1.5 Summary

All the architectures discussed above are improvements for the architecture before it, they all attack the weaknesses in the architecture before it. We can see that double DQN fixes the overestimation in vanilla DQN, dueling networks improves the architecture to allow for more efficient learning. Prioritized experience buffer adds different weight for different experience samples.

A useful way to think about the progression is in terms of what each method 
improves:

| Method | What it fixes |
|---|---|
| DQN | Scales Q-learning to high-dimensional state spaces |
| Double DQN | Fixes overestimation bias in the target |
| Dueling Networks | Decouples state value from action advantage for faster learning |
| Prioritized Replay | Focuses training on the most informative experiences |

### References

1. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, *518*(7540), 529–533. https://doi.org/10.1038/nature14236

2. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. In *Proceedings of the AAAI Conference on Artificial Intelligence*. arXiv:1509.06461

3. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

4. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. In *Proceedings of the 33rd International Conference on Machine Learning (ICML)*. arXiv:1511.06581

---

_To cite this chapter, please use the following:_

```bibtex
@misc{hamada_2026_ReinforcementLearning,
  author       = {Abdelrahman Hamada},
  title        = {Reinforcement Learning: A Gentle Introduction, Chapter 8.1},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/amrmsab/reinforcement_learning_book}},
  note         = {Accessed: April 30, 2026}
}
```
