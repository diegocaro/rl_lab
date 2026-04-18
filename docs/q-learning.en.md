# Q-Learning

Q-learning is a model-free, off-policy reinforcement learning algorithm that enables agents to learn optimal actions through trial-and-error, maximizing future rewards without needing a model of the environment.

## Core concepts

- **Agent & environment:** the agent interacts with the environment by choosing actions and receiving rewards.
- **Q-table:** the agent's memory — stores the expected value for each (state $s$, action $a$) pair.
- **Q-function:** $Q(s, a)$ represents the total expected reward for taking action $a$ in state $s$.
- **Bellman equation:** the central formula for updating Q-values based on the immediate reward and the best possible future value:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

where:

- $\alpha$ — learning rate: how much weight new information gets over old.
- $\gamma$ — discount factor: importance of future rewards (0 = only the present matters, 1 = future equally important).
- $r$ — immediate reward received after taking action $a$.
- $Q(s', a')$ — the row in the Q-table for the next state $s'$, the state the agent lands in after executing $a$. $\max_{a'} Q(s', a')$ picks the best action available from that state according to current knowledge.

That last point is what makes Q-learning *off-policy*: even if the agent won't necessarily pick the optimal action in $s'$ (it may explore), it still uses it as the reference to update the current value. It learns the optimal policy regardless of how it's actually behaving at that moment.

## Training process

1. Initialize the Q-table to zero.
2. Choose an action using $\varepsilon$-greedy strategy (occasional random exploration).
3. Execute the action, receive reward $r$, and observe the new state $s'$.
4. Update the Q-table using the Bellman equation.
5. Repeat until the policy converges.

## Key characteristics

- *Model-free:* learns purely from experience, no environment model needed.
- *Off-policy:* learns the optimal policy regardless of the agent's current behavior.
- *Discrete action space:* works best with finite sets of states and actions.
- *Limitation:* becomes inefficient with very large state spaces; those cases call for [Deep Q-Networks (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602).

## References

- Watkins, C.J.C.H. & Dayan, P. (1992). [Q-learning](https://link.springer.com/article/10.1007/BF00992698). *Machine Learning*, 8, 279–292. — original Q-learning paper.
- Bellman, R. (1957). *Dynamic Programming*. Princeton University Press. — origin of the Bellman equation.
- Sutton, R.S. & Barto, A.G. (2018). [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf). MIT Press. — the standard reference textbook.
- Mnih, V. et al. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602). DeepMind. — introduced Deep Q-Networks.
