# RL Algorithm Overview

## Q-Learning

### Overview
- Model free, off-policy temporal difference (TD) algorithm
- Agent does not know the state transition $P(s'|s, a)$ or reward probability $R(s,a)$ - i.e. it doesn't know the MDP
- **Deterministic** - requires an exploration strategy, otherwise will act greedily
- **Value-based method** 
    - Constantly  improves estimates of state-action values pairs $Q(s,a)$
    - Implicitly improves the policy
- **Discrete observations & actions**
- For **any finite MDP** Q-learning will **eventually find an optimal policy**
    - Learns the value $Q(s,a)$ of being in a given state $s$ and taking a specific action $a$
    - Learns long-term expected rewards

### Approach
- Iteratively approximates Q-Function using Bellman's equations
- Learned Q-function directly approximates optimal $q^*$ function, independently of the policy being followed (off-policy)

$$Q(S_t, A_t ) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_z Q(S_{t+1},a) - Q(S_t, A_t) \right]$$


## Deep Q-Network
