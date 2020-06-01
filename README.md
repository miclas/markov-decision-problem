# markov-decision-process
## Task
The goal is to compute solutions to simple Markov decision problems (MDP). The agent operates in a flat grid world of size NxM using horizontal and vertical moves, with single field steps, at each step receiving the reward specific to the state she entered. In this world the fields are called states: unique start state (S), terminal states (T), the walls and prohibited states (F), and the special states (B). The agent starts in the starting state (S) and stops upon entering a terminal state (T). She can never traverse a wall or step into a prohibited state (F). All unmarked states are normal states, with the default reward value. The rewards received in special (B) and terminal (T) states are individually defined for each state. 

Example worlds:

![alt text](https://github.com/miclas/markov-decision-process/blob/master/screenshots/ss.PNG)

The rules:

a) The agent can move: (L)eft, R(ight), (U)p, i D(own). Executing a move in any direction transfers the agent to the intended state             with probability p1, with probability p2 it transfers the agent to the state left of the action's origin, with probability p3 it transfers the agent to the state right of the action's origin, and with probability (1-p1-p2-p3) it transfers the agent to the    state opposite of the intended state (for exemple p1=0.8 and p2=p3=0.1).

b) When the agent executes a move outside the wall, or into a forbidden state, then as the result of such move she stays in place (and receives the appropriate reward). It is as if she bounced off the walls and forbidden states. 

c) Each move causes the agent to receive a reward (if negative, it can be considered to be the cost of a move). The value of the reward depends on the state the agent has just entered. For normal states the standard reward value r applies. For the terminal and special states the rewards are defined individually for each such state. 

d) The result of the agent's activity is the discounted sum of the rewards from all steps, computed using the discounting factor γ.

The parameters of the problem are thus: N,M,p1,p2,p3,r,γ and the individual reward values for all the terminal and special states.
The aim is to find the optimal agent policy, achieving the maximum expected value of the sum of rewards received. 

All parameters are set in the data.json file

### Example

![alt text](https://github.com/miclas/markov-decision-process/blob/master/screenshots/ss1.PNG)

