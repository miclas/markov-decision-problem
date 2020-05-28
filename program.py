import numpy as np
import json



with open("dane.json", "r") as read_file:
    data = json.load(read_file)
data


class MarkovDecisionProcess:
    def __init__(self, *, N, M, S, T, F, B, p1, p2, p3, r, r_T, r_B, gamma):
        self.N, self.M = N, M
        self.S = S
        self.T = self.lists_to_tuples(T)
        self.F = self.lists_to_tuples(F)
        self.B = self.lists_to_tuples(B)
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.r, self.r_T, self.r_B = r, r_T, r_B
        self.gamma = gamma

    def lists_to_tuples(self, I):
        return [(a, b) for a, b in I]


mdp = MarkovDecisionProcess(**data['world']['size'],
                            **data['world']['states'],
                            **data['transition_rates'],
                            **data['reward_function'],
                            gamma=data['gamma'])

states = {(x,y) for x in range(1, mdp.N + 1)
                for y in range(1, mdp.M + 1) if (x, y) not in mdp.F}

rewards = {(x, y): mdp.r for (x, y) in states}
rewards.update({(x, y): terminal_r for (x, y), terminal_r in zip(mdp.T, mdp.r_T)})
rewards.update({(x, y):    bonus_r for (x, y), bonus_r    in zip(mdp.B, mdp.r_B)})


def invalid_state(x, y):
    out_of_world = x < 1 or x > mdp.N or y < 1 or y > mdp.M
    return (x, y) in mdp.F or out_of_world


def generate_transitions(states):
    probabilities = [mdp.p1, mdp.p2, mdp.p3, 1 - (mdp.p1 + mdp.p2 + mdp.p3)]
    transitions = dict()
    for x, y in states:
        src_state = (x, y)
        for move in ['U', 'L', 'R', 'D']:
            if move == 'U':
                destination = (x, y + 1)
                left = (x - 1, y)
                right = (x + 1, y)
                opposite = (x, y - 1)

            if move == 'L':
                destination = (x - 1, y)
                left = (x, y - 1)
                right = (x, y + 1)
                opposite = (x + 1, y)

            if move == 'D':
                destination = (x, y - 1)
                left = (x + 1, y)
                right = (x - 1, y)
                opposite = (x, y + 1)

            if move == 'R':
                destination = (x + 1, y)
                left = (x, y + 1)
                right = (x, y - 1)
                opposite = (x - 1, y)

            for dst_state, transition_rate in zip([destination, left, right, opposite], probabilities):
                if invalid_state(*dst_state):
                    dst_state = x, y

                if src_state in transitions:
                    if move in transitions[src_state]:
                        transitions[src_state][move].append((dst_state, transition_rate))
                    else:
                        transitions[src_state][move] = [(dst_state, transition_rate)]
                else:
                    transitions[src_state] = {move: [(dst_state, transition_rate)]}
    return transitions


transitions = generate_transitions(states)
transitions.update({(x, y): {'END': [((x, y), 0)]} for x, y in mdp.T})


class ValueIteration:
    def __init__(self, states, rewards, transitions, stop_cond=0.0001):
        self.states = states
        self.rewards = rewards
        self.transitions = transitions
        self.stop_cond = stop_cond
        self.t = 0  # liczba iteracji
        self.converged = False

    def R(self, state):
        return self.rewards[state]

    def T(self, state, action):
        return self.transitions[state][action]

    def actions(self, state):
        return self.transitions[state].keys()

    def iterate(self):
        utility_history = []
        U1 = {s: 0 for s in states}
        while not self.converged:
            self.converged = True
            self.t += 1
            U = U1.copy()
            utility_history.append(U)
            delta = 0
            for s in self.states:
                U1[s] = self.R(s) + mdp.gamma * max([sum([p * U[s1] for (s1, p) in self.T(s, a)])
                                                     for a in self.actions(s)])
                delta = max(delta, abs(U1[s] - U[s]))

            if delta >= self.stop_cond:
                self.converged = False
        return U, utility_history

    def best_policy(self, U):
        pi = {}
        for s in self.states:
            pi[s] = max(self.actions(s), key=lambda a: self.expected_utility(a, s, U))
        return pi

    def expected_utility(self, a, s, U):
        return sum([p * U[s1] for (s1, p) in self.T(s, a)])


vi = ValueIteration(states, rewards, transitions)
u, history = vi.iterate()
print("Liczba iteracji: ", vi.t)
pi = vi.best_policy(u)

for j in reversed(range(1, mdp.M + 1)):
    for i in range(1, mdp.N + 1):
        print("{:.4f}".format(u.get((i, j), 0.)), end=' ');
    print(end='\n')

print()

arrows = {'U': '^', 'L': '<', 'R': '>', 'D': 'v'}

for j in reversed(range(1, mdp.M + 1)):
    for i in range(1, mdp.N + 1):
        print("{}".format(arrows.get(pi.get((i, j), ' '), ' ')), end=' ');
    print(end='\n')

print()


state_history = {}
for u in history:
    for s in u:
        if s in state_history:
            state_history[s].append(u[s])
        else:
            state_history[s] = [u[s]]
state_history

# tworzenie listy kolumn oraz listy 
columns_to_concatenate = []
state_order = []

for s in state_history:
    columns_to_concatenate.append(np.array(state_history[s])[:, np.newaxis])
    state_order.append(s)

# sklejenie kolumn wzdłuż drugiej współrzędnej (axis=1) i doklejenie numerów iteracji jako pierwszej kolumny
A = np.concatenate(columns_to_concatenate, axis=1)
A = np.concatenate((np.arange(len(history))[:, np.newaxis], A), axis=1).T


