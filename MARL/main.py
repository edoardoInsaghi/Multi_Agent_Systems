# simplified_football_belief_learner.py
# Full implementation of belief-based joint-action learner on Simplified Football (Littman'94)

import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt


class SimpleFootball:

    ROWS = 4
    COLS = 5
    ACTIONS = ['N', 'S', 'E', 'W', '-']

    def __init__(self):
        self.reset()

    def reset(self):
        # A starts in column 0, B in column 4, random row; ownership random
        self.pos_A = (1, 3)
        self.pos_B = (2, 1)
        self.possession = random.choice(['A', 'B'])
        return self._encode_state()

    def _encode_state(self):
        # encode pos_A, pos_B, possession into integer 0..759
        rA, cA = self.pos_A
        rB, cB = self.pos_B
        idx = ((rA * self.COLS + cA) * (self.ROWS * self.COLS) + (rB * self.COLS + cB)) * 2
        if self.possession == 'B': idx += 1
        return idx

    def _decode_state(self, idx):
        s = idx // 2
        pos = 'A' if idx % 2 == 0 else 'B'
        flatA = s // (self.ROWS * self.COLS)
        flatB = s % (self.ROWS * self.COLS)
        rA, cA = divmod(flatA, self.COLS)
        rB, cB = divmod(flatB, self.COLS)
        return (rA, cA), (rB, cB), pos

    def step(self, a_A, a_B): # a_A, a_B are action indices between 0 and 4
        # map action indices to deltas
        moves = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1), '-': (0, 0)}
        order = ['A', 'B']
        random.shuffle(order)
        reward_A = 0
        reward_B = 0

        for player in order:
            if player == 'A':
                act = self.ACTIONS[a_A]
                src = self.pos_A
                dst = (src[0] + moves[act][0], src[1] + moves[act][1])
                other = self.pos_B
                holder = 'A'
            else:
                act = self.ACTIONS[a_B]
                src = self.pos_B
                dst = (src[0] + moves[act][0], src[1] + moves[act][1])
                other = self.pos_A
                holder = 'B'

            dst = (max(0, min(self.ROWS - 1, dst[0])), max(0, min(self.COLS - 1, dst[1])))

            # collision: moving into other -> possession change, no move
            if dst == other:
                self.possession = {'A':'B','B':'A'}[holder]
            else:
                # commit move
                if player == 'A':
                    self.pos_A = dst
                else:
                    self.pos_B = dst

            # check goal
            if self.possession == 'A' and self.pos_A[1] == self.COLS - 1:
                return self.reset(), +1, -1, True
            if self.possession == 'B' and self.pos_B[1] == 0:
                return self.reset(), -1, +1, True

        # no goal
        return self._encode_state(), reward_A, reward_B, False


class BeliefLearner:
    def __init__(self, name, epsilon=0.2, alpha0=1.0, gamma=0.9, T=1e6):
        self.name = name  # 'A' or 'B'
        self.epsilon = epsilon
        self.alpha0 = alpha0
        self.gamma = gamma
        self.alpha = alpha0
        self.T = T
        # Q-table: states x A-actions x B-actions
        self.Q = np.zeros((SimpleFootball.ROWS*SimpleFootball.COLS*SimpleFootball.ROWS*SimpleFootball.COLS*2, 5, 5))
        # counts for opponent actions
        self.N = [defaultdict(lambda: 1.0) for _ in range(self.Q.shape[0])]
        self.steps = 0

    def get_belief(self, s):
        counts = self.N[s]
        total = sum(counts.values())
        return {a: counts[a]/total for a in counts}

    def select_action(self, s):
        if random.random() < self.epsilon:
            return random.randrange(5)
        belief = self.get_belief(s)
        evs = np.zeros(5)
        for a_i in range(5):
            for a_j, p in belief.items():
                evs[a_i] += p * self.Q[s, a_i, a_j]
        return int(np.argmax(evs))

    def update(self, s, a_A, a_B, r, s_next):
        # identify own and opp actions
        if self.name == 'A':
            my_a, opp_a = a_A, a_B
        else:
            my_a, opp_a = a_B, a_A
        # compute V(s_next)
        belief_next = self.get_belief(s_next)
        V_next = max(
            sum(belief_next.get(a_op,0) * self.Q[s_next, a_pr, a_op] for a_op in belief_next)
            for a_pr in range(5)
        )
        # Q update
        self.Q[s, a_A, a_B] = (1-self.alpha)*self.Q[s, a_A, a_B] + \
                              self.alpha*(r + self.gamma * V_next)
        # update belief counts
        self.N[s][opp_a] += 1
        # decay learning rate
        self.steps += 1
        self.alpha = self.alpha0 * (10 ** (np.log10(0.01) * self.steps / self.T))

# --- 3. Training and Evaluation ---
def train(agent, opponent_policy, steps=int(1e6)):
    env = SimpleFootball()
    s = env.reset()
    for t in range(steps):
        a_i = agent.select_action(s)
        a_o = opponent_policy(s)
        s2, rA, rB, done = env.step(a_i if agent.name=='A' else a_o,
                                    a_o if agent.name=='A' else a_i)
        reward = rA if agent.name=='A' else rB
        agent.update(s, a_i if agent.name=='A' else a_o,
                     a_o if agent.name=='A' else a_i,
                     reward, s2)
        s = env.reset() if done else s2
    return agent


def evaluate(agent_A, agent_B, steps=int(1e5), gamma=0.9):
    env = SimpleFootball()
    s = env.reset()
    games, wins_A = 0, 0
    for t in range(steps):
        # early termination draw
        if random.random() > gamma:
            games += 1
            continue
        a_A = agent_A.select_action(s)
        a_B = agent_B.select_action(s)
        s2, rA, rB, done = env.step(a_A, a_B)
        if done:
            games += 1
            if rA > 0: wins_A += 1
            s = env.reset()
        else:
            s = s2
    return games, wins_A / games * 100 if games>0 else 0


if __name__ == '__main__':
    # Train A vs random
    rand_policy = lambda s: random.randrange(5)
    agent_A = BeliefLearner('A')
    agent_A = train(agent_A, rand_policy)
    games, winpct = evaluate(agent_A, BeliefLearner('B'), steps=int(1e5))
    print(f"Agent A vs random: games={games}, win%={winpct:.2f}")

    # Train B vs random
    agent_B = BeliefLearner('B')
    agent_B = train(agent_B, rand_policy)
    games, losepct = evaluate(BeliefLearner('A'), agent_B, steps=int(1e5))
    print(f"Agent B vs random: games={games}, B win%={losepct:.2f}")

    # Train A vs B (identical learners)
    agent1 = BeliefLearner('A')
    agent2 = BeliefLearner('B')
    # alternating updates
    env = SimpleFootball()
    s = env.reset()
    for t in range(int(1e6)):
        a1 = agent1.select_action(s)
        a2 = agent2.select_action(s)
        s2, rA, rB, done = env.step(a1, a2)
        agent1.update(s, a1, a2, rA, s2)
        agent2.update(s, a1, a2, rB, s2)
        s = env.reset() if done else s2
    games, winpct = evaluate(agent1, agent2)
    print(f"Agent1 vs Agent2: games={games}, A win%={winpct:.2f}")

    # OPTIONAL: plot learning curves or distributions
    # (left for extension)
