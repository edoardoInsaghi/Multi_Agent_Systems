import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import patches


class SimpleFootball:
    ROWS = 4
    COLS = 5
    ACTIONS = ['N', 'S', 'E', 'W', '-']  # stand is '-'

    def __init__(self):
        self.reset()
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def reset(self):
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
    
    def step(self, a_A, a_B):
        moves = {
            'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1), '-': (0, 0)
        }
        order = ['A', 'B']
        random.shuffle(order)

        for player in order:
            if player == 'A':
                act = self.ACTIONS[a_A]
                src = self.pos_A
                other = self.pos_B
                holder = 'A'
            else:
                act = self.ACTIONS[a_B]
                src = self.pos_B
                other = self.pos_A
                holder = 'B'

            # compute intended destination
            dr, dc = moves[act]
            dst_row = src[0] + dr
            dst_col = src[1] + dc

            # check goal crossing: only if holding ball, moving beyond left/right, and in centre rows (1 or 2)
            if holder == self.possession and dst_row in [1, 2]:

                if player == 'B' and dst_col >= self.COLS:
                    # A scores
                    self.reset()
                    return self._encode_state(), +1, -1, True
                
                if player == 'A' and dst_col < 0:
                    # B scores
                    self.reset()
                    return self._encode_state(), -1, +1, True

            # clamp to field boundaries for non-scoring moves
            dst_row = max(0, min(self.ROWS - 1, dst_row))
            dst_col = max(0, min(self.COLS - 1, dst_col))
            dst = (dst_row, dst_col)

            # collision: moving into other -> possession change, no move
            if dst == other:
                self.possession = 'B' if holder == 'A' else 'A'
            else:
                # commit move
                if player == 'A':
                    self.pos_A = dst
                else:
                    self.pos_B = dst

        return self._encode_state(), 0, 0, False   


    def render(self, pause=0.5, wait_for_input=False):
        self.ax.clear()
        self.ax.set_xlim(-2, self.COLS + 2)
        self.ax.set_ylim(-1, self.ROWS + 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        for row in range(-1, self.ROWS + 1):
            for col in range(-2, self.COLS + 2):
                if (row < 0 or row >= self.ROWS) or (col < 0 or col >= self.COLS):
                    self.ax.add_patch(patches.Rectangle((col, row), 1, 1, facecolor='black'))

        for row in range(self.ROWS):
            for col in range(-1, self.COLS + 1):
                if col < 0 or col >= self.COLS:
                    if row in [1, 2]:
                        self.ax.add_patch(patches.Rectangle((col, row), 1, 1, color='lightgreen', alpha=0.5, ec='black'))

        for row in range(self.ROWS):
            for col in range(self.COLS):
                self.ax.add_patch(patches.Rectangle((col, row), 1, 1, edgecolor='black', facecolor='white'))
                if (row, col) == self.pos_A:
                    circle = patches.Circle((col + 0.5, row + 0.5), 0.3, color='blue')
                    self.ax.add_patch(circle)
                    label = 'A*' if self.possession == 'A' else 'A'
                    self.ax.text(col + 0.5, row + 0.5, label, ha='center', va='center', fontsize=12, color='white')
                elif (row, col) == self.pos_B:
                    circle = patches.Circle((col + 0.5, row + 0.5), 0.3, color='red')
                    self.ax.add_patch(circle)
                    label = 'B*' if self.possession == 'B' else 'B'
                    self.ax.text(col + 0.5, row + 0.5, label, ha='center', va='center', fontsize=12, color='white')

        self.ax.set_title(f"Ball Possession: {self.possession}")
        self.fig.canvas.draw()
        plt.pause(pause)

        if wait_for_input:
            input("Press Enter for next move...")




# --- 2. Belief-Based Joint-Action Learner ---
class BeliefLearner:
    def __init__(self, name, epsilon=0.2, alpha0=1.0, gamma=0.9, T=1e6):
        self.name = name  # 'A' or 'B'
        self.epsilon = epsilon
        self.alpha0 = alpha0
        self.gamma = gamma
        self.alpha = alpha0
        self.T = T
        # Q-table: states x A-actions x B-actions
        self.Q = np.zeros((SimpleFootball.ROWS * SimpleFootball.COLS * SimpleFootball.ROWS * SimpleFootball.COLS * 2,
                            len(SimpleFootball.ACTIONS), len(SimpleFootball.ACTIONS)))
        # counts for opponent actions, initialized to 1 for uniform prior
        self.N = [defaultdict(lambda: 1.0) for _ in range(self.Q.shape[0])]
        self.steps = 0

    def get_belief(self, s):
        counts = self.N[s]
        total = sum(counts.values())
        return {a: counts[a] / total for a in counts}

    def select_action(self, s):

        if self.steps < int(1e5):
            return random.randrange(len(SimpleFootball.ACTIONS))
        
        # ε-greedy with behavioural best-response
        if random.random() < self.epsilon:
            return random.randrange(len(SimpleFootball.ACTIONS))
        
        belief = self.get_belief(s)
        if belief == {}:
            belief = {a: 1.0 / len(SimpleFootball.ACTIONS) for a in range(len(SimpleFootball.ACTIONS))}
        #print(f"Agent {self.name} belief: {belief}")
        # compute expected Q for each of our actions
        evs = np.zeros(len(SimpleFootball.ACTIONS))
        for my_a in range(len(SimpleFootball.ACTIONS)):
            for opp_a, p in belief.items():
                evs[my_a] += p * self.Q[s, my_a, opp_a]
        return int(np.argmax(evs))

    def update(self, s, a_A, a_B, r, s_next):
        # determine our and opponent's action indices
        if self.name == 'A':
            my_a, opp_a = a_A, a_B
        else:
            my_a, opp_a = a_B, a_A

        # calculate V(s_next)
        belief_next = self.get_belief(s_next)
        V_next = max(
            sum(belief_next.get(a_op, 0) * self.Q[s_next, a_pr, a_op]
                for a_op in belief_next)
            for a_pr in range(len(SimpleFootball.ACTIONS))
        )

        # Q-update on joint action (a_A, a_B)
        self.Q[s, a_A, a_B] = (1 - self.alpha) * self.Q[s, a_A, a_B] + self.alpha * (r + self.gamma * V_next)

        # belief count update (observed opponent action)
        self.N[s][opp_a] += 1

        # decay learning rate
        self.steps += 1
        self.alpha = self.alpha0 * (10 ** (np.log10(0.01) * self.steps / self.T))

# --- 3. Training and Evaluation ---
def train(agent, opponent_policy, steps=int(1e6)):
    games_completed = 0
    env = SimpleFootball()
    s = env.reset()
    for _ in range(steps):
        a_i = agent.select_action(s)
        a_o = opponent_policy(s)
        # map correct ordering for step
        if agent.name == 'A':
            s2, rA, rB, done = env.step(a_i, a_o)
            reward = rA
            agent.update(s, a_i, a_o, reward, s2)
        else:
            s2, rA, rB, done = env.step(a_o, a_i)
            reward = rB
            agent.update(s, a_o, a_i, reward, s2)
        if done:
            games_completed += 1
            s = env.reset()
        else:
            s = s2
    print(f"Agent {agent.name} completed {games_completed} games")
    return agent


def evaluate(agent_A, agent_B, steps=int(1e5), gamma=0.9):
    env = SimpleFootball()
    s = env.reset()
    games, wins_A = 0, 0
    for _ in range(steps):
        # early termination draw
        if random.random() > gamma:
            games += 1
            continue
        a_A = agent_A.select_action(s)
        a_B = agent_B.select_action(s)
        s2, rA, rB, done = env.step(a_A, a_B)
        env.render(pause=0.5, wait_for_input=False)
        if done:
            games += 1
            if rA > 0:
                wins_A += 1
            s = env.reset()
        else:
            s = s2
    win_pct_A = (wins_A / games * 100) if games > 0 else 0
    return games, win_pct_A



import seaborn as sns

def train_and_visualize_visits_and_qvalues(agent1, agent2, steps=int(1e6)):

    env = SimpleFootball()
    s = env.reset()
    games_completed = 0

    visits = np.zeros_like(agent1.Q, dtype=int)

    for _ in range(steps):
        a1 = agent1.select_action(s)
        a2 = agent2.select_action(s)

        s2, rA, rB, done = env.step(a1, a2)

        # Update agents
        agent1.update(s, a1, a2, rA, s2)
        agent2.update(s, a1, a2, rB, s2)

        # Track visits
        visits[s, a1, a2] += 1

        if done:
            games_completed += 1
            s = env.reset()
        else:
            s = s2

    print(f"Training completed: {games_completed} games")

    # === Visualizations ===
    plt.ioff()

    # 1. Total state visit frequency (aggregated over joint actions)
    state_visits = visits.sum(axis=(1, 2))
    plt.figure(figsize=(10, 4))
    plt.plot(state_visits)
    plt.title("State Visit Frequency")
    plt.xlabel("State Index")
    plt.ylabel("Visit Count")
    plt.tight_layout()
    plt.show()


    # 2. Heatmap of joint action frequencies for a few interesting states
    interesting_states = np.argsort(state_visits)[-3:]  # top 3 visited states
    for s_id in interesting_states:
        heat = visits[s_id]
        print(f"State {env._decode_state(s_id)}")
        plt.figure(figsize=(6, 5))
        sns.heatmap(heat, annot=True, fmt="d", xticklabels=SimpleFootball.ACTIONS, yticklabels=SimpleFootball.ACTIONS)
        plt.title(f"Joint Action Visits for State {s_id}")
        plt.xlabel("Opponent Action")
        plt.ylabel("Agent Action")
        plt.tight_layout()
        plt.show()

    non_interesting_states = np.argsort(state_visits)[:3]  # top 3 visited states
    for s_id in non_interesting_states:
        heat = visits[s_id]
        print(f"State {env._decode_state(s_id)}")
        plt.figure(figsize=(6, 5))
        sns.heatmap(heat, annot=True, fmt="d", xticklabels=SimpleFootball.ACTIONS, yticklabels=SimpleFootball.ACTIONS)
        plt.title(f"Joint Action Visits for State {s_id}")
        plt.xlabel("Opponent Action")
        plt.ylabel("Agent Action")
        plt.tight_layout()
        plt.show()

    sorted_visits = np.argsort(state_visits)
    s_visits = np.sort(state_visits)
    for i in range(50):
        print(f"State {i}: {env._decode_state(sorted_visits[i])}, total visits = {s_visits[i]}")

    q_keep = agent1.Q.flatten()[agent1.Q.flatten() > 1e-6]

    # 3. Histogram of Q-values
    q_values = q_keep
    q_values = q_values[np.abs(q_values) > 1e-6]  # filter out unvisited
    plt.figure(figsize=(8, 4))
    plt.hist(q_values, bins=100, color='steelblue')
    plt.title("Histogram of Learned Q-values (Agent A)")
    plt.xlabel("Q-value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return visits, agent1.Q




if __name__ == '__main__':

    agent1 = BeliefLearner('A')
    agent2 = BeliefLearner('B')
    visits, q_table = train_and_visualize_visits_and_qvalues(agent1, agent2)

