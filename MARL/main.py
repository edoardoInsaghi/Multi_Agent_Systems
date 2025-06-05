import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import pandas as pd


class SimpleFootball:
    ROWS = 4
    COLS = 5
    ACTIONS = ['N', 'S', 'E', 'W', '-']

    def __init__(self):
        self.step_counter = 0
        self.reset()
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def reset(self):
        self.step_counter = 0
        self.pos_A = (1, 3)
        self.pos_B = (2, 1)
        self.possession = random.choice(['A', 'B'])
        return self._encode_state()

    def _encode_state(self):
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
    
    def step(self, a_A, a_B, log=False):

        if log:
            print(f"Step {self.step_counter}: A = {self.ACTIONS[a_A]} B = {self.ACTIONS[a_B]}")
            self.step_counter += 1

        moves = {
            'N': (-1, 0), 
            'S': (1, 0), 
            'E': (0, 1), 
            'W': (0, -1), 
            '-': (0, 0)
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

            dr, dc = moves[act]
            dst_row = src[0] + dr
            dst_col = src[1] + dc

            if holder == self.possession and dst_row in [1, 2]:
                if player == 'B' and dst_col >= self.COLS:
                    # B scores
                    self.reset()
                    return self._encode_state(), -1, +1, True
                
                if player == 'A' and dst_col < 0:
                    # A scores
                    self.reset()
                    return self._encode_state(), +1, -1, True

            # allows for actions that move outer bounds, but turns them into -
            dst_row = max(0, min(self.ROWS - 1, dst_row))
            dst_col = max(0, min(self.COLS - 1, dst_col))
            dst = (dst_row, dst_col)

            # moving into other -> possession change if active player has the ball, no move
            if dst == other:
                if holder == self.possession:
                    self.possession = 'B' if holder == 'A' else 'A'
            else:
                if player == 'A':
                    self.pos_A = dst
                else:
                    self.pos_B = dst

        return self._encode_state(), 0, 0, False   


    def render(self, pause=0.5, wait_for_input=False, score_a=None, score_b=None):
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

        # Display score if provided
        if score_a is not None and score_b is not None:
            self.ax.text(self.COLS / 2, -1.2, f"Score A: {score_a}   |   Score B: {score_b}",
                        ha='center', va='center', fontsize=12, fontweight='bold')

        self.fig.canvas.draw()
        plt.pause(pause)

        if wait_for_input:
            input("Press Enter for next move...")



def random_policy(s):
    return random.randrange(len(SimpleFootball.ACTIONS))


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
                           len(SimpleFootball.ACTIONS), 
                           len(SimpleFootball.ACTIONS)))
        
        # counts for opponent actions, initialized to 1 for weakly informative uniform prior
        self.N = [{0:1, 1:1, 2:1, 3:1, 4:1} for _ in range(self.Q.shape[0])]
        self.steps = 0


    def get_belief(self, s):
        counts = self.N[s]
        total = sum(counts.values())
        return {a: counts[a] / total for a in counts}


    def select_action(self, s, explore=True, exploring_starts=False, log=False):

        if self.steps < int(1e5) and exploring_starts:
            return random.randrange(len(SimpleFootball.ACTIONS))
        
        # ε-greedy
        if random.random() < self.epsilon and explore:
            return random.randrange(len(SimpleFootball.ACTIONS))
        
        belief = self.get_belief(s)
        if log:
            print(f"Agent {self.name} beliefs: {[f'{SimpleFootball.ACTIONS[a]}: {belief[a]:.2f}' for a in belief.keys()]}")

        # compute expected Q for each of player actions over opponent action beliefs
        evs = np.zeros(len(SimpleFootball.ACTIONS))
        for my_a in range(len(SimpleFootball.ACTIONS)):
            for opp_a, p in belief.items():
                evs[my_a] += p * self.Q[s, my_a, opp_a]

        return int(np.argmax(evs))


    def update(self, s, a_A, a_B, r, s_next):

        if self.name == 'A':
            my_a, opp_a = a_A, a_B
        else:
            my_a, opp_a = a_B, a_A

        # calculate V(s') using the next state and beliefs about opponent actions from s'
        belief_next = self.get_belief(s_next)
        V_next = max(
            sum(belief_next.get(a_op) * self.Q[s_next, a_pr, a_op] for a_op in belief_next.keys())
            for a_pr in range(len(SimpleFootball.ACTIONS))
        )

        # update Q, beliefs, and lr
        self.Q[s, my_a, opp_a] = (1 - self.alpha) * self.Q[s, my_a, opp_a] + self.alpha * (r + self.gamma * V_next)
        self.N[s][opp_a] += 1
        self.steps += 1
        self.alpha = self.alpha0 * (10 ** (np.log10(0.01) * self.steps / self.T))



def train_and_eval_against_random(steps=int(1e6), 
                                  exploring_starts=False, 
                                  log=False, 
                                  wait=True):

    agent = BeliefLearner('A')
    
    training_games = 0
    visits = np.zeros_like(agent.Q, dtype=int)
    rewards_a = []
    rewards_b = []
    env = SimpleFootball()
    s = env.reset()

    # Training against random opponent
    for _ in range(steps):

        a_i = agent.select_action(s, explore=True, exploring_starts=exploring_starts, log=False)
        a_o = random_policy(s)

        if agent.name == 'A':
            s2, rA, rB, done = env.step(a_i, a_o)
            reward = rA
            rewards_a.append(rA)
            rewards_b.append(rB)
            visits[s, a_i, a_o] += 1
            agent.update(s, a_i, a_o, reward, s2)
        else:
            s2, rA, rB, done = env.step(a_o, a_i)
            reward = rB
            rewards_a.append(rA)
            rewards_b.append(rB)
            visits[s, a_o, a_i] += 1
            agent.update(s, a_o, a_i, reward, s2)

        if done:
            training_games += 1
            s = env.reset()
        else:
            s = s2

    print(("\n======================================================================================================================"))
    print(f"Training Belief Based Learner against Random, {training_games} games completed in {steps} steps.")
    plt.ioff()

    # State visits frequencies
    state_visits = visits.sum(axis=(1, 2))
    plt.figure(figsize=(10, 4))
    plt.plot(state_visits)
    plt.title("State Visit Frequency")
    plt.xlabel("State Index")
    plt.ylabel("Visit Count")
    plt.tight_layout()
    plt.show()

    # State action visits of most visited states
    # most_visited_states = np.argsort(state_visits)[-3:]
    # for s_id in most_visited_states:
    #     heat = visits[s_id]
    #     print(f"State {env._decode_state(s_id)}")
    #     plt.figure(figsize=(6, 5))
    #     sns.heatmap(heat, annot=True, fmt="d", xticklabels=SimpleFootball.ACTIONS, yticklabels=SimpleFootball.ACTIONS)
    #     plt.title(f"Joint Action Visits for State {s_id}")
    #     plt.xlabel("Opponent Action")
    #     plt.ylabel("Agent Action")
    #     plt.tight_layout()
    #     plt.show()

    # Q-values histogram
    q_values = agent.Q.flatten()
    q_values = q_values[np.abs(q_values) > 1e-5]

    plt.figure(figsize=(10, 5))
    sns.histplot(q_values, bins=100, color='steelblue', stat='density', kde=False, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Learned Q-values vs Random Opponent", fontsize=14)
    plt.xlabel("Q-value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Bar plot of immediate rewards for both agents
    bins = [-1, 0, 1]
    counts_a = [rewards_a.count(b) for b in bins]
    counts_b = [rewards_b.count(b) for b in bins]
    x = np.arange(len(bins))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, counts_a, width, label='Agent A', color='blue', alpha=0.5)
    plt.bar(x + width/2, counts_b, width, label='Agent B', color='red', alpha=0.5)
    plt.xticks(x, bins)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Bar Plot of Immediate Rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Evaluation against random policy
    s = env.reset()
    eval_games, wins_A = 0, 0
    game_lengths = []
    game_length = 0
    for _ in range(steps//10):

        game_length += 1

        # early termination draw
        if random.random() > 0.9:
            s = env.reset()
            game_length = 0
            continue

        a_A = agent.select_action(s, explore=False, exploring_starts=False, log=log)
        a_B = random_policy(s)
        s2, rA, rB, done = env.step(a_A, a_B, log=log)

        if log:
            plt.ion()
            env.render(pause=0.5, wait_for_input=wait)

        if done:
            eval_games += 1
            game_lengths.append(game_length)
            game_length = 0
            if rA > 0:
                wins_A += 1
            s = env.reset()
        else:
            s = s2

    win_pct_A = (wins_A / eval_games * 100) if eval_games > 0 else 0
    print(f"Agent {agent.name} win percentage against Random: {win_pct_A:.2f}% over {eval_games} non early terminated games.")
    print(f"Average length of non early terminated games: {np.mean(game_lengths) if game_lengths else 0:.2f} steps.")
    print(("======================================================================================================================\n"))

    return q_values


def train_and_eval_against_same(steps=int(1e6), 
                                exploring_starts=False, 
                                log=False, 
                                wait=True):
    
    agent1 = BeliefLearner('A')
    agent2 = BeliefLearner('B')

    training_games = 0
    visits = np.zeros_like(agent1.Q, dtype=int)
    rewards_a = []
    rewards_b = []
    env = SimpleFootball()
    s = env.reset()

    # Training against other belief learner
    for _ in range(steps):
        a1 = agent1.select_action(s, explore=True, exploring_starts=exploring_starts, log=False)
        a2 = agent2.select_action(s, explore=True, exploring_starts=exploring_starts, log=False)

        s2, rA, rB, done = env.step(a1, a2)
        rewards_a.append(rA)
        rewards_b.append(rB)

        agent1.update(s, a1, a2, rA, s2)
        agent2.update(s, a1, a2, rB, s2)

        visits[s, a1, a2] += 1

        if done:
            training_games += 1
            s = env.reset()
        else:
            s = s2
    print(("\n======================================================================================================================"))
    print(f"Training Belief Based Learner against other Belief Learner, {training_games} games completed in {steps} steps.")
    plt.ioff()

    # State visits frequencies
    state_visits = visits.sum(axis=(1, 2))
    plt.figure(figsize=(10, 4))
    plt.plot(state_visits)
    plt.title("State Visit Frequency")
    plt.xlabel("State Index")
    plt.ylabel("Visit Count")
    plt.tight_layout()
    plt.show()

    # State action visits of most visited states
    # most_visited_states = np.argsort(state_visits)[-3:]
    # for s_id in most_visited_states:
    #     heat = visits[s_id]
    #     print(f"State {env._decode_state(s_id)}")
    #     plt.figure(figsize=(6, 5))
    #     sns.heatmap(heat, annot=True, fmt="d", xticklabels=SimpleFootball.ACTIONS, yticklabels=SimpleFootball.ACTIONS)
    #     plt.title(f"Joint Action Visits for State {s_id}")
    #     plt.xlabel("Opponent Action")
    #     plt.ylabel("Agent Action")
    #     plt.tight_layout()
    #     plt.show()

    # Q-values histogram
    q_values = agent1.Q.flatten()
    q_values = q_values[np.abs(q_values) > 1e-5]

    plt.figure(figsize=(10, 5))
    sns.histplot(q_values, bins=100, color='steelblue', stat='density', kde=False, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Learned Q-values vs other Belief Learner", fontsize=14)
    plt.xlabel("Q-value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    q_values = agent2.Q.flatten()
    q_values = q_values[np.abs(q_values) > 1e-5]

    plt.figure(figsize=(10, 5))
    sns.histplot(q_values, bins=100, color='steelblue', stat='density', kde=False, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Learned Q-values vs Belief Learner", fontsize=14)
    plt.xlabel("Q-value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Bar plot of immediate rewards for both agents
    bins = [-1, 0, 1]
    counts_a = [rewards_a.count(b) for b in bins]
    counts_b = [rewards_b.count(b) for b in bins]
    x = np.arange(len(bins))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, counts_a, width, label='Agent A', color='blue', alpha=0.5)
    plt.bar(x + width/2, counts_b, width, label='Agent B', color='red', alpha=0.5)
    plt.xticks(x, bins)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Bar Plot of Immediate Rewards")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Evaluation against other belief learner
    s = env.reset()
    eval_games, wins_A = 0, 0
    game_lengths = []
    game_length = 0
    for _ in range(steps//10):

        game_length += 1

        # early termination draw
        if random.random() > 0.9:
            s = env.reset()
            game_length = 0
            continue

        a_A = agent1.select_action(s, explore=False, exploring_starts=False, log=log)
        a_B = agent2.select_action(s, explore=False, exploring_starts=False, log=log)
        s2, rA, rB, done = env.step(a_A, a_B, log=log)

        if log:
            plt.ion()
            env.render(pause=0.5, wait_for_input=wait)

        if done:
            eval_games += 1
            game_lengths.append(game_length)
            game_length = 0
            if rA > 0:
                wins_A += 1
            s = env.reset()
        else:
            s = s2

    win_pct_A = (wins_A / eval_games * 100) if eval_games > 0 else 0

    print(f"Agent {agent1.name} win percentage against other Belief Learner: {win_pct_A:.2f}% over {eval_games} non early terminated games.")
    print(f"Average length of non early terminated games: {np.mean(game_lengths) if game_lengths else 0:.2f} steps.")


    # evaluation against random opponent
    s = env.reset()
    eval_games, wins_A = 0, 0
    game_lengths = []
    game_length = 0
    for _ in range(steps//10):

        game_length += 1

        # early termination draw
        if random.random() > 0.9:
            s = env.reset()
            game_length = 0
            continue

        a_A = agent1.select_action(s, explore=False, exploring_starts=False, log=log)
        a_B = random_policy(s)
        s2, rA, rB, done = env.step(a_A, a_B, log=log)

        if log:
            plt.ion()
            env.render(pause=0.5, wait_for_input=wait)

        if done:
            eval_games += 1
            game_lengths.append(game_length)
            game_length = 0
            if rA > 0:
                wins_A += 1
            s = env.reset()
        else:
            s = s2

    win_pct_A = (wins_A / eval_games * 100) if eval_games > 0 else 0

    print(f"Agent {agent1.name} win percentage against Random: {win_pct_A:.2f}% over {eval_games} non early terminated games.")
    print(f"Average length of non early terminated games: {np.mean(game_lengths) if game_lengths else 0:.2f} steps.")
    print(("======================================================================================================================\n"))

    return q_values



def traing_against_random_eval_against_same(steps=int(1e6), 
                                            exploring_starts=False, 
                                            log=False, 
                                            wait=True):
    
    agent1 = BeliefLearner('A')
    agent2 = BeliefLearner('B')
    agent3 = BeliefLearner('A')

    training_games = 0
    env = SimpleFootball()
    s = env.reset()

    # Training against random opponent
    for _ in range(steps):
        a1 = agent1.select_action(s, explore=True, exploring_starts=exploring_starts, log=False)
        a2 = random.randrange(len(SimpleFootball.ACTIONS))

        s2, rA, rB, done = env.step(a1, a2)
        agent1.update(s, a1, a2, rA, s2)

        if done:
            training_games += 1
            s = env.reset()
        else:
            s = s2

    print(("\n======================================================================================================================"))
    print(f"Training Belief Based Learner against Random, {training_games} games completed in {steps} steps.")

    # Training two belief learners one against the other
    training_games = 0
    s = env.reset()
    for _ in range(steps):

        a1 = agent3.select_action(s, explore=True, exploring_starts=exploring_starts, log=False)
        a2 = agent2.select_action(s, explore=True, exploring_starts=exploring_starts, log=False)
        
        s2, rA, rB, done = env.step(a1, a2)
        agent2.update(s, a1, a2, rB, s2)
        agent3.update(s, a1, a2, rA, s2)

        if done:
            training_games += 1
            s = env.reset()
        else:
            s = s2

    print(f"Training Belief Based Learner against other Belief Learner, {training_games} games completed in {steps} steps.")

    # Evaluation against other belief learner
    s = env.reset()
    eval_games, wins_A, wins_B = 0, 0, 0
    game_lengths = []
    game_length = 0
    for _ in range(steps//10):

        game_length += 1

        if random.random() > 0.9:
            s = env.reset()
            game_length = 0
            continue

        a_A = agent1.select_action(s, explore=False, exploring_starts=False, log=log)
        a_B = agent2.select_action(s, explore=False, exploring_starts=False, log=log)
        s2, rA, rB, done = env.step(a_A, a_B, log=log)

        if log:
            plt.ion()
            env.render(pause=0.5, wait_for_input=wait, score_a=wins_A, score_b=wins_B)

        if done:
            eval_games += 1
            game_lengths.append(game_length)
            game_length = 0
            if rA > 0:
                wins_A += 1
            else: 
                wins_B += 1
            s = env.reset()
        else:
            s = s2

    win_pct_A = (wins_A / eval_games * 100) if eval_games > 0 else 0
    print(f"Agent {agent1.name} trained against random win percentage against other Belief Learner: {win_pct_A:.2f}% over {eval_games} non early terminated games.")
    print(f"Average length of non early terminated games: {np.mean(game_lengths) if game_lengths else 0:.2f} steps.")
    print(("======================================================================================================================\n"))







if __name__ == '__main__':

    ## q_random = train_and_eval_against_random(steps=int(1e6), exploring_starts=False, log=False, wait=False)
    ## q_belief = train_and_eval_against_same(steps=int(1e6), exploring_starts=False, log=False, wait=False)

    ## q_random_flat = np.array(q_random).flatten()
    ## q_belief_flat = np.array(q_belief).flatten()


    ## df = pd.DataFrame({
    ##     'Q-value': np.concatenate([q_random_flat, q_belief_flat]),
    ##     'Training Opponent': ['Random'] * len(q_random_flat) + ['Belief Learner'] * len(q_belief_flat)
    ## })
    ## plt.figure(figsize=(10, 5))
    ## sns.histplot(data=df, x='Q-value', hue='Training Opponent', bins=100, stat='density',
    ##             palette={'Random': 'blue', 'Belief Learner': 'red'}, alpha=0.5, element='step')

    ## plt.title("Comparison of Q-value Distributions")
    ## plt.xlabel("Q-value")
    ## plt.ylabel("Density")
    ## plt.tight_layout()
    ## plt.show()

    traing_against_random_eval_against_same(steps=int(1e6), exploring_starts=False, log=False, wait=False)



