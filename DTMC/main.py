import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta

# Set seed
np.random.seed(42)

# True transition probabilities
true_p = 0.3
true_q = 0.6

# Simulation parameters
N = 200  # number of traces
n = 50   # length of each trace

def simulate_trace(p, q, length, init_dist='stationary'):
    if init_dist == 'stationary':
        pi_A = q / (p + q)
        current = np.random.choice(['A', 'B'], p=[pi_A, 1 - pi_A])
    else:
        current = np.random.choice(['A', 'B'])

    trace = [current]
    for _ in range(length - 1):
        if current == 'A':
            current = np.random.choice(['A', 'B'], p=[1 - p, p])
        else:
            current = np.random.choice(['A', 'B'], p=[q, 1 - q])
        trace.append(current)
    return trace

# Simulate traces
traces = [simulate_trace(true_p, true_q, n) for _ in range(N)]

# Count transitions
N_AA = N_AB = N_BA = N_BB = 0
for trace in traces:
    for i in range(len(trace) - 1):
        curr, next_ = trace[i], trace[i + 1]
        if curr == 'A' and next_ == 'A':
            N_AA += 1
        elif curr == 'A' and next_ == 'B':
            N_AB += 1
        elif curr == 'B' and next_ == 'A':
            N_BA += 1
        elif curr == 'B' and next_ == 'B':
            N_BB += 1

# MLE estimates
p_hat = N_AB / (N_AA + N_AB)
q_hat = N_BA / (N_BA + N_BB)

# Bayesian estimation
alpha_p, beta_p_ = 1, 1
alpha_q, beta_q_ = 1, 1

posterior_p = beta(alpha_p + N_AB, beta_p_ + N_AA)
posterior_q = beta(alpha_q + N_BA, beta_q_ + N_BB)

# Posterior plots
x = np.linspace(0, 1, 1000)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(x, posterior_p.pdf(x))
axs[0].axvline(true_p, color='red', linestyle='--', label="True p")
axs[0].set_title("Posterior of p")
axs[0].set_xlabel("p")
axs[0].legend()

axs[1].plot(x, posterior_q.pdf(x))
axs[1].axvline(true_q, color='red', linestyle='--', label="True q")
axs[1].set_title("Posterior of q")
axs[1].set_xlabel("q")
axs[1].legend()

plt.tight_layout()
plt.savefig("posterior_distributions.png")
plt.close()

# Posterior-weighted sampled trajectories
posterior_samples = 100
p_samples = posterior_p.rvs(posterior_samples)
q_samples = posterior_q.rvs(posterior_samples)

trajectory_length = 30
posterior_traces = []
for ps, qs in zip(p_samples, q_samples):
    trace = simulate_trace(ps, qs, trajectory_length, init_dist='stationary')
    posterior_traces.append([0 if state == 'A' else 1 for state in trace])

posterior_traces = np.array(posterior_traces)

plt.figure(figsize=(10, 6))
sns.heatmap(posterior_traces, cmap='coolwarm', cbar_kws={'label': 'State (0=A, 1=B)'})
plt.xlabel("Time step")
plt.ylabel("Posterior Sample Index")
plt.title("Posterior-Weighted Sampled Trajectories")
plt.savefig("posterior_trajectories.png")
plt.close()

# Empirical transition matrix heatmap
empirical_P = np.array([
    [N_AA / (N_AA + N_AB), N_AB / (N_AA + N_AB)],
    [N_BA / (N_BA + N_BB), N_BB / (N_BA + N_BB)]
])

plt.figure(figsize=(6, 5))
sns.heatmap(empirical_P, annot=True, cmap="Blues", xticklabels=["A", "B"], yticklabels=["A", "B"])
plt.title("Empirical Transition Matrix")
plt.savefig("empirical_transition_matrix.png")
plt.close()

# State frequency over time
state_counts = np.zeros((n, 2))  # [timestep, state: A=0, B=1]
for trace in traces:
    for t, state in enumerate(trace):
        if state == 'A':
            state_counts[t, 0] += 1
        else:
            state_counts[t, 1] += 1
state_freq = state_counts / N

plt.figure(figsize=(10, 5))
plt.plot(state_freq[:, 0], label="State A")
plt.plot(state_freq[:, 1], label="State B")
plt.title("Empirical State Frequencies Over Time")
plt.xlabel("Time step")
plt.ylabel("Proportion")
plt.legend()
plt.savefig("state_frequencies_over_time.png")
plt.close()

# MLE convergence with increasing data
N_vals = np.linspace(10, N, 10, dtype=int)
p_hats = []
q_hats = []

for n_val in N_vals:
    N_AA = N_AB = N_BA = N_BB = 0
    for trace in traces[:n_val]:
        for i in range(len(trace) - 1):
            curr, next_ = trace[i], trace[i + 1]
            if curr == 'A' and next_ == 'A':
                N_AA += 1
            elif curr == 'A' and next_ == 'B':
                N_AB += 1
            elif curr == 'B' and next_ == 'A':
                N_BA += 1
            elif curr == 'B' and next_ == 'B':
                N_BB += 1
    p_hats.append(N_AB / (N_AA + N_AB))
    q_hats.append(N_BA / (N_BA + N_BB))

plt.figure(figsize=(10, 5))
plt.plot(N_vals, p_hats, label="MLE p")
plt.plot(N_vals, q_hats, label="MLE q")
plt.axhline(true_p, color='red', linestyle='--', label="True p")
plt.axhline(true_q, color='blue', linestyle='--', label="True q")
plt.xlabel("Number of Sequences Used")
plt.ylabel("Estimated Parameter")
plt.title("Convergence of MLE Estimates")
plt.legend()
plt.savefig("mle_convergence.png")
plt.close()
