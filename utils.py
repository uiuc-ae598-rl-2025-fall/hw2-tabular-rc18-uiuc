#Visualization utilities
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

def visualize_policy(policy):
    size=4
    grid = []
    for s in range(policy.shape[0]):
        if np.allclose(policy[s], 0.25):  # uniform = probably hole or terminal
            grid.append('·')  # mark as "don't care"
        else:
            best_action = np.argmax(policy[s])
            grid.append(arrows[best_action])
    # reshape into grid
    for i in range(size):
        print(' '.join(grid[i*size:(i+1)*size]))

def visualize_Q(Q):
    size=4
    grid = []
    for s in range(Q.shape[0]):
        if np.allclose(Q[s], 0.25):  # uniform = probably hole or terminal
            grid.append('·')  # mark as "don't care"
        else:
            best_action = np.argmax(Q[s])
            grid.append(arrows[best_action])
    # reshape into grid
    for i in range(size):
        print(' '.join(grid[i*size:(i+1)*size]))


def plot_mean_val(iterations, mean_val, iteration, is_slippery):
    slippery_text = "(Slippery)" if is_slippery else "(Not Slippery)"
    fig, ax = plt.subplots()
    ax.set_title("Mean of Value Function over Iterations")
    ax.plot(range(iterations), mean_val)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean of Value Function")
    fig.savefig(f"img/mean_val_fn_{iteration} {slippery_text}")

def plot_V(V, iteration, is_slippery, grid_size, title="State Value Function"):
    slippery_text = "(Slippery)" if is_slippery else "(Not Slippery)"
    V_grid = np.array(V).reshape(grid_size)
    plt.figure(figsize=(6,6))
    plt.imshow(V_grid, cmap='coolwarm', interpolation='nearest')
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            plt.text(j, i, f"{V_grid[i,j]:.2f}", 
                     ha='center', va='center', color='black')
    plt.title(title)
    plt.colorbar()
    plt.savefig(f"img/state_val_graph_{iteration} {slippery_text}")

def plot_policy(policy, iteration, is_slippery, grid_size, title="Policy"):
    slippery_text = "(Slippery)" if is_slippery else "(Not Slippery)"
    action_arrows = {0:'←', 1:'↓', 2:'→', 3:'↑'}
    plt.figure(figsize=(6,6))
    plt.xlim(-0.5, grid_size[1]-0.5)
    plt.ylim(-0.5, grid_size[0]-0.5)
    plt.gca().invert_yaxis()
    
    for s in range(grid_size[0]*grid_size[1]):
        i, j = divmod(s, grid_size[1])
        best_a = np.argmax(policy[s])
        plt.text(j, i, action_arrows[best_a], ha='center', va='center', fontsize=20)
    
    plt.title(title)
    plt.savefig(f"img/policy_graph_{iteration} {slippery_text}")

def run_episode(policy, P, start_state=0, max_steps=100):
    s = start_state
    trajectory = [s]
    for t in range(max_steps):
        a = np.argmax(policy[s])  # greedy action
        # pick one next state according to transition probabilities
        probs = [prob for (prob, ns, r, done) in P[s][a]]
        next_states = [ns for (prob, ns, r, done) in P[s][a]]
        next_s = np.random.choice(next_states, p=probs)
        trajectory.append(next_s)
        
        if any(done for (prob, ns, r, done) in P[s][a] if ns == next_s):
            break
        s = next_s
    return trajectory

def plot_trajectory(traj, grid_size=(4,4)):
    plt.figure(figsize=(6,6))
    plt.xlim(-0.5, grid_size[1]-0.5)
    plt.ylim(-0.5, grid_size[0]-0.5)
    plt.gca().invert_yaxis()
    
    for idx, s in enumerate(traj):
        i, j = divmod(s, grid_size[1])
        plt.text(j, i, str(idx), ha='center', va='center', fontsize=12, color='red')
    
    plt.title("Example Trajectory")
    plt.show()
