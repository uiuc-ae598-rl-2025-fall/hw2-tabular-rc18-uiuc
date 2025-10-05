#Visualization utilities
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def evaluate_policy(policy, env):
	num_episodes = 100
	total_returns = 0
	for i in range(num_episodes):
		state, _ = env.reset()
		is_end = False
		G = 0
		while is_end == False:
			action = np.random.choice(np.arange(4), p=policy[state])
			state, reward, is_end, _, _ = env.step(action)
			G += reward
		total_returns += G
	return total_returns / num_episodes

def evaluate_val_fn(Q, env):
	num_episodes = 100
	total_returns = 0
	max_steps = 100
	steps = 0
	for i in range(num_episodes):
		state, _ = env.reset()
		is_end = False
		G = 0
		while is_end == False and steps < max_steps:
			action = np.argmax(Q[state])
			state, reward, is_end, _, _ = env.step(action)
			G += reward
			steps += 1
		total_returns += G
	return total_returns / num_episodes

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

def plot_eval(avg_returns, is_slippery, title):
	slippery_text = "(Slippery)" if is_slippery else "(Not Slippery)"

	plt.figure(figsize=(10,5))
	window=100
	avg_returns = np.convolve(avg_returns, np.ones(window)/window, mode='valid')
	plt.plot(np.arange(len(avg_returns)), avg_returns, label=f"Training avg return")

	plt.xlabel("Environmental time steps / episodes")
	plt.ylabel("Average return")
	plt.title(f"{title} {slippery_text}")
	plt.legend()
	plt.grid(True)
	plt.savefig(f"img/{title} {slippery_text}")
	plt.show()

def plot_policy(policy, is_slippery, title):
	grid_size = (4, 4)
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
	
	plt.title(f"{title} {slippery_text}")
	plt.savefig(f"img/{title} {slippery_text}")
