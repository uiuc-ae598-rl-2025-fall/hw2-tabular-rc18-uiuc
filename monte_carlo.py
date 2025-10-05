import numpy as np
import gymnasium as gym
from utils import * 

is_slippery = True
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=is_slippery)
env.reset()

def on_monte_carlo(iterations):
	epsilon = 0.1 if is_slippery else 0.2
	gamma = 0.95
	state_space = set(range(16))
	action_space = set(range(4))
	e_soft = epsilon / len(action_space)
	policy = np.full((16, 4), 0.25)
	Q = np.zeros((16, 4))
	returns = [[[0, 0] for _ in range(4)] for _ in range(16)] #sum, count instead of storing as lists
	eval_time_step = iterations // 1000
	returns_metric = []

	for iter in range(iterations):
		if iter % 100 == 0:
			print(f"Iteration: {iter}") #debug
		
		#Generate an episode
		episode = []
		is_end = False
		state = env.reset()[0]
		while is_end == False:
			action_prob = policy[state]
			action = np.random.choice(np.arange(4), p=action_prob)
			next_state, reward, is_end, _, _ = env.step(action)
			episode.append((state, action, reward))
			state = next_state

		#update
		G = 0
		episode_length = len(episode)
		for i in range(episode_length-1, -1, -1):
			current_state, current_action = episode[i][0], episode[i][1]
			G = gamma * G + episode[i][2]
			exists = any((prev_state, prev_action) == (current_state, current_action) for prev_state, prev_action, _ in episode[:i])
			if not exists:
				return_sum, count = returns[current_state][current_action]
				return_sum += G
				count += 1
				returns[current_state][current_action] = [return_sum, count]

				Q[current_state, current_action] = return_sum / count
				best_action = np.argmax(Q[current_state, :])
				for a in range(len(action_space)):
					policy[current_state][a] = (1 - epsilon + e_soft) if a == best_action else e_soft
		
		if iter % eval_time_step == 0:
			returns_metric.append(evaluate_policy(policy, env))

	return policy, returns_metric


policy, avg_returns = on_monte_carlo(iterations=50000)

visualize_policy(policy)
plot_eval(avg_returns, is_slippery, "Average Returns using Monte Carlo")
plot_policy(policy, is_slippery, "Policy using Monte Carlo")