import numpy as np
import gymnasium as gym
from utils import * 

is_slippery = True
env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=is_slippery)
env.reset()

def Q_learning(iterations, stepsize=0.2):
	Q = np.zeros((16, 4))
	gamma = 0.95
	epsilon = 0.1 if is_slippery else 0.2
	eval_time_step = iterations // 1000
	returns_metric = []
	
	for iter in range(iterations):
		if iter % 100 == 0:
			print(f"Iteration: {iter}")
	
		state = env.reset()[0] #initialize S

		is_end = False
		while is_end == False:
			action = np.random.choice(np.arange(4)) if np.random.rand() < epsilon else np.argmax(Q[state])
			next_state, reward, is_end, _, _ = env.step(action)

			Q[state, action] += stepsize * (reward + gamma * np.max(Q[next_state, :]) * (not is_end) - Q[state, action])
			state = next_state	
		
		if iter % eval_time_step == 0:
			returns_metric.append(evaluate_val_fn(Q, env))

	return Q, returns_metric

Q, avg_returns = Q_learning(iterations=50000)
visualize_policy(Q)
plot_eval(avg_returns, is_slippery, "Average Returns using Q-Learning")
plot_policy(Q, is_slippery, "Policy using Q-Learning")