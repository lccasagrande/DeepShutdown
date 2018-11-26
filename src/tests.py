import gym
import gym_grid.envs.grid_env as g
from src.agents.a2c import SimpleA2C
from collections import defaultdict
from src.utils.networks import mlp
from src.agents.rl_agents import MCPredictionAgent
from src.agents.heuristics import *


def run():
	# env = gym.make("grid-v0")
	discount_factor = 1.0
	env = gym.make("CliffWalking-v0")
	agent = SimpleA2C(mlp([32]),
	                  input_shape=(1,),
	                  nb_actions=env.action_space.n,
	                  name="A2C",
	                  seed=123)

	agent.build_model(env.observation_space.n)

	for _ in range(10):
		obs = env.reset()
		while True:
			action = agent.act(obs)
			next_obs, reward, done, _ = env.step(action)

			state_value = agent.predict_value(obs)
			next_state_value = agent.predict_value(next_obs)
			td_target = reward + discount_factor * next_state_value
			td_error = (td_target - state_value) if not done else td_target

			agent.update_value(obs, td_target)
			agent.update(obs, td_error, action)
			obs = next_obs
			if done:
				break


# agent = MCPredictionAgent(env, SJFAgent(seed=123, name=""))
# agent.train(1, verbose=True, visualize=False)

# table = defaultdict(list)
# for state, actions in agent.Q.items():
#	for act, value in enumerate(actions):
#		if act != 0:
#			for i, s in enumerate(state):
#				table["x_{}".format(i)].append(s)
#			table["action"].append(act)
#			table["value"].append(value)
#

if __name__ == "__main__":
	run()
