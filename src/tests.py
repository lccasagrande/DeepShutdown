import gym
import gym_grid.envs.grid_env as g
import pandas as pd
from collections import defaultdict
from src.agents.rl_agents import  MCPredictionAgent
from src.agents.heuristics import *


def run():
	env = gym.make("grid-v0")
	agent = MCPredictionAgent(env, SJFAgent(seed=123, name=""))
	agent.train(1, verbose=True, visualize=False)

	table = defaultdict(list)
	for state, actions in  agent.Q.items():
		for act, value in enumerate(actions):
			if act != 0:
				for i, s in enumerate(state):
					table["x_{}".format(i)].append(s)
				table["action"].append(act)
				table["value"].append(value)

if __name__ == "__main__":
	run()
