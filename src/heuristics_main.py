import gym
import src.GridEnv.envs.grid_env as g
import argparse
import pandas as pd
from src.agents.heuristics import *


def run(args):
	env = gym.make(args.env)

	if args.heuristic == "user":
		agent = UserAgent(seed=args.seed, name="")
	elif args.heuristic == "sjf":
		agent = SJFAgent(seed=args.seed, name="")
	elif args.heuristic == "random":
		agent = RandomAgent(seed=args.seed, name="")
	elif args.heuristic == "ljf":
		agent = LJFAgent(seed=args.seed, name="")
	elif args.heuristic == "packer":
		agent = PackerAgent(seed=args.seed, name="")
	elif args.heuristic == "tetris":
		agent = TetrisAgent(seed=args.seed, name="")
	elif args.heuristic == "firstfit":
		agent = FirstFitAgent(seed=args.seed, name="")
	else:
		raise NotImplementedError("Agent not implemented")

	results = agent.evaluate(env, args.nb_episodes, args.visualize)
	if args.output is not None:
		dt = pd.DataFrame([results])
		dt.to_csv(args.output, index=False)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="grid-v0")
	parser.add_argument("--heuristic", type=str, default="sjf")
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--visualize", default=False, action="store_true")
	parser.add_argument("--nb_episodes", default=1, type=int)
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
