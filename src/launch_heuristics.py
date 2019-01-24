import gym
import gridgym.envs.grid_env
import argparse
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

	agent.evaluate(env, args.nb_episodes, args.visualize, save_hist= args.hist is not None)

	if args.hist is not None:
		agent.save_history(args.hist)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="batsim-v0")
	parser.add_argument("--heuristic", type=str, default="sjf")
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--visualize", default=True, action="store_true")
	parser.add_argument("--hist", default=None, type=str) #"sjf-hist.csv"
	parser.add_argument("--nb_episodes", default=1, type=int)
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
