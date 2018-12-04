import gym
import gridgym.envs.grid_env as g
import argparse
from src.agents.ppo import PPOAgent

METRICS = [
	"energy_consumed",
	"makespan",
	"total_slowdown",
	"mean_slowdown",
	"total_turnaround_time",
	"mean_turnaround_time",
	"total_waiting_time",
	"mean_waiting_time",
]


def run(args):
	agent = PPOAgent("lstm", args.env, args.num_env, args.seed, args.reward_scale)

	if args.test:
		agent.evaluate(args.weights, METRICS, args.output_fn, args.verbose, args.render)
	else:
		agent.train(int(args.num_timesteps), save_path=args.weights)

	print("Done!")


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="grid-v0")
	parser.add_argument("--num_env", default=24, type=int)
	parser.add_argument("--seed", type=int, default=123)
	parser.add_argument("--reward_scale", type=float, default=1.0)
	parser.add_argument("--weights", default="../weights/ppo_weights", type=str)
	parser.add_argument("--output_fn", default=None, type=str)
	parser.add_argument("--verbose", default=True, action="store_true")
	parser.add_argument("--render", default=False, action="store_true")
	parser.add_argument("--num_timesteps", type=int, default=20e6)
	parser.add_argument("--test", default=True, action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
