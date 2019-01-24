import gym
import gym_grid.envs.grid_env as g
import argparse
import os
import pandas as pd
from utils.networks import mlp
from utils.commons import make_vec_env
from agents.reinforce import ReinforceAgent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run(args):
	env = gym.make(args.env)
	network = mlp([20])
	agent = ReinforceAgent(network,
	                       gamma=1.0,
	                       input_shape=env.observation_space.shape,
	                       nb_actions=env.action_space.n,
	                       name=args.name,
	                       save_path=args.save_path,
	                       seed=args.seed)

	agent.build_model(lr=args.lr)

	if not args.test:
		env = make_vec_env(args.env, args.nb_env, args.seed)
		agent.fit(env,
		          args.nb_iteration,
		          args.nb_env,
		          args.nb_max_steps,
		          args.log_interval,
		          args.save_interval,
		          args.summarize)
	else:
		agent.load_model()
		results = agent.evaluate(env, 1, False)
		if args.test_outputfn is not None:
			dt = pd.DataFrame([results])
			dt.to_csv(args.test_outputfn, index=False)
		return results


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="grid-v0")
	parser.add_argument("--name", type=str, default="reinforce22")
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--lr", default=1e-3, type=float)
	parser.add_argument("--nb_env", default=12, type=int)
	parser.add_argument("--nb_iteration", type=int, default=100)  # 10e6)
	parser.add_argument("--nb_max_steps", default=1000, type=int)
	parser.add_argument("--save_interval", default=1000, type=int)
	parser.add_argument("--log_interval", default=1000, type=int)
	parser.add_argument("--test_outputfn", default=None, type=str)
	parser.add_argument("--save_path", default=None, type=str)
	parser.add_argument("--test", default=False, action="store_true")
	parser.add_argument("--summarize", default=False, action="store_true")
	return parser.parse_args()


if __name__ == "__main__":
	run(parse_args())
