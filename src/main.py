import gym
import gym_grid.envs.grid_env as g
import argparse
import os
from src.utils.networks import mlp
from src.utils.common import make_vec_env
from src.agents.reinforce import ReinforceAgent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run(args):
	test_env = gym.make(args.env)
	a = test_env.reset()
	train_env = make_vec_env(args.env, args.nb_env, args.seed)
	# env.seed(args.seed)
	network = mlp([20])
	agent = ReinforceAgent(network,
	                       gamma=1.0,
	                       input_shape=train_env.observation_space.shape,
	                       nb_actions=train_env.action_space.n,
	                       name="reinforce",
	                       seed=args.seed)

	agent.build_model(lr=args.lr)

	agent.fit(train_env,
	          args.nb_iters,
	          args.nb_epi,
	          args.log_interval,
	          args.save_interval,
	          args.nb_max_steps,
	          args.summarize)

	#test_env = gym.make(args.env)


# agent.load_model()
# agent.evaluate(test_env, args.nb_epi_test, args.render)
# test_env.close()
# train_env.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", type=str, default="grid-v0")
	parser.add_argument("--nb_iters", default=10, type=int)
	parser.add_argument("--nb_epi", default=20, type=int)
	parser.add_argument("--nb_env", default=12, type=int)
	parser.add_argument("--nb_max_steps", default=250, type=int)
	parser.add_argument("--nb_epi_test", default=1, type=int)
	parser.add_argument("--save_interval", default=10, type=int)
	parser.add_argument("--log_interval", default=10, type=int)
	parser.add_argument("--seed", default=123, type=int)
	parser.add_argument("--lr", default=0.001, type=float)
	parser.add_argument("--summarize", default=False, action="store_true")
	parser.add_argument("--render", default=False, action="store_true")
	args = parser.parse_args()
	run(args)
