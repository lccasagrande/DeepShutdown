from pandas.core.config import is_callable
from pygments.lexer import default
import tensorflow as tf
import numpy as np
import gym
import time
from collections import defaultdict, deque
import os
import joblib
from src.utils.commons import *
from src.utils.runners import AbstractEnvRunner
from src.utils.agents import TFAgent
from src.utils.tf_utils import *
from src.utils.env_wrappers import *


def constfn(val):
	def f(_):
		return val

	return f


class Runner(AbstractEnvRunner):
	"""
	We use this object to make a mini batch of experiences
	__init__:
	- Initialize the runner

	run():
	- Make a mini batch
	"""

	def __init__(self, *, env, model, nsteps, gamma, lam):
		super().__init__(env=env, model=model, nsteps=nsteps)
		self.lam = lam
		self.best = defaultdict(lambda: [-np.inf, -1])
		self.gamma = gamma

	def run(self):
		# Here, we init the lists that will contain the mb of experiences
		mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, epinfos = [], [], [], [], [], [], []
		# For n in range number of steps
		for _ in range(self.nsteps):
			# Given observations, get action value and neglopacs
			# We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
			actions, values, neglogpacs = self.model.step(self.obs)
			mb_obs.append(self.obs.copy())
			mb_actions.append(actions)
			mb_values.append(values)
			mb_neglogpacs.append(neglogpacs)
			mb_dones.append(self.dones)

			# Take actions in env and look the results
			# Infos contains a ton of useful informations
			self.obs[:], rewards, self.dones, infos = self.env.step(actions)
			for e, info in enumerate(infos):
				maybeepinfo = info.get('episode')
				if maybeepinfo:
					epinfos.append(maybeepinfo)
			mb_rewards.append(rewards)
		# batch of steps to batch of rollouts
		mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
		mb_actions = np.asarray(mb_actions)
		mb_values = np.asarray(mb_values, dtype=np.float32)
		mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
		mb_dones = np.asarray(mb_dones, dtype=np.bool)
		last_values = self.model.value(self.obs)

		# discount/bootstrap off value fn
		mb_advs = np.zeros_like(mb_rewards)
		lastgaelam = 0
		for t in reversed(range(self.nsteps)):
			if t == self.nsteps - 1:
				nextnonterminal = 1.0 - self.dones
				nextvalues = last_values
			else:
				nextnonterminal = 1.0 - mb_dones[t + 1]
				nextvalues = mb_values[t + 1]
			delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
			mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
		mb_returns = mb_advs + mb_values

		return (*map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_dones, mb_neglogpacs)), epinfos)


def sf01(arr):
	"""
	swap and then flatten axes 0 and 1
	"""
	s = arr.shape
	return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


class PPOAgent(TFAgent):
	def __init__(self, env_id, seed=None, nbframes=1, monitor_dir=None, normalize_obs=True, clip_obs=10.):
		super().__init__(env_id, seed)
		self.nframes = nbframes
		self._compiled = False
		self.monitor_dir = monitor_dir
		self.normalize_obs = normalize_obs
		self.clip_obs = clip_obs
		self.summary_episode_interval = 100
		env = self._build_env(1)
		self.input_shape, self.nb_actions = env.observation_space.shape, env.action_space.n
		env.close()

	def _update(self, n, clip_value, obs, returns, actions, values, neglogpacs, normalize=False):
		results = []
		advs = returns - values
		advs = (advs - advs.mean()) / (advs.std() + 1e-8) if normalize else advs

		self.session.run(self.data_iterator.initializer, feed_dict={
			self.obs: obs, self.returns: returns, self.actions: actions, self.advs: advs,
			self.old_neglogprob: neglogpacs, self.old_vpred: values
		})

		try:
			ops = [
				self.train_op, self.p_loss, self.v_loss, self.entropy, self.approxkl, self.clipfrac,
				self.learning_rate, self.clip_value
			]
			while True:
				results.append(self.session.run(ops, feed_dict={self.clip_value: clip_value, self.global_step: n})[1:])
		except tf.errors.OutOfRangeError:
			pass

		stats = np.mean(results, axis=0)
		stats = {
			'p_loss': stats[0], 'v_loss': stats[1], 'p_entropy': stats[2],
			'approxkl': stats[3], 'clip_frac': stats[4], 'lr': stats[5], 'clip_value': stats[6]
		}
		return stats

	def _build_env(self, n):
		env = make_vec_env(env_id=self.env_id, nenv=n, seed=self.seed, monitor_dir=self.monitor_dir)
		if self.normalize_obs:
			env = VecNormalize(env, ret=False, clipob=self.clip_obs)
		if self.nframes > 1:
			env = VecFrameStack(env, self.nframes, include_action=False)
		return env

	def load_value(self, fn):
		network = tf.trainable_variables("value_network")
		network.extend(tf.trainable_variables("value_logits"))
		values = joblib.load(os.path.expanduser(fn))
		restores = [v.assign(values[v.name]) for v in network]
		self.session.run(restores)

	def compile(
			self, p_network, v_network=None, lr=0.01, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
			shared=False, decay_steps=50, decay_rate=.1, batch_size=32, epochs=1):
		with tf.name_scope("input"):
			self.obs = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name='observations')
			self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')
			self.advs = tf.placeholder(tf.float32, shape=[None], name='advantages')
			self.returns = tf.placeholder(tf.float32, shape=[None], name='returns')
			self.old_neglogprob = tf.placeholder(tf.float32, shape=[None], name='oldneglogprob')
			self.old_vpred = tf.placeholder(tf.float32, shape=[None], name='oldvalueprediction')

			self.global_step = tf.placeholder(tf.int32)
			self.clip_value = tf.placeholder(tf.float32)

			dataset = tf.data.Dataset.from_tensor_slices(
				(self.obs, self.actions, self.advs, self.returns, self.old_neglogprob, self.old_vpred))
			dataset = dataset.shuffle(buffer_size=50000)
			dataset = dataset.batch(batch_size)
			dataset = dataset.repeat(epochs)
			self.data_iterator = dataset.make_initializable_iterator()

			self.X, ACT, ADV, RET, OLD_NEGLOGPROB, OLD_VPRED = self.data_iterator.get_next()

		with tf.variable_scope("policy_network", reuse=tf.AUTO_REUSE):
			self.p_net = p_network(self.X)

		if v_network is not None:
			with tf.variable_scope("value_network", reuse=tf.AUTO_REUSE):
				self.v_net = v_network(self.X)
		elif shared:
			self.v_net = self.p_net
		else:  # is copy
			with tf.variable_scope("value_network", reuse=tf.AUTO_REUSE):
				self.v_net = p_network(self.X)

		# POLICY NETWORK
		self.p_logits = tf.layers.dense(self.p_net, self.nb_actions, name='policy_logits')
		self.act_probs = tf.nn.softmax(self.p_logits)
		u = tf.random_uniform(tf.shape(self.p_logits), dtype=self.p_logits.dtype)
		self.sample_action = tf.argmax(self.p_logits - tf.log(-tf.log(u)), axis=-1)
		# self.sample_action = tf.squeeze(tf.multinomial(tf.log(self.act_probs), 1))

		self.p_neglogprob = sparse_softmax_cross_entropy_with_logits(self.p_logits, self.sample_action)

		# VALUE NETWORK
		self.pred_value = tf.layers.dense(self.v_net, 1, name='value_logits')[:, 0]

		# VALUE NETWORK LOSS
		self.value_clipped = OLD_VPRED + tf.clip_by_value(self.pred_value - OLD_VPRED, -self.clip_value,
		                                                  self.clip_value)
		self.v_loss1 = tf.square(self.pred_value - RET)
		self.v_loss2 = tf.square(self.value_clipped - RET)
		self.v_loss = tf.reduce_mean(tf.maximum(self.v_loss1, self.v_loss2))

		# POLICY NETWORK LOSS
		self.neglogpac = sparse_softmax_cross_entropy_with_logits(self.p_logits, ACT)
		self.ratio = tf.exp(OLD_NEGLOGPROB - self.neglogpac)
		self.p_loss1 = -ADV * self.ratio
		self.p_loss2 = -ADV * tf.clip_by_value(self.ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
		self.p_loss = tf.reduce_mean(tf.maximum(self.p_loss1, self.p_loss2))

		# ENTROPY
		self.entropy = entropy(self.p_logits)
		self.entropy = tf.reduce_mean(self.entropy)

		# TRAIN
		self.loss = self.p_loss - self.entropy * ent_coef + self.v_loss * vf_coef
		self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps, decay_rate, staircase=False)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)

		trainable_vars = tf.trainable_variables()
		gradients, variables = zip(*optimizer.compute_gradients(self.loss, trainable_vars))
		if max_grad_norm is not None:  # Clip the gradients (normalize)
			gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)

		grads_and_vars = list(zip(gradients, variables))
		self.train_op = optimizer.apply_gradients(grads_and_vars)

		self.approxkl = .5 * tf.reduce_mean(tf.square(self.neglogpac - OLD_NEGLOGPROB))
		self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(self.ratio - 1.0), self.clip_value), dtype=tf.float32))
		self.session.run(tf.global_variables_initializer())
		self._compiled = True

	def _pred(self, ops, obs):
		assert self._compiled
		return self.session.run(ops, {self.X: obs})

	def step(self, obs):
		return self._pred([self.sample_action, self.pred_value, self.p_neglogprob], obs)

	def value(self, obs):
		return self._pred(self.pred_value, obs)

	def act(self, obs, argmax=False):
		if argmax:
			probs = self._pred(self.act_probs, obs)
			return [np.argmax(p) for p in probs]
		return self._pred(self.sample_action, obs)

	def fit(
			self, timesteps, nsteps, nb_batches=8, num_envs=1, log_interval=1,
			summary=False, loggers=None, gamma=.98, lam=.95, clip_value=0.2):
		assert self._compiled, "You should compile the model first."
		n_batch = nsteps * num_envs
		assert n_batch % nb_batches == 0

		n_updates = int(timesteps // n_batch)

		max_score, nepisodes = np.nan, 0
		history = deque(maxlen=self.summary_episode_interval) if self.summary_episode_interval > 0 else []
		clip_value = clip_value if callable(clip_value) else constfn(clip_value)

		# START UPDATES
		try:
			env = self._build_env(num_envs)
			runner = Runner(env=env, model=self, nsteps=nsteps, gamma=gamma, lam=lam)
			for nupdate in range(1, n_updates + 1):
				tstart = time.time()
				frac = 1.0 - (nupdate - 1.0) / n_updates
				obs, acts, returns, values, dones, neglogps, infos = runner.run()
				stats = self._update(
					n=nupdate,
					clip_value=clip_value(frac),
					obs=obs,
					returns=returns,
					actions=acts,
					values=values,
					neglogpacs=neglogps)

				elapsed_time = time.time() - tstart
				nepisodes += len(infos)
				history.extend(infos)
				eprew_max = np.max([h['score'] for h in history]) if history else np.nan
				if max_score is np.nan:
					max_score = eprew_max
				elif max_score < eprew_max:
					max_score = eprew_max
				# self.save("../weights/best_ppo_shutdown")
				if loggers is not None and (nupdate % log_interval == 0 or nupdate == 1):
					loggers.log('v_explained_variance', round(explained_variance(values, returns), 4))
					loggers.log('frac', frac)
					loggers.log('nupdates', nupdate)
					loggers.log('elapsed_time', elapsed_time)
					loggers.log('ntimesteps', nupdate * n_batch)
					loggers.log('nepisodes', nepisodes)
					loggers.log('fps', int(n_batch / elapsed_time))
					loggers.log('eprew_avg', safemean([h['score'] for h in history]))
					loggers.log('eplen_avg', safemean([h['nsteps'] for h in history]))
					loggers.log('eprew_max', float(eprew_max))
					loggers.log('eprew_max_score', float(max_score))
					loggers.log('eprew_min', int(np.min([h['score'] for h in history])) if history else np.nan)
					for key, value in stats.items():
						loggers.log(key, float(value))
					loggers.dump()
		finally:
			env.close()
			if loggers is not None:
				loggers.close()

		return history

	def play(self, render, verbose):
		env = self._build_env(1)
		obs, done = env.reset(), False
		while not done:
			if render:
				# env.render('console')
				env.render()
			action = self.act(obs, True)
			obs, reward, done, info = env.step(action)
		# print("Obs: {} Action: {} Reward: {}".format('', action, reward))

		ep = info[0].pop('episode')
		if verbose:
			print("[EVAL] Score {:.4f} - Steps {:.4f} - Time {:.4f} sec".format(ep['score'], ep['nsteps'], ep['time']))
		info[0]['reward'] = ep['score']
		env.close()
		return info[0]
