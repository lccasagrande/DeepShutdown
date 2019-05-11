from pandas.core.config import is_callable
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
		self.gamma = gamma

	def run(self):
		# Here, we init the lists that will contain the mb of experiences
		mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, epinfos = [], [], [], [], [], [], []
		# For n in range number of steps
		for _ in range(self.nsteps):
			# Given observations, get action value and neglopacs
			# We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
			act_shut, act_sched, values, neglogpacs_shut, neglogpacs_sched = self.model.step(self.obs)
			actions = np.column_stack((act_shut, act_sched))
			mb_obs.append(self.obs.copy())
			mb_actions.append(actions)
			mb_values.append(values)
			mb_neglogpacs.append(np.column_stack((neglogpacs_shut, neglogpacs_sched)))
			mb_dones.append(self.dones)

			# Take actions in env and look the results
			# Infos contains a ton of useful informations
			self.obs[:], rewards, self.dones, infos = self.env.step(actions)
			for info in infos:
				maybeepinfo = info.get('episode')
				if maybeepinfo: epinfos.append(maybeepinfo)
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
		# print("[ACTU] Max: {} - Min: {}".format(np.max(mb_returns[:, 0]), np.min(mb_returns[:, 0])))
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
		self.summary_episode_interval = 50
		env = self._build_env(1)
		self.input_shape, self.action_shape = env.observation_space.shape, env.action_space.nvec
		env.close()

	def _train(self, global_step, clip_value, obs, returns, actions, values, neglogpacs, normalize=True):
		advs = returns - values
		advs = (advs - advs.mean()) / (advs.std() + 1e-8) if normalize else advs

		feed_dict = {
			self.obs: obs, self.returns: returns, self.actions_shut: actions[..., 0], self.actions_sched: actions[..., 1],
			self.advs: advs, self.old_neglogprob_shut: neglogpacs[..., 0], self.old_neglogprob_sched: neglogpacs[..., 1],
			self.old_vpred: values, self.global_step: global_step,
			self.clip_value: clip_value
		}
		ops = [self.train_op, self.p_loss, self.v_loss, self.entropy, self.learning_rate, self.clip_value]
		return self.session.run(fetches=ops, feed_dict=feed_dict)[1:]

	def _update(self, nupdate, clip_value, nbatch, epochs, batch_size, obs, returns, actions, values, neglogpacs):
		inds = np.arange(nbatch)
		results = []

		for _ in range(epochs):
			np.random.shuffle(inds)
			for start in range(0, nbatch, batch_size):
				ind = inds[start:start + batch_size]
				batch = (dt[ind] for dt in (obs, returns, actions, values, neglogpacs))

				results.append(self._train(nupdate, clip_value, *batch))

		stats = np.mean(results, axis=0)
		return {'p_loss': stats[0], 'v_loss': stats[1], 'p_entropy': stats[2], 'lr': stats[3]}

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
			shared=False, decay_steps=50, decay_rate=.1):
		with tf.name_scope("input"):
			self.obs = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name='observations')
			self.actions_shut = tf.placeholder(tf.int32, shape=[None], name='actions')
			self.actions_sched = tf.placeholder(tf.int32, shape=[None], name='actions')
			self.advs = tf.placeholder(tf.float32, shape=[None], name='advantages')
			self.returns = tf.placeholder(tf.float32, shape=[None], name='returns')
			self.global_step = tf.placeholder(tf.int32)

			# PPO specific
			self.clip_value = tf.placeholder(tf.float32)
			self.old_neglogprob_shut = old_neglogprob_shut = tf.placeholder(tf.float32, shape=[None])
			self.old_neglogprob_sched = old_neglogprob_sched = tf.placeholder(tf.float32, shape=[None])
			self.old_vpred = old_vpred = tf.placeholder(tf.float32, shape=[None])

			self.X = self.obs

		with tf.variable_scope("policy_network", reuse=tf.AUTO_REUSE):
			p_net = p_network(self.X)

		if v_network is not None:
			with tf.variable_scope("value_network", reuse=tf.AUTO_REUSE):
				v_net = v_network(self.X)
		elif shared:
			v_net = p_net
		else:  # is copy
			with tf.variable_scope("value_network", reuse=tf.AUTO_REUSE):
				v_net = p_network(self.X)

		self.p_net = tf.layers.flatten(p_net)
		self.v_net = tf.layers.flatten(v_net)

		# POLICY NETWORK SHUTDOWN
		self.p_logits_shut = tf.layers.dense(self.p_net, self.action_shape[0], name='policy_logits_2')
		self.act_probs_shut = tf.nn.softmax(self.p_logits_shut)
		u = tf.random_uniform(tf.shape(self.p_logits_shut), dtype=self.p_logits_shut.dtype)
		self.sample_action_shut = tf.argmax(self.p_logits_shut - tf.log(-tf.log(u)), axis=-1)
		# self.sample_action = tf.squeeze(tf.multinomial(tf.log(self.act_probs), 1))
		self.p_neglogprob_shut = sparse_softmax_cross_entropy_with_logits(self.p_logits_shut, self.sample_action_shut)

		# POLICY NETWORK SCHEDULER
		self.p_logits_sched = tf.layers.dense(self.p_net, self.action_shape[1], name='policy_logits')
		self.act_probs_sched = tf.nn.softmax(self.p_logits_sched)
		u = tf.random_uniform(tf.shape(self.p_logits_sched), dtype=self.p_logits_sched.dtype)
		self.sample_action_sched = tf.argmax(self.p_logits_sched - tf.log(-tf.log(u)), axis=-1)
		# self.sample_action = tf.squeeze(tf.multinomial(tf.log(self.act_probs), 1))
		self.p_neglogprob_sched = sparse_softmax_cross_entropy_with_logits(self.p_logits_sched,
		                                                                   self.sample_action_sched)

		# POLICY NETWORK LOSS (SHUT)
		self.neglogpac_shut = sparse_softmax_cross_entropy_with_logits(self.p_logits_shut, self.actions_shut)
		ratio = tf.exp(old_neglogprob_shut - self.neglogpac_shut)
		self.p_loss1_shut = -self.advs * ratio
		self.p_loss2_shut = -self.advs * tf.clip_by_value(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
		self.p_loss_shut = tf.reduce_mean(tf.maximum(self.p_loss1_shut, self.p_loss2_shut))

		# POLICY NETWORK LOSS (SCHED)
		self.neglogpac_sched = sparse_softmax_cross_entropy_with_logits(self.p_logits_sched, self.actions_sched)
		ratio = tf.exp(old_neglogprob_sched - self.neglogpac_sched)
		self.p_loss1_sched = -self.advs * ratio
		self.p_loss2_sched = -self.advs * tf.clip_by_value(ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
		self.p_loss_sched = tf.reduce_mean(tf.maximum(self.p_loss1_sched, self.p_loss2_sched))

		self.p_loss = self.p_loss_sched + self.p_loss_shut

		# ENTROPY
		self.entropy_sched = tf.reduce_mean(entropy(self.p_logits_sched))
		self.entropy_shut = tf.reduce_mean(entropy(self.p_logits_shut))
		self.entropy = self.entropy_shut + self.entropy_sched

		# VALUE NETWORK
		self.pred_value = tf.layers.dense(self.v_net, 1, name='value_logits')[:, 0]

		# VALUE NETWORK LOSS
		self.value_clipped = old_vpred + tf.clip_by_value(self.pred_value - old_vpred, -self.clip_value,
		                                                  self.clip_value)
		self.v_loss1 = tf.square(self.pred_value - self.returns)
		self.v_loss2 = tf.square(self.value_clipped - self.returns)
		self.v_loss = tf.reduce_mean(tf.maximum(self.v_loss1, self.v_loss2))

		# TRAIN
		self.loss = self.p_loss - self.entropy * ent_coef + self.v_loss * vf_coef
		self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps, decay_rate, staircase=False)
		optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)

		trainable_vars = tf.trainable_variables()
		gradients, variables = zip(*optimizer.compute_gradients(self.loss, trainable_vars))
		if max_grad_norm is not None:  # Clip the gradients (normalize)
			gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)

		grads_and_vars = list(zip(gradients, variables))
		self.train_op = optimizer.apply_gradients(grads_and_vars)

		# self.approxkl = .5 * tf.reduce_mean(tf.square(self.neglogpac - old_neglogprob))
		# self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(self.ratio - 1.0), self.clip_value), dtype=tf.float32))
		self.session.run(tf.global_variables_initializer())
		self._compiled = True

	def _pred(self, ops, obs):
		assert self._compiled
		if hasattr(self, 'rms'):
			self.session.run(self.rms.update_ops, {self.obs: obs})
		# if isinstance(ops, list):
		#	ops.append(self.rms.update_ops)
		#	return self.session.run(ops, {self.obs: obs})[:-1]
		# else:
		#	ops = [ops, self.rms.update_ops]
		#	return self.session.run(ops, {self.obs: obs})[0]
		# else:

		return self.session.run(ops, {self.obs: obs})

	def step(self, obs):
		return self._pred([self.sample_action_shut, self.sample_action_sched, self.pred_value, self.p_neglogprob_shut, self.p_neglogprob_sched], obs)

	def value(self, obs):
		return self._pred(self.pred_value, obs)

	def act(self, obs, argmax=False):
		if argmax:
			probs = self._pred([self.act_probs_shut, self.actions_sched], obs)
			return [[np.argmax(p_shut), np.argmax(p_sched)] for p_shut, p_sched in probs]
		return self._pred([self.sample_action_shut, self.sample_action_sched], obs)

	def fit(
			self, timesteps, nsteps, nb_batches=8, epochs=1, num_envs=1, log_interval=1,
			summary=False, loggers=None, gamma=.98, lam=.95, clip_value=0.2):
		assert self._compiled, "You should compile the model first."

		n_batch = nsteps * num_envs
		assert n_batch % nb_batches == 0

		# PREPARE
		env = self._build_env(num_envs)
		n_updates = int(timesteps // n_batch)
		batch_size = n_batch // nb_batches
		max_score, nepisodes = np.nan, 0
		history = deque(maxlen=self.summary_episode_interval) if self.summary_episode_interval > 0 else []
		runner = Runner(env=env, model=self, nsteps=nsteps, gamma=gamma, lam=lam)

		clip_value = clip_value if callable(clip_value) else constfn(clip_value)

		# START UPDATES
		try:
			for nupdate in range(1, n_updates + 1):
				tstart = time.time()
				frac = 1.0 - (nupdate - 1.0) / n_updates
				obs, acts, returns, values, dones, neglogps, infos = runner.run()
				stats = self._update(
					nupdate=nupdate,
					clip_value=clip_value(frac),
					nbatch=n_batch,
					epochs=epochs,
					batch_size=batch_size,
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
			action = self.act(obs, 0)
			obs, reward, done, info = env.step(action)
		# print("Obs: {} Action: {} Reward: {}".format('', action, reward))

		ep = info[0].pop('episode')
		if verbose:
			print("[EVAL] Score {:.4f} - Steps {:.4f} - Time {:.4f} sec".format(ep['score'], ep['nsteps'], ep['time']))
		info[0]['reward'] = ep['score']
		env.close()
		return info[0]
