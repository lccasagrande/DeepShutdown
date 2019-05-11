import tensorflow as tf
import numpy as np
import time as tm
from utils.agents import TFAgent
from collections import defaultdict
from utils.env_wrappers import SubprocVecEnv
from utils.commons import discount, normalize
from src.utils.commons import *
from src.utils.runners import AbstractEnvRunner
from src.utils.agents import TFAgent
from src.utils.tf_utils import *
from src.utils.env_wrappers import *


class FixedRunner(AbstractEnvRunner):
	def __init__(self, env, model, gamma=0.99):
		super().__init__(env=env, model=model, nsteps=-1)
		self.gamma = gamma

	def run(self):
		def discount(rewards):
			discounted, r = [], 0
			for reward in rewards[::-1]:
				r = reward + self.gamma * r
				discounted.append(r)
			return discounted[::-1]

		rollout_actions, rollout_obs, rollout_rew, epinfos, trajectories = [], [], [], [], []
		self.dones = [False] * self.nenv
		obs = self.env.reset()
		while any(not done for done in self.dones):
			actions = [0] * self.nenv
			rollout_actions.append(actions)
			rollout_obs.append(np.copy(obs))
			obs, rewards, dones, infos = self.env.step(actions)
			rollout_rew.append(rewards)

			for i, done in enumerate(dones):
				if done and not self.dones[i]:
					acts = np.asarray(rollout_actions).swapaxes(0, 1)[i]
					o = np.asarray(rollout_obs).swapaxes(0, 1)[i]
					rets = discount(np.asarray(rollout_rew).swapaxes(0, 1)[i])
					trajectories.append((o, acts, rets))
					self.dones[i] = True

		return trajectories

class Runner(AbstractEnvRunner):
	def __init__(self, env, model, gamma=0.99):
		super().__init__(env=env, model=model, nsteps=-1)
		self.gamma = gamma

	def run(self):
		def discount(rewards):
			discounted, r = [], 0
			for reward in rewards[::-1]:
				r = reward + self.gamma * r
				discounted.append(r)
			return discounted[::-1]

		rollout_actions, rollout_obs, rollout_rew, epinfos, trajectories = [], [], [], [], []
		self.dones = [False] * self.nenv
		obs = self.env.reset()
		while any(not done for done in self.dones):
			actions = self.model.act(obs)
			rollout_actions.append(actions)
			rollout_obs.append(np.copy(obs))
			obs, rewards, dones, infos = self.env.step(actions)
			rollout_rew.append(rewards)

			for i, done in enumerate(dones):
				if done and not self.dones[i]:
					acts = np.asarray(rollout_actions).swapaxes(0, 1)[i]
					o = np.asarray(rollout_obs).swapaxes(0, 1)[i]
					rets = discount(np.asarray(rollout_rew).swapaxes(0, 1)[i])
					trajectories.append((o, acts, rets))
					self.dones[i] = True
			for info in infos:
				ep_info = info.get('episode')
				if ep_info:
					epinfos.append(ep_info)
		return trajectories, epinfos


class ReinforceAgent(TFAgent):
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

	def _build_env(self, n):
		env = make_vec_env(env_id=self.env_id, nenv=n, seed=self.seed, monitor_dir=self.monitor_dir)
		if self.normalize_obs:
			env = VecNormalize(env, ret=False, clipob=self.clip_obs)
		if self.nframes > 1:
			env = VecFrameStack(env, self.nframes, include_action=False)
		return env

	def _update(self, nupdate, trajectories):
		stats = []
		for o, a, r in trajectories:
			baseline = self.session.run(self.value_estimate, {self.obs: o})
			adv = r - baseline
			adv = (adv - adv.mean()) / (adv.std() + 1e-8)
			feed_dict = {self.obs: o, self.actions: a, self.returns: r, self.advs: adv, self.global_step: nupdate}
			ops = [self.train_policy, self.train_value, self.p_loss, self.v_loss, self.entropy, self.learning_rate]
			stats.append(self.session.run(fetches=ops, feed_dict=feed_dict)[2:])

		s = np.mean(stats, axis=0)
		return {"p_loss": s[0], "v_loss": s[1], "entropy": s[2], "lr": s[3]}

	def trainv(self, n, gamma):
		env = self._build_env(1)
		runner = FixedRunner(env=env, model=self, gamma=gamma)
		trajectory = runner.run()

		# START UPDATES
		for update in range(1, n + 1):
			loss = 0
			for o, a, r in trajectory:
				feed_dict = {self.obs: o, self.returns: r, self.global_step: 1}
				ops = [self.train_value, self.v_loss]
				loss = self.session.run(fetches=ops, feed_dict=feed_dict)[1:]

			print("Loss: {}".format(loss))



	def compile(
			self, network, lr=0.01, ent_coef=0.01, max_grad_norm=0.5, decay_steps=50, decay_rate=.1):

		with tf.name_scope("input"):
			self.obs = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name='observations')
			self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')
			self.advs = tf.placeholder(tf.float32, shape=[None], name='returns')
			self.returns = tf.placeholder(tf.float32, shape=[None], name='returns')
			self.global_step = tf.placeholder(tf.int32)

		with tf.variable_scope("policy_network", reuse=tf.AUTO_REUSE):
			p_net = network(self.obs)

		with tf.variable_scope("value_network", reuse=tf.AUTO_REUSE):
			v_net = network(self.obs)

		# POLICY
		self.p_net = tf.layers.flatten(p_net)
		self.p_logits = tf.layers.dense(self.p_net, self.nb_actions, name='policy_logits')
		self.act_probs = tf.nn.softmax(self.p_logits)
		self.x_entropy = sparse_softmax_cross_entropy_with_logits(logits=self.p_logits, labels=self.actions)
		self.p_loss = tf.reduce_mean(self.x_entropy * self.advs)

		# VALUE
		self.v_net = tf.layers.flatten(v_net)
		self.value_estimate = tf.layers.dense(self.v_net, 1, name='value_logits')[:, 0]
		self.v_loss = tf.reduce_mean(tf.squared_difference(self.value_estimate, self.returns))


		# TRAIN
		self.learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_steps, decay_rate)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)

		#self.p_grads = optimizer.compute_gradients(self.p_loss)
		#for i, (grad, var) in enumerate(self.p_grads):
		#	if grad is not None:
		#		self.p_grads[i] = (grad * self.advs, var)

		self.train_policy = optimizer.minimize(self.p_loss)
		self.train_value = optimizer.minimize(self.v_loss)

		self.entropy = tf.reduce_mean(entropy(self.p_logits))
		self.session.run(tf.global_variables_initializer())
		self._compiled = True

	def act(self, obs):
		assert self._compiled
		probs = self.session.run(self.act_probs, {self.obs: obs})
		return [np.random.choice(np.arange(self.nb_actions), p=p) for p in probs]

	def fit(self, nb_updates, num_envs=1, log_interval=1, loggers=None, gamma=.98):
		assert self._compiled, "You should compile the model first."

		# PREPARE
		env = self._build_env(num_envs)
		max_score = np.nan
		nepisodes = 0
		history = deque(maxlen=self.summary_episode_interval) if self.summary_episode_interval > 0 else []
		runner = Runner(env=env, model=self, gamma=gamma)

		# START UPDATES
		try:
			for update in range(1, nb_updates + 1):
				tstart = time.time()
				trajectories, infos = runner.run()
				stats = self._update(nupdate=update, trajectories=trajectories)

				elapsed_time = time.time() - tstart
				history.extend(infos)
				nepisodes += len(infos)
				eprew_max = np.max([h['score'] for h in history]) if history else np.nan
				if max_score is np.nan:
					max_score = eprew_max
				elif max_score < eprew_max:
					max_score = eprew_max
				# self.save("../weights/best_ppo_shutdown")
				if loggers is not None and (update % log_interval == 0 or update == 1):
					loggers.log('nbupdates', update)
					loggers.log('elapsed_time', elapsed_time)
					loggers.log('nepisodes', nepisodes)
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
			action = self.act(obs)
			obs, reward, done, info = env.step(action)
		# print("Obs: {} Action: {} Reward: {}".format('', action, reward))

		ep = info[0].pop('episode')
		if verbose:
			print("[EVAL] Score {:.4f} - Steps {:.4f} - Time {:.4f} sec".format(ep['score'], ep['nsteps'], ep['time']))
		info[0]['reward'] = ep['score']
		env.close()
		return info[0]
