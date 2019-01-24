import tensorflow as tf
import numpy as np
import gym
import time
from collections import defaultdict, deque
from src.utils.commons import safemean, discount_with_dones, softmax
from src.utils.runners import AbstractEnvRunner
from src.utils.agents import TFAgent
from src.utils.tf_utils import entropy, variable_summaries
from src.utils.env_wrappers import make_vec_env


class Runner(AbstractEnvRunner):
	def __init__(self, env, model, nsteps=5, gamma=0.99):
		super().__init__(env=env, model=model, nsteps=nsteps)
		self.gamma = gamma

	def run(self):
		def discount_rewards(envs_rewards, envs_dones):
			discounted_rewards = np.empty_like(envs_rewards, dtype=np.float32)
			if self.gamma > 0.0:
				_, last_obs_value = self.model.predict(self.obs)
				for i, (rewards, dones, value) in enumerate(zip(envs_rewards, envs_dones, last_obs_value)):
					rewards = rewards.tolist()
					dones = dones.tolist()
					if dones[-1]:
						discounted_rewards[i] = discount_with_dones(rewards, dones, self.gamma)
					else:
						discounted_rewards[i] = discount_with_dones(rewards + [value], dones + [False], self.gamma)[:-1]

			return discounted_rewards

		# We initialize the lists that will contain the mb of experiences
		rollout_actions, rollout_obs, rollout_rew, rollout_dones, rollout_values, epinfos = [], [], [], [], [], []
		for n in range(self.nsteps):
			# Given observations, take action and value (V(s))
			# We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
			actions, values = self.model.predict(self.obs)
			next_obs, rewards, dones, infos = self.env.step(actions)

			for info in infos:
				ep_info = info.get('episode')
				if ep_info:
					epinfos.append(ep_info)

			rollout_actions.append(actions)
			rollout_obs.append(np.copy(self.obs))
			rollout_rew.append(rewards)
			rollout_dones.append(dones)
			rollout_values.append(values)
			self.obs = next_obs

		envs_acts = np.asarray(rollout_actions).swapaxes(1, 0)
		envs_obs = np.asarray(rollout_obs).swapaxes(1, 0)
		envs_rewards = np.asarray(rollout_rew, dtype=np.float32).swapaxes(1, 0)
		envs_dones = np.asarray(rollout_dones, dtype=np.bool).swapaxes(1, 0)
		envs_values = np.asarray(rollout_values, dtype=np.float32).swapaxes(1, 0)
		envs_discounted_rewards = discount_rewards(envs_rewards, envs_dones)
		return envs_obs, envs_acts, envs_discounted_rewards, envs_rewards, envs_values, envs_dones, epinfos


class A2CAgent(TFAgent):
	def __init__(self, env_id, input_shape, nb_actions, network):
		super().__init__(
			env_id=env_id,
			input_shape=input_shape,
			nb_actions=nb_actions,
			network=network)
		self._compiled = False
		self.summary_episode_interval = 100

	def compile(self, lr=0.001, ent_coef=0.01, vf_coef=.5, max_grad_norm=.5):
		with tf.name_scope("input"):
			self.obs = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name='observations')
			self.mask = tf.placeholder(tf.float32, shape=(None,) + self.input_shape, name='mask')
			self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')
			self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')
			self.advs = tf.placeholder(tf.float32, shape=[None], name='advantages')

			self.X = self.obs  # Do some preprocessing here
			self.policy_labels = self.actions  # Do some preprocessing here
			self.value_labels = self.rewards  # Do some preprocessing here

		with tf.variable_scope("policy_network", reuse=tf.AUTO_REUSE):
			self.policy_network = tf.layers.dense(self._network(self.X), self.nb_actions)
			self.act_probs = tf.nn.softmax(self.policy_network)
			self.policy_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.policy_labels,
			                                                                logits=self.policy_network)

			self.entropy = entropy(self.policy_network)
			self.policy_loss = tf.reduce_mean(self.policy_ce * self.advs)

		with tf.variable_scope("value_network", reuse=tf.AUTO_REUSE):
			self.value_network = tf.layers.dense(self._network(self.X), 1)[:, 0]
			self.value_loss = tf.losses.mean_squared_error(labels=self.value_labels, predictions=self.value_network)
			self.mae = tf.losses.absolute_difference(labels=self.value_labels, predictions=self.value_network)

		self.loss = self.policy_loss + vf_coef * self.value_loss - ent_coef * self.entropy
		trainable_vars = tf.trainable_variables()
		gradients = tf.gradients(self.loss, trainable_vars)
		if max_grad_norm is not None:  # Clip the gradients (normalize)
			gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)

		gradients = list(zip(gradients, trainable_vars))
		self.train_op = tf.train.AdamOptimizer(learning_rate=lr).apply_gradients(gradients)
		self.session.run(tf.global_variables_initializer())
		self._compiled = True

	def act(self, obs):
		actions, _ = self.predict(obs)
		return actions

	def predict(self, obs, argmax=False):
		assert self._compiled

		act_probs, value, policy = self.session.run([self.act_probs, self.value_network, self.policy_network],
		                                    feed_dict={self.obs: obs})

		#act_probs = [softmax(policy[i, :self.get_nb_valid_actions(o)]) for i, o in enumerate(obs)]
		actions = [np.argmax(p) if argmax else np.random.choice(np.arange(len(p)), p=p) for p in act_probs]
		return actions, value

	def _update(self, n_batch, obs, rewards, advantages, actions):
		o = obs.reshape((n_batch,) + self.input_shape)
		r = rewards.flatten()
		a = advantages.flatten()
		acts = actions.flatten()

		_, policy_loss, entropy, value_loss, mae = self.session.run(
			[self.train_op, self.policy_loss, self.entropy, self.value_loss, self.mae],
			feed_dict={self.obs: o, self.advs: a, self.actions: acts, self.rewards: r}
		)

		return policy_loss, value_loss, mae, entropy

	def fit(self, timesteps, nsteps, discount=1., num_envs=1, log_interval=10, summary=False, seed=None, loggers=None,
	        monitor_dir=None):
		assert self._compiled, "You should compile the model first."

		# PREPARE
		env = make_vec_env(self.env_id, num_envs, seed, monitor_dir=monitor_dir)
		n_batch = nsteps * env.num_envs
		n_updates = int(timesteps // n_batch)
		runner = Runner(env, self, nsteps, discount)
		nepisodes = 0
		history = deque(maxlen=self.summary_episode_interval) if self.summary_episode_interval > 0 else []
		tstart = time.time()

		# START UPDATES
		try:
			for update in range(1, n_updates + 1):
				obs, acts, disc_rewards, rewards, values, dones, infos = runner.run()
				advantages = disc_rewards - values
				p_loss, v_loss, mae, entropy = self._update(n_batch, obs, disc_rewards, advantages, acts)

				nseconds = time.time() - tstart
				nepisodes += len(infos)
				history.extend(infos)

				if loggers is not None and (update % log_interval == 0 or update == 1):
					loggers.log('nupdates', update)
					loggers.log('ntimesteps', update * n_batch)
					loggers.log('nepisodes', nepisodes)
					loggers.log('fps', int((update * n_batch) / nseconds))
					loggers.log('policy_entropy', float(entropy))
					loggers.log('policy_loss', float(p_loss))
					loggers.log('value_loss', float(v_loss))
					loggers.log('value_mae', float(mae))
					loggers.log('eprewmean', safemean([h['score'] for h in history]))
					loggers.log('eplenmean', safemean([h['nsteps'] for h in history]))
					loggers.log('eprewmax', np.max([h['score'] for h in history]) if history else np.nan)
					loggers.log('eprewmin', np.min([h['score'] for h in history]) if history else np.nan)
					loggers.dump()
		finally:
			env.close()
			if loggers is not None:
				loggers.close()
			if summary:
				self.writer.close()

		return history

	def play(self, render, verbose):
		env = gym.make(self.env_id)
		obs = env.reset()
		done = False
		epi_history = dict(score=0, lenght=0)
		while not done:
			if render:
				env.render()
			actions, _ = self.predict([obs], argmax=True)
			print(obs, actions)
			obs, reward, done, info = env.step(actions[0])
			epi_history['score'] += reward
			epi_history['lenght'] += 1

		if verbose:
			print("Evaluation - " + " ".join("{}: {}".format(k, v) for k, v in epi_history.items()))

		env.close()
		return epi_history
