import tensorflow as tf
import numpy as np
import time as tm
from collections import deque
from src.utils.agent import TFAgent
from collections import defaultdict
from src.utils.env_wrappers import SubprocVecEnv
from src.utils.common import discount, normalize, variable_summaries


class SimpleA2C(TFAgent):
	def __init__(self, network, **kwargs):
		super(SimpleA2C, self).__init__(**kwargs)
		self.compiled = False
		self.network = network
		self.memory = deque(maxlen=2000)
		self.tau = .125

	def build_model(self, obs_n, lr=1e-3):
		self._build_policy_network(obs_n, lr)
		self._build_value_network(obs_n, lr)
		self.session.run(tf.global_variables_initializer())
		self.compiled = True

	def _build_value_network(self, obs_n, lr=1e-3):
		with tf.variable_scope("value_network"):
			with tf.name_scope('input'):
				self.observations2 = tf.placeholder(dtype=tf.int32, shape=(1,))
				self.targets2 = tf.placeholder(dtype=tf.float32, shape=(1,))
				self.one_hot2 = tf.one_hot(self.observations2, obs_n)

			with tf.name_scope('network'):
				self.value_network = self.network(self.one_hot2)
				self.value_logits = tf.layers.dense(tf.layers.flatten(self.value_network), 1)
				self.value_estimate = tf.squeeze(self.value_logits, axis=0)
				self.value_loss = tf.losses.mean_squared_error(predictions=self.value_estimate, labels=self.targets2)
				self.value_optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.value_loss)

	def _build_policy_network(self, obs_n, lr=1e-3):
		with tf.variable_scope("policy_network"):
			with tf.name_scope('input'):
				self.observations = tf.placeholder(dtype=tf.int32, shape=(1,))
				self.actions = tf.placeholder(dtype=tf.int32, shape=(1,))
				self.targets = tf.placeholder(dtype=tf.float32, shape=(1,))
				self.one_hot = tf.one_hot(self.observations, obs_n)

			with tf.name_scope('network'):
				self.policy_network = self.network(self.one_hot)
				self.policy_logits = tf.layers.dense(tf.layers.flatten(self.policy_network), self.nb_actions)
				self.act_probs = tf.squeeze(tf.nn.softmax(self.policy_logits))
				self.picked_action = tf.gather(self.act_probs, self.actions)

				self.neglog2 = -tf.log(self.picked_action)
				self.loss2 = self.neglog2 * self.targets

				self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_logits,
				                                                                    labels=self.actions)
				self.policy_loss_norm = tf.multiply(self.cross_entropy, self.targets)
				self.policy_loss = tf.reduce_mean(self.policy_loss_norm)
				self.policy_optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.policy_loss)

	def update(self, state, target, action):
		feed_dict = {self.observations: [state], self.targets: [target], self.actions: [action]}
		_, a_p, p_a, nl2, l2, nl, l, lt = self.session.run(
			[self.policy_optimize, self.act_probs, self.picked_action, self.neglog2, self.loss2, self.cross_entropy,
			 self.policy_loss, self.policy_loss_norm], feed_dict=feed_dict)

		print("OK")

	def update_value(self, state, target):
		feed_dict = {self.observations2: [state], self.targets2: [target]}
		_, value, loss = self.session.run([self.value_optimize, self.value_estimate, self.value_loss], feed_dict=feed_dict)

		print("OK")


	def predict_value(self, state):
		assert self.compiled
		state_value = self.session.run(self.value_estimate, feed_dict={self.observations2: [state]})
		return state_value[0]

	def act(self, state):
		assert self.compiled
		act_probs, o_h = self.session.run([self.act_probs, self.one_hot], {self.observations: [state]})
		return np.random.choice(np.arange(self.nb_actions), p=act_probs)

	def evaluate(self, env, n_episodes, visualize=False):
		pass

	def fit(self, env, nb_iteration, nb_env, nb_max_steps, log_interval, save_interval, summarize=False):
		pass


class A2CAgent(TFAgent):
	def __init__(self, network, **kwargs):
		super(A2CAgent, self).__init__(**kwargs)
		self.compiled = False
		self.network = network
		self.memory = deque(maxlen=2000)
		self.tau = .125

	def _build_policy(self):
		with tf.name_scope("input"):
			self.observations = tf.placeholder(dtype=tf.float32, shape=(None,) + self.input_shape)

		with tf.name_scope('network'):
			with tf.variable_scope('policy', reuse=tf.AUTO_REUSE):
				self.policy_network = self.network(self.X)
				self.policy_logits = tf.layers.dense(tf.layers.Flatten(self.policy_network), self.nb_actions)

			with tf.variable_scope('value', reuse=tf.AUTO_REUSE):
				self.value_network = self.network(self.X)
				self.value_logits = tf.layers.dense(tf.layers.Flatten(self.value_network), 1)

		with tf.name_scope("output"):
			self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.policy_network,
			                                                                labels=self.actions)
			self.act_probs = tf.squeeze(tf.nn.softmax(self.policy_logits))
			self.state_value = self.value_logits

	def build_model(self, lr, batch_size=1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, alpha=0.99, epsilon=1e-5):
		self.actions = tf.placeholder(dtype=tf.int32, shape=(None,))
		self.rewards = tf.placeholder(dtype=tf.float32, shape=(None,))
		self.adv = tf.placeholder(dtype=tf.float32, shape=(None,))

		with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
			# step_model is used for sampling
			self.step_model = self._build_policy()

			# train_model is used to train our network
			self.train_model = self._build_policy()

		with tf.name_scope("train"):
			a0 = self.train_model.policy_logits - tf.reduce_max(self.self.train_model.policy_logits, axis=-1,
			                                                    keepdims=True)
			ea0 = tf.exp(a0)
			z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
			p0 = ea0 / z0
			self.entropy = tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)
			# L = A(s,a) * -logpi(a|s)
			self.policy_loss = tf.reduce_mean(self.adv * self.train_model.neglogpac)
			self.entropy_loss = tf.reduce_mean(self.entropy)
			self.value_loss = tf.losses.mean_squared_error(tf.squeeze(self.train_model.state_value), self.rewards)
			self.loss = self.policy_loss - self.entropy * ent_coef + self.value_loss * vf_coef
			params = tf.trainable_variables('a2c_model')
			self.grads = tf.gradients(self.loss, params)
			if max_grad_norm is not None:
				self.grads, grad_norm = tf.clip_by_global_norm(self.grads, max_grad_norm)

			self.grads = list(zip(self.grads, params))

			optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)

			self.train_op = optimizer.apply_gradients(self.grads)

	def act(self, state):
		pass

	def evaluate(self, env, n_episodes, visualize=False):
		pass

	def fit(self, env, nb_iteration, nb_env, nb_max_steps, log_interval, save_interval, summarize=False):
		pass
