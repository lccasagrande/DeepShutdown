import tensorflow as tf
import numpy as np
import time as tm
from src.utils.agent import TFAgent
from src.utils.common import discount, normalize, variable_summaries


class ReinforceAgent(TFAgent):
	def __init__(self, network, gamma, **kwargs):
		super(ReinforceAgent, self).__init__(**kwargs)
		self.network = network
		self.gamma = gamma
		self.compiled = False

	def build_model(self, lr):
		with tf.name_scope("input"):
			self.states = tf.placeholder(dtype=tf.float32, shape=(None,) + self.input_shape, name="state", )
			self.actions = tf.placeholder(dtype=tf.int32, shape=(None,), name="action")
			self.returns = tf.placeholder(dtype=tf.float32, shape=(None,), name="return")
			dataset = tf.data.Dataset.from_tensor_slices((self.states, self.actions, self.returns)).batch(1)
			self.iterator = dataset.make_initializable_iterator()
			self.X, self.Y, self.R = self.iterator.get_next()

		with tf.name_scope("policy"):
			with tf.name_scope("output"):
				self.logits = tf.layers.dense(self.network(self.X), self.nb_actions)
				self.act_probs = tf.squeeze(tf.nn.softmax(self.logits))

			with tf.name_scope("KL"):
				self.old_logits = tf.placeholder(dtype=tf.float32, shape=self.logits.shape)
				a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
				a1 = self.old_logits - tf.reduce_max(self.old_logits, axis=-1, keepdims=True)
				ea0 = tf.exp(a0)
				ea1 = tf.exp(a1)
				z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
				z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
				p0 = ea0 / z0
				self.kl = tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

			with tf.name_scope("entropy"):
				a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
				ea0 = tf.exp(a0)
				z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
				p0 = ea0 / z0
				self.entropy = tf.reduce_mean(tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1))

			with tf.name_scope("optimize"):
				self.global_step = tf.train.get_or_create_global_step()
				self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
				self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
				self.loss = tf.reduce_mean(self.cross_entropy) - 0.01 * self.entropy
				self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
				self.grads, _ = zip(*self.grads_and_vars)
				self.grads_holder, grads_and_vars_holder = [], []
				for grad, variable in self.grads_and_vars:
					grad_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
					self.grads_holder.append(grad_placeholder)
					grads_and_vars_holder.append((grad_placeholder, variable))

				# self.grads = [grad if grad is None else grad * self.R for grad in self.grads]
				#self.grads_and_vars = list(zip(self.grads, variables))
				self.apply_grads = self.optimizer.apply_gradients(grads_and_vars_holder, global_step=self.global_step)

			with tf.name_scope("summary"):
				for grad, var in self.grads_and_vars:
					tf.summary.histogram(var.name, var, family=var.name)
					if grad is not None: variable_summaries(grad, var.name)
				tf.summary.scalar("entropy", tf.reduce_mean(self.entropy))
				tf.summary.scalar("loss", self.loss)
				tf.summary.scalar("advantage", tf.reduce_mean(self.R))
				self.summarize = tf.summary.merge_all()

			self.session.run(tf.global_variables_initializer())
			self.compiled = True

	def act(self, state):
		assert self.compiled
		x = [state] if state.ndim == 1 else state
		act_probs = self.session.run(self.act_probs, {self.X: x})
		if act_probs.ndim == 1:
			return np.random.choice(np.arange(self.nb_actions), p=act_probs)
		else:
			return [np.random.choice(np.arange(self.nb_actions), p=p) for p in act_probs]

	def evaluate(self, env, n_episodes, visualize=False):
		assert self.compiled
		super(ReinforceAgent, self).evaluate(env, n_episodes, visualize)

	def fit(self, env, n_iterations, n_episodes, log_interval, save_interval, nb_max_steps, summarize=False):
		def get_trajectories(env, nb):
			trajectories, ob = dict(obs=[], acts=[], rews=[], dones=[]), env.reset()
			episodes = 0
			for _ in range(nb_max_steps):
				action = self.act(ob)
				trajectories['obs'].append(ob)
				trajectories['acts'].append(action)
				ob, rew, done, _ = env.step(action)
				trajectories['rews'].append(rew)
				trajectories["dones"].append(done)
				episodes += np.sum(done)
				if episodes >= nb:
					break
			return trajectories

		def calc_advantages(trajs_rews):
			max_trajectory = max(len(r) for r in trajs_rews)
			padded_rwds = [np.pad(r, (0, max_trajectory - len(r)), "constant", constant_values=0) for r in trajs_rews]
			baseline = np.mean(padded_rwds, axis=0)
			advs = np.concatenate([r - baseline[: len(r)] for r in trajs_rews])
			return advs  # normalize(advs)

		def train(states, actions, rewards):
			loss, entropy, steps, all_gradients = 0, 0, 0, []
			feed_dict = {self.states: states, self.actions: actions, self.returns: rewards}
			self.session.run(self.iterator.initializer, feed_dict=feed_dict)
			try:
				while True:
					grads, l, entpy = self.session.run([self.grads, self.loss, self.entropy])
					loss += l
					entropy += entpy
					steps += 1
					all_gradients.append(grads)
			# if summarize: self.writer.add_summary(summary, g_step)
			except tf.errors.OutOfRangeError:
				pass

			loss /= steps
			entropy /= steps
			feed_dict = {}
			for i, placeholder in enumerate(self.grads_holder):
				gr = [r * all_gradients[step][i] for step, r in enumerate(rewards)]
				mean_gradient = np.mean(gr, axis=0)
				feed_dict[placeholder] = mean_gradient
			self.session.run(self.apply_grads, feed_dict=feed_dict)

			return loss, entropy, steps

		def get_log_msg():
			log_msg = " ---------------------\
                         \n [*] [Iteration]: \t{}\
                         \n [*] [Total Steps]: \t{}\
                         \n [*] [Total Episodes]: \t{}\
                         \n [*] [Avg. Steps]: \t{}\
                         \n [*] [Avg. Loss]: \t{:.4f}\
                         \n [*] [Avg. Entropy]: \t{:.4f}\
                         \n [*] [Avg. Reward]: \t{:.2f}\
                         \n [*] [Max. Reward]: \t{:.2f}\
                         \n [*] [Elapsed Time]: \t{:.1f} s\
                         \n ---------------------".format(
				self._log["iteration"][-1],
				self._log["total_steps"][-1],
				self._log["total_episodes"][-1],
				self._log["avg_steps"][-1],
				self._log["avg_loss"][-1],
				self._log["avg_entropy"][-1],
				self._log["avg_reward"][-1],
				self._log["max_reward"][-1],
				self._log["elapsed_time"][-1],
			)
			return log_msg

		assert self.compiled
		total_steps = 0
		for iteration in range(1, n_iterations + 1):
			t_start = tm.time()
			trajectories = get_trajectories(env, n_episodes)
			agents_dones = [np.where(traj == True)[0] + 1 for traj in np.swapaxes(trajectories['dones'], 1, 0)]
			agents_obs, agents_acts, agents_rews = np.swapaxes(trajectories['obs'], 1, 0), np.swapaxes(
				trajectories['acts'], 1, 0), np.swapaxes(trajectories['rews'], 1, 0)
			trajs_obs, trajs_acts, trajs_rews = [], [], []
			for agent, agent_dones in enumerate(agents_dones):
				if len(agent_dones) == 0:
					trajs_obs.append(agents_obs[agent][:])
					trajs_acts.append(agents_acts[agent][:])
					trajs_rews.append(discount(agents_rews[agent][:], self.gamma))
				else:
					start = 0
					for traj_done in agent_dones:
						trajs_obs.append(agents_obs[agent][start:traj_done])
						trajs_acts.append(agents_acts[agent][start:traj_done])
						trajs_rews.append(discount(agents_rews[agent][start:traj_done], self.gamma))
						start = traj_done
						start = traj_done

			all_obs, all_acts, all_advs = (
				np.concatenate(trajs_obs), np.concatenate(trajs_acts), calc_advantages(trajs_rews))

			loss, entropy, steps = train(all_obs, all_acts, all_advs)

			elapsed_time = round(tm.time() - t_start, 2)
			total_steps += steps
			episodes_rewards, episodes_steps = zip(*[(r[0], len(r)) for r in trajs_rews])

			self._log["iteration"].append(iteration)
			self._log["total_episodes"].append(iteration * len(episodes_steps))
			self._log["total_steps"].append(total_steps)
			self._log["avg_loss"].append(loss)
			self._log["avg_entropy"].append(entropy)
			self._log["avg_steps"].append(np.mean(episodes_steps))
			self._log["avg_reward"].append(np.mean(episodes_rewards))
			self._log["max_reward"].append(np.max(episodes_rewards))
			self._log["elapsed_time"].append(elapsed_time)

			if iteration % log_interval == 0:
				self.save_log()

			if iteration % save_interval == 0:
				self.save_model(step=iteration)

			print(get_log_msg())

		self.save_model(step=iteration)
		env.close()
