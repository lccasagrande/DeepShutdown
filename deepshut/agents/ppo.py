import os

import pandas as pd
import joblib
from tqdm import tqdm

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
    def __init__(self, env, model, gamma, lam):
        super().__init__(env=env, model=model)
        self.lam = lam
        self.gamma = gamma

    def get_experiences(self, nsteps):
        def swap_and_flat(arr):
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, epinfos = [
        ], [], [], [], [], [], []
        # For n in range number of steps
        for _ in range(nsteps):
            actions, values, neglogpacs, _ = self.model.step(self.obs)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
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
        last_values = self.model.predict_value(self.obs)

        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            if t == nsteps - 1:
                nextnonterminal = (1.0 - self.dones)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]
            delta = mb_rewards[t] + self.gamma * \
                nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * \
                self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values
        return (*map(swap_and_flat, (mb_obs, mb_actions, mb_returns, mb_values, mb_dones, mb_neglogpacs)), epinfos)


class PPOAgent(TFAgent):
    def __init__(self, seed=None):
        super().__init__(seed)
        self._is_compiled = False

    def compile(
            self, input_shape, nb_actions, p_network, v_network=None, lr=0.01, end_lr=.0001, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            shared=False, decay_steps=50, decay_rate=.1, batch_size=32, epochs=1):
        tf.reset_default_graph()
        with tf.name_scope("input"):
            self.obs = tf.placeholder(tf.float32, shape=(
                None,) + input_shape, name='observations')
            self.actions = tf.placeholder(
                tf.int32, shape=[None], name='actions')
            self.advs = tf.placeholder(
                tf.float32, shape=[None], name='advantages')
            self.returns = tf.placeholder(
                tf.float32, shape=[None], name='returns')
            self.old_neglogprob = tf.placeholder(
                tf.float32, shape=[None], name='oldneglogprob')
            self.old_vpred = tf.placeholder(
                tf.float32, shape=[None], name='oldvalueprediction')

            self.global_step = tf.placeholder(tf.int32)
            self.clip_value = tf.placeholder(tf.float32)

            dataset = tf.data.Dataset.from_tensor_slices((
                self.obs,
                self.actions,
                self.advs,
                self.returns,
                self.old_neglogprob,
                self.old_vpred))
            dataset = dataset.shuffle(buffer_size=50000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat(epochs)
            self.data_iterator = dataset.make_initializable_iterator()
            self.X, self.ACT, ADV, RET, OLD_NEGLOGPAC, OLD_VPRED = self.data_iterator.get_next()

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
        p_logits = tf.keras.layers.Dense(
            nb_actions, name='policy_logits')(self.p_net)
        self.act_probs = tf.keras.layers.Softmax()(p_logits)
        prob_dist = tf.distributions.Categorical(probs=self.act_probs)
        self.sample_action = prob_dist.sample()

        self.best_action = tf.squeeze(tf.nn.top_k(self.act_probs).indices)
        self.entropy = tf.reduce_mean(prob_dist.entropy())
        self.neglogpac = sparse_softmax_cross_entropy_with_logits(
            p_logits, self.ACT)
        self.neglogprob = sparse_softmax_cross_entropy_with_logits(
            p_logits, self.sample_action)

        self.ratio = tf.exp(OLD_NEGLOGPAC - self.neglogpac)
        self.ratio_clipped = tf.clip_by_value(
            self.ratio, 1.0 - self.clip_value, 1.0 + self.clip_value)
        self.p_loss1 = -ADV * self.ratio
        self.p_loss2 = -ADV * self.ratio_clipped
        self.p_loss_max = tf.maximum(self.p_loss1, self.p_loss2)
        self.p_loss = tf.reduce_mean(self.p_loss_max)

        # VALUE NETWORK
        self.v_logits = tf.keras.layers.Dense(
            1, name='value_logits')(self.v_net)[:, 0]
        self.v_clipped = OLD_VPRED + \
            tf.clip_by_value(self.v_logits - OLD_VPRED, -
                             self.clip_value, self.clip_value)
        self.v_loss1 = tf.math.squared_difference(self.v_logits, RET)  #
        self.v_loss2 = tf.math.squared_difference(self.v_clipped, RET)  #
        self.v_loss = tf.reduce_mean(tf.maximum(self.v_loss1, self.v_loss2))

        self.loss = self.p_loss - self.entropy * ent_coef + self.v_loss * vf_coef

        self.learning_rate = tf.train.exponential_decay(
            lr, self.global_step, decay_steps, decay_rate, staircase=False)
        self.learning_rate = tf.maximum(self.learning_rate, end_lr)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=1e-5)

        trainable_vars = tf.trainable_variables()
        gradients, variables = zip(
            *optimizer.compute_gradients(self.loss, trainable_vars))
        if max_grad_norm is not None:  # Clip the gradients (normalize)
            gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)

        grads_and_vars = list(zip(gradients, variables))
        self.train_op = optimizer.apply_gradients(grads_and_vars)

        self.approxkl = .5 * \
            tf.reduce_mean(tf.square(self.neglogpac - OLD_NEGLOGPAC))
        self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(
            tf.abs(self.ratio - 1.0), self.clip_value), dtype=tf.float32))
        self.session.run(tf.global_variables_initializer())

        self._is_compiled = True

    def step(self, obs):
        assert self._is_compiled
        return self.session.run([self.sample_action, self.v_logits, self.neglogprob, self.act_probs], {self.X: obs})

    def predict_value(self, obs):
        return self.session.run(self.v_logits, {self.X: obs})

    def act(self, obs, argmax=False):
        assert self._is_compiled
        op = self.sample_action if not argmax else self.best_action
        return self.session.run(op, {self.X: obs})

    def fit(self, env, timesteps, nsteps, nb_batches=8, log_interval=1, loggers=None, gamma=.98, lam=.95, clip_vl=0.2):
        assert self._is_compiled, "You should compile the model first."
        n_batch = nsteps * env.num_envs
        assert n_batch % nb_batches == 0

        # START
        try:
            n_updates = int(timesteps // n_batch)
            nepisodes = 0
            eprew_max = eprew_min = eprew_avg = eplen_avg = np.nan
            clip_value = clip_vl if callable(clip_vl) else constfn(clip_vl)
            runner = Runner(env=env, model=self, gamma=gamma, lam=lam)
            history = deque(maxlen=100)
            for nupdate in tqdm(range(1, n_updates + 1), desc="Training"):
                tstart = time.time()
                experiences = runner.get_experiences(nsteps)
                obs, acts, returns, values, dones, neglogps, infos = experiences

                stats = self._update(
                    n=nupdate,
                    clip_value=clip_value(1.0 - (nupdate - 1.0) / n_updates),
                    obs=obs,
                    returns=returns,
                    actions=acts,
                    values=values,
                    neglogpacs=neglogps,
                    normalize=True)

                nepisodes += len(infos)
                history.extend(infos)
                if loggers and (nupdate % log_interval == 0 or nupdate == 1):
                    if len(history) > 0:
                        eprew = [h.get('score', 0) for h in history]
                        eplen = [h.get('nsteps', 0) for h in history]
                        eprew_max = np.max(eprew)
                        eprew_min = np.min(eprew)
                        eprew_avg = np.mean(eprew)
                        eplen_avg = np.mean(eplen)

                    elapsed_time = time.time() - tstart
                    ev = round(explained_variance(values, returns), 4)
                    loggers.log('v_explained_variance', ev)
                    loggers.log('nupdates', nupdate)
                    loggers.log('elapsed_time', elapsed_time)
                    loggers.log('fps', int(n_batch / elapsed_time))
                    loggers.log('ntimesteps', nupdate * n_batch)
                    loggers.log('nepisodes', nepisodes)
                    loggers.log('eplen_avg', float(eplen_avg))
                    loggers.log('eprew_avg', float(eprew_avg))
                    loggers.log('eprew_max', float(eprew_max))
                    loggers.log('eprew_min', float(eprew_min))
                    for key, value in stats.items():
                        loggers.log(key, float(value))
                    loggers.dump()
        finally:
            env.close()
            if loggers:
                loggers.close()

        return history

    def play(self, env, render=False):
        assert self._is_compiled
        obs, score, done = env.reset(), 0, False
        while not done:
            if render:
                env.render()
            obs, reward, done, info = env.step(self.act(obs, False))
            score += reward
        return score, info

    def _update(self, n, clip_value, obs, returns, actions, values, neglogpacs, normalize=False):
        advs = returns - values

        if normalize:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Feed iterator
        self.session.run(self.data_iterator.initializer, feed_dict={
            self.obs: obs, self.returns: returns, self.actions: actions, self.advs: advs,
            self.old_neglogprob: neglogpacs, self.old_vpred: values
        })

        try:
            results = []
            feed_dict = {self.clip_value: clip_value, self.global_step: n}
            ops = [
                self.train_op, self.p_loss, self.v_loss, self.entropy, self.approxkl, self.clipfrac,
                self.learning_rate, self.clip_value
            ]

            while True:
                results.append(self.session.run(ops, feed_dict=feed_dict)[1:])
        except tf.errors.OutOfRangeError:
            pass

        stats = np.mean(results, axis=0)
        stats = {
            'p_loss': stats[0],
            'v_loss': stats[1],
            'p_entropy': stats[2],
            'approxkl': stats[3],
            'clip_frac': stats[4],
            'lr': stats[5],
            'clip_value': stats[6]
        }
        return stats
