import os

import pandas as pd
import joblib

from src.utils.commons import *
from src.utils.runners import AbstractEnvRunner
from src.utils.agents import TFAgent
from src.utils.tf_utils import *
from src.utils.env_wrappers import *
from gridgym.envs.grid_env import GridEnv


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
        def sf01(arr):
            """
                    swap and then flatten axes 0 and 1
                    """
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, epinfos = [
        ], [], [], [], [], [], []
        # For n in range number of steps
        for _ in range(self.nsteps):
            actions, values, neglogpacs, _ = self.model.step(self.obs)
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

        return (*map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_dones, mb_neglogpacs)), epinfos)


class PPOAgent(TFAgent):
    def __init__(self, seed=None):
        super().__init__(seed)
        self._compiled = False
        self.summary_episode_interval = 100

    def load_value(self, fn):
        network = tf.trainable_variables("value_network")
        network.extend(tf.trainable_variables("value_logits"))
        values = joblib.load(os.path.expanduser(fn))
        restores = [v.assign(values[v.name]) for v in network]
        self.session.run(restores)

    def compile(
            self, input_shape, nb_actions, p_network, v_network=None, lr=0.01, end_lr=.0001, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            shared=False, decay_steps=50, decay_rate=.1, batch_size=32, epochs=1, summ_dir=None):
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

            dataset = tf.data.Dataset.from_tensor_slices(
                (self.obs, self.actions, self.advs, self.returns, self.old_neglogprob, self.old_vpred))
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
        # u = tf.random_uniform(tf.shape(self.p_logits), dtype=self.p_logits.dtype)
        # self.sample_action = tf.argmax(self.p_logits - tf.log(-tf.log(u)), axis=-1)
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

        # Summary
        #tf.summary.scalar("Learning Rate", self.learning_rate, family='Train Metrics', collections=['train'])
        #tf.summary.scalar("Policy Clipping Fraction", self.clipfrac, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Probs", self.act_probs, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Ratio Unclipped", self.ratio, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Ratio Clipped", self.ratio_clipped, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Loss Unclipped", self.p_loss1, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Loss Clipped", self.p_loss2, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Loss", self.p_loss, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Policy Entropy", prob_dist.entropy(), family='Train Metrics', collections=['train'])
        # tf.summary.scalar("Policy Loss Clipped", tf.reduce_mean(self.p_loss2), family='Train Metrics',
        #                  collections=['train'])
        #tf.summary.scalar("Policy Loss", self.p_loss, family='Train Metrics', collections=['train'])
        #tf.summary.scalar("Policy Entropy", self.entropy, family='Train Metrics', collections=['train'])
#
        # tf.summary.scalar("Value Loss Unclipped", tf.reduce_mean(self.v_loss1), family='Train Metrics',
        #                  collections=['train'])
        #tf.summary.scalar("Value Loss", self.v_loss, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Value Logits", self.v_logits, family='Train Metrics', collections=['train'])
        #tf.summary.histogram("Value Logits Clipped", self.v_clipped, family='Train Metrics', collections=['train'])

        # Episodic
        #self.scores = tf.placeholder(tf.float32, shape=[None])
        #self.steps = tf.placeholder(tf.float32, shape=[None])
        #self.waiting_time = tf.placeholder(tf.float32, shape=[None])
        #self.idle_time = tf.placeholder(tf.float32, shape=[None])
        #self.max_idle_time = tf.placeholder(tf.float32, shape=[None])
        #self.energy = tf.placeholder(tf.float32, shape=[None])
        #self.switches = tf.placeholder(tf.float32, shape=[None])
#
        # variable_summaries(self.scores, name='Score', family='Episode Metrics', collections=['episodic'],
        #                   plot_hist=True)
        #variable_summaries(self.steps, name='Steps', family='Episode Metrics', collections=['episodic'], plot_hist=True)
        # variable_summaries(self.waiting_time, name='Waittime', family='Episode Metrics', collections=['episodic'],
        #                   plot_hist=True)
        # variable_summaries(self.idle_time, name='MeanIdleTime', family='Episode Metrics', collections=['episodic'],
        #                   plot_hist=True)
        # variable_summaries(self.max_idle_time, name='MaxIdleTime', family='Episode Metrics', collections=['episodic'],
        #                   plot_hist=True)
        # variable_summaries(self.energy, name='Energy', family='Episode Metrics', collections=['episodic'],
        #                   plot_hist=True)
        # variable_summaries(self.switches, name='Switches', family='Episode Metrics', collections=['episodic'],
        #                   plot_hist=True)

        #self.summary_train = tf.summary.merge_all('train')
        #self.summary_episodic = tf.summary.merge_all('episodic')
        self.session.run(tf.global_variables_initializer())

        self.summary_writer = None
        # if summ_dir is not None:
        #	os.makedirs(summ_dir, exist_ok=True)
        #	self.summary_writer = tf.summary.FileWriter(summ_dir, self.session.graph)

        self._compiled = True

    def _pred(self, ops, obs):
        assert self._compiled
        return self.session.run(ops, {self.X: obs})

    def step(self, obs):
        return self._pred([self.sample_action, self.v_logits, self.neglogprob, self.act_probs], obs)

    def value(self, obs):
        return self._pred(self.v_logits, obs)

    def act(self, obs, argmax=False):
        op = self.sample_action if not argmax else self.best_action
        return self._pred(op, obs)

    def _update(self, n, clip_value, obs, returns, actions, values, neglogpacs, normalize=False):
        results = []
        advs = returns - values

        advs = (advs - advs.mean()) / \
            (advs.std() + 1e-8) if normalize else advs

        self.session.run(self.data_iterator.initializer, feed_dict={
            self.obs: obs, self.returns: returns, self.actions: actions, self.advs: advs,
            self.old_neglogprob: neglogpacs, self.old_vpred: values
        })

        try:
            ops = [
                self.train_op, self.p_loss, self.v_loss, self.entropy, self.approxkl, self.clipfrac,
                self.learning_rate, self.clip_value
            ]
            if self.summary_writer is not None:
                ops.append(self.summary_train)

            while True:
                output = self.session.run(
                    ops, feed_dict={self.clip_value: clip_value, self.global_step: n})[1:]
                if self.summary_writer is not None:
                    self.summary_writer.add_summary(output[-1], n)
                    output = output[:-1]
                results.append(output)
        except tf.errors.OutOfRangeError:
            pass

        stats = np.mean(results, axis=0)
        stats = {
            'p_loss': stats[0], 'v_loss': stats[1], 'p_entropy': stats[2],
            'approxkl': stats[3], 'clip_frac': stats[4], 'lr': stats[5], 'clip_value': stats[6]
        }
        return stats

    def fit(
            self, env, timesteps, nsteps, nb_batches=8, log_interval=1, loggers=None, gamma=.98, lam=.95,
            clip_value=0.2, checkpoint=None):
        assert self._compiled, "You should compile the model first."
        n_batch = nsteps * env.num_envs
        assert n_batch % nb_batches == 0

        n_updates = int(timesteps // n_batch)
        if checkpoint:
            os.makedirs(checkpoint, exist_ok=True)

        max_score, nepisodes, max_score_avg = np.nan, 0, np.nan
        history = deque(
            maxlen=self.summary_episode_interval) if self.summary_episode_interval > 0 else []
        clip_value = clip_value if callable(
            clip_value) else constfn(clip_value)

        # START UPDATES
        try:
            runner = Runner(env=env,
                            model=self,
                            nsteps=nsteps,
                            gamma=gamma,
                            lam=lam)

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
                    neglogpacs=neglogps,
                    normalize=True)

                elapsed_time = time.time() - tstart
                nepisodes += len(infos)
                history.extend(infos)
                eprew_max = np.max([h['score']
                                    for h in history]) if history else np.nan

                max_score = eprew_max if max_score is np.nan else max(
                    max_score, eprew_max)

                if checkpoint and nupdate % (min(n_updates, n_updates // 10)) == 0:
                    self.save("{}{}".format(checkpoint, nupdate))

                if self.summary_writer is not None and history and (nupdate % log_interval == 0 or nupdate == 1):
                    feed_dict = {
                        self.scores: [h['score'] for h in history],
                        self.steps: [h['nsteps'] for h in history],
                        self.waiting_time: [h['mean_waiting_time'] for h in history],
                        self.idle_time: [h['nb_jobs_queue'] for h in history],
                        self.max_idle_time: [h['nb_jobs_finished'] for h in history],
                        self.energy: [h['consumed_joules'] for h in history],
                        self.switches: [h['nb_switches'] for h in history],
                    }
                    self.summary_writer.add_summary(self.session.run(
                        self.summary_episodic, feed_dict), nupdate)

                if loggers is not None and (nupdate % log_interval == 0 or nupdate == 1):
                    score_avg = safemean([h['score'] for h in history])
                    len_avg = safemean([h['nsteps'] for h in history])
                    max_score_avg = score_avg if max_score_avg is np.nan else max(
                        max_score_avg, score_avg)

                    loggers.log('v_explained_variance', round(
                        explained_variance(values, returns), 4))
                    loggers.log('frac', frac)
                    loggers.log('nupdates', nupdate)
                    loggers.log('elapsed_time', elapsed_time)
                    loggers.log('ntimesteps', nupdate * n_batch)
                    loggers.log('nepisodes', nepisodes)
                    loggers.log('fps', int(n_batch / elapsed_time))
                    loggers.log('eplen_avg', len_avg)
                    loggers.log('eprew_avg_max', float(max_score_avg))
                    loggers.log('eprew_avg', score_avg)
                    loggers.log('eprew_max', float(eprew_max))
                    loggers.log('eprew_max_all', float(max_score))
                    loggers.log('eprew_min', int(
                        np.min([h['score'] for h in history])) if history else np.nan)
                    for key, value in stats.items():
                        loggers.log(key, float(value))
                    loggers.dump()
        finally:
            env.close()
            if loggers is not None:
                loggers.close()

        return history

    def play(self, env, render=False, verbose=False):
        history = defaultdict(list)
        obs, score, done = env.reset(), 0, False
        while not done:
            if render:
                env.render()
            for i, o in enumerate(obs[-1][-348:]):
                history[i].append(o)
            obs, reward, done, info = env.step(self.act(obs, False))
            history['reward'].append(reward[-1])
            #score += reward

        pd.DataFrame(history).to_csv("history.csv", index=False)
        results = pd.read_csv(os.path.join(
            GridEnv.OUTPUT, '_schedule.csv')).to_dict('records')[0]
        results['score'] = info[0]['episode']['score']
        if verbose:
            #info[0]['score'] = score
            m = " - ".join("[{}: {}]".format(k, v) for k, v in results.items())
            print("[RESULTS] {}".format(m))
            print("[INFO] {}".format(info[0]['episode']))
        env.close()
        return results  # info[0]