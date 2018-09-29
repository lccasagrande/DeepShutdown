from __future__ import division
from PIL import Image
import argparse
import numpy as np
import random
import gym
import time
import gym_grid.envs.grid_env as g
import keras.backend as K
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Convolution2D, BatchNormalization, Permute, GRU, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from dqn_utils import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, Policy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class CustomEpsGreedyQPolicy(Policy):
    def __init__(self, nb_res, eps=.1):
        super(CustomEpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.nb_res = nb_res

    def get_config(self):
        config = super(CustomEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

    def select_action(self, q_values, state):
        if np.random.uniform() >= self.eps:
            return self.select_best_action(q_values, state)

        resources = state[-1][1][0:self.nb_res]
        valid_actions = [0] + [i+1 for i in range(q_values.shape[0]-1) if resources[i][0] == 0.]
        return random.choice(valid_actions)

    def select_best_action(self, q_values, state):
        actions = [0]
        q_max = q_values[0]

        resources = state[-1][1][0:self.nb_res]

        for i in range(q_values.shape[0]-1):
            if resources[i][0] == 0.:
                act = i+1
                if q_values[act] > q_max:
                    q_max = q_values[act]
                    actions = [act]
                elif q_values[act] == q_max:
                    actions.append(act)

        return random.choice(actions)


class CustomGreedyQPolicy(CustomEpsGreedyQPolicy):
    def __init__(self, nb_res):
        super(CustomGreedyQPolicy, self).__init__(nb_res, -1)

    def select_action(self, q_values, state):
        return super(CustomGreedyQPolicy, self).select_best_action(q_values, state)


class GridProcessor(Processor):
    def __init__(self, job_slots, backlog, time_window, nb_res, max_slowdown, max_efficiency):
        super().__init__()
        self.max_slowdown = max_slowdown
        self.max_efficiency = max_efficiency
        self.time_window = time_window
        self.job_slots = job_slots
        self.backlog = backlog
        self.nb_res = nb_res
        self.output_shape = (self.time_window, nb_res +
                             self.job_slots + self.backlog, 1)

    def process_state_batch(self, batch):
        return np.asarray([v[0] for v in batch])

    def process_observation(self, observation):
        nb_resources = len(observation['gantt'])
        assert nb_resources == self.nb_res
        gantt = observation['gantt']
        jobs_in_queue = observation['job_queue']['jobs']
        nb_jobs_in_queue = observation['job_queue']['nb_jobs_in_queue']
        nb_jobs_waiting = observation['job_queue']['nb_jobs_waiting']

        obs = np.zeros(
            shape=(self.time_window, nb_resources + self.job_slots + self.backlog), dtype=np.float)

        for res, resource_space in enumerate(gantt):
            job = resource_space['queue'][0]
            obs[0][res] = resource_space['resource'].cost_to_compute / \
                self.max_efficiency
            if job != None:
                for tim in range(1, min(self.time_window, int(round(job.remaining_time)))):
                    obs[tim][res] = 1

        job_slots = min(self.job_slots, nb_jobs_in_queue)
        for job_slot in range(job_slots):
            job = jobs_in_queue[job_slot]
            obs[0][job_slot+nb_resources] = min(
                self.max_slowdown, job.waiting_time / job.requested_time)
            for tim in range(1, min(self.time_window, job.requested_time)):
                obs[tim][job_slot+nb_resources] = 1

        nb_jobs_in_queue -= len(jobs_in_queue) + nb_jobs_waiting
        backlog_slot = nb_resources + self.job_slots
        window = 0
        for tim in range(min(self.time_window*self.backlog, nb_jobs_in_queue)):
            if tim == self.time_window:
                window = 0
                backlog_slot += 1
            obs[window][backlog_slot] = 1
            window += 1

        obs = obs.reshape(obs.shape + (1,))
        assert obs.ndim == 3  # (height, width, channel)
        return obs


def build_model(output_shape, input_shape):
    model = Sequential()
    model.add(Permute((1, 2, 3), input_shape=input_shape))
    model.add(Convolution2D(16, (2, 2), strides=(1, 1), data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    print(model.summary())
    return model


if __name__ == "__main__":
    K.set_image_dim_ordering('tf')
    env = gym.make('grid-v0')
    name = "dqn_keras_6"
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    processor = GridProcessor(job_slots=117,
                              backlog=1,
                              time_window=128,
                              nb_res=nb_actions-1,
                              max_slowdown=env.max_slowdown,
                              max_efficiency=env.max_energy_consumption)


    model = build_model(nb_actions, processor.output_shape)

    memory = SequentialMemory(limit=50000, window_length=10)

    # Select a policy. We use eps-greedy action selection, which means that a random action is selected
    # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
    # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
    # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
    # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
    policy = LinearAnnealedPolicy(CustomEpsGreedyQPolicy(nb_actions-1), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=200000)

    test_policy = CustomGreedyQPolicy(nb_actions-1)

    # The trade-off between exploration and exploitation is difficult and an on-going research topic.
    # If you want, you can experiment with the parameters or use a different policy. Another popular one
    # is Boltzmann-style exploration:
    # policy = BoltzmannQPolicy(tau=1.)
    # Feel free to give it a try!

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, test_policy=test_policy, memory=memory,
                   processor=processor, nb_steps_warmup=20000, gamma=.99, target_model_update=10000,
                   train_interval=10, delta_clip=1.)
    dqn.compile(Adam(lr=.000001), metrics=['mae','mse'])

    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that you can the built-in Keras callbacks!
    callbacks = [ModelIntervalCheckpoint(
        'weights/'+name+'_1_weights_{step}.h5f', interval=100000)]
    callbacks += [FileLogger('log/'+name+'/'+name+'_1_log.json', interval=1)]
    callbacks += [TensorBoard(log_dir='log/'+name)]
    dqn.fit(env, callbacks=callbacks, nb_steps=500000,
            log_interval=10000, visualize=False, verbose=2)

    # After training is done, we save the final weights one more time.
    dqn.save_weights('weights/'+name+'_1_weights.h5f', overwrite=True)
    #time.sleep(10)
    #dqn.load_weights('weights/dqn_keras_4_1_weights_300000.h5f')
    #dqn.test(env, nb_episodes=1, visualize=True)
