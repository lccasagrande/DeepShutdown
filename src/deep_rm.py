import utils
import gym
import sys
import csv
import pandas as pd
import numpy as np
import time as t
from keras.layers import Convolution2D, Permute, Dense, Flatten, Dropout, Activation
from keras import Sequential
from keras.optimizers import Adam, RMSprop
from random import choice
from gym_grid.envs.grid_env import GridEnv
from collections import defaultdict, deque


class DeepRMProcessor():
    def __init__(self, nb_resources, job_slots=20, backlog=2, time_window=32):
        self.job_slots = job_slots
        self.time_window = time_window
        self.backlog = backlog
        self.nb_resources = nb_resources

        self.output_shape = (time_window, nb_resources +
                             job_slots + backlog, 1)    

    @property
    def shape(self):
        return self.time_window * self.nb_resources * self.job_slots * self.backlog

    def process_observation(self, observation):
        gantt = observation['gantt']
        assert len(gantt) == self.nb_resources
        jobs_in_queue = observation['job_queue']['jobs']
        nb_jobs_in_queue = observation['job_queue']['nb_jobs_in_queue']
        nb_jobs_waiting = observation['job_queue']['nb_jobs_waiting']

        obs = np.zeros(
            shape=(self.time_window, self.nb_resources + self.job_slots + self.backlog), dtype=np.int)

        for res, resource_space in enumerate(gantt):
            job = resource_space['queue'][0]
            if job != None:
                for tim in range(min(self.time_window, int(round(job.remaining_time)))):
                    obs[tim][res] = 1

        job_slots = min(self.job_slots, nb_jobs_in_queue)
        for job_slot in range(job_slots):
            job = jobs_in_queue[job_slot]
            for tim in range(min(self.time_window, job.requested_time)):
                obs[tim][job_slot+self.nb_resources] = 1

        nb_jobs_waiting += nb_jobs_in_queue  # Verificar

        backlog_slot = self.nb_resources + self.job_slots
        window = 0
        for tim in range(min(self.time_window*self.backlog, nb_jobs_waiting)):
            if tim == self.time_window:
                window = 0
                backlog_slot += 1
            obs[window][backlog_slot] = 1
            window += 1

        return obs.reshape(self.output_shape)


class ReinforceAgent():
    def __init__(self, model, action_size, gamma=.99):
        self.compiled = False
        self.action_size = action_size
        self.gamma = gamma
        self.model = model

    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy')
        self.compiled = True

    def select_action(self, observation, action_filter):
        assert self.compiled
        obs = observation.reshape((1,) + observation.shape)
        policy = self.model.predict(obs, batch_size=1).flatten()
        policy = np.multiply(policy, action_filter)
        probs = policy / np.sum(policy)
        return np.random.choice(self.action_size, 1, p=probs)[0]

    def fit(self, episodes):
        assert self.compiled
        discounted_rewards, all_states, all_actions = [],[],[]
        episodes_len = len(episodes)
        for ep in episodes:
            all_states += [state for state in ep['states']]
            all_actions += [act for act in ep['actions']]
            discounted_rewards.append(self._discount_rewards(ep['rewards']))

        max_ep_len = max(len(rew) for rew in discounted_rewards)
        padded_rets = [np.concatenate([rew, np.zeros(max_ep_len - len(rew))]) for rew in discounted_rewards]

        # Compute time-dependent baseline
        baseline = np.mean(padded_rets, axis=0)

        # Compute advantage function
        advs = [rew - baseline[:len(rew)] for rew in discounted_rewards]
        test = []
        for i in range(episodes_len):
            test += [rew for rew in advs[i]]
        
        #discounted_rewards -= np.mean(discounted_rewards)
        #discounted_rewards /= np.std(discounted_rewards)
        advantages = np.zeros((len(all_states), self.action_size))

        for i in range(len(all_states)):
            advantages[i][all_actions[i]] = test[i]
        
        all_states = np.array(all_states).flatten()
        # TRAIN
        self.model.fit(all_states, advantages, epochs=1, verbose=1)

    def _discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards


def build_model(input_shape, output_shape):
    model = Sequential()
    #model.add(Convolution2D(16, (2, 2), strides=(1, 1),
    #                        data_format="channels_last", input_shape=input_shape))
    #model.add(Activation('relu'))
    #model.add(Flatten())
    model.add(Dense(10, input_shape=input_shape, activation='tanh'))
    #model.add(Dense(20, activation='relu'))
    model.add(Dense(output_shape, activation='softmax'))
    print(model.summary())
    return model


def get_valid_actions(state, action_size):
    return [1] + [int(state['gantt'][i]['resource'].is_available)
                  for i in range(action_size-1)]


def run(output_dir, n_ep=50, ep_length=2000, out_freq=10, plot=False, render=True):
    env = gym.make('grid-v0')
    episodic_scores = deque(maxlen=out_freq)
    avg_scores = deque(maxlen=n_ep)
    eps_traj = []
    action_history = []
    processor = DeepRMProcessor(env.action_space.n-1)
    s = processor.shape
    model = build_model(input_shape=s,
                        output_shape=env.action_space.n)
    agent = ReinforceAgent(model, env.action_space.n)
    agent.compile(Adam(lr=0.001))

    batch_size = 100
    for b in range(batch_size):
        t_start = t.time()
        for i in range(1, n_ep + 1):
            episode_trajectory = dict(states=[], actions=[], rewards=[])
            #utils.print_progress(t_start, i, n_ep)
            epi_actions = np.zeros(env.action_space.n)
            score = 0
            state = env.reset()
            for _ in range(ep_length):
                valid_actions = get_valid_actions(state, env.action_space.n)

                state = processor.process_observation(state)

                action = agent.select_action(state, valid_actions)

                next_state, reward, done, _ = env.step(action)

                episode_trajectory['states'].append(state)
                episode_trajectory['actions'].append(action)
                episode_trajectory['rewards'].append(reward)

                score += reward
                state = next_state
                epi_actions[action] += 1
                if render:
                    env.render()

                if done:
                    break

            eps_traj.append(episode_trajectory)
            if i % out_freq == 0:
                avg_scores.append(np.mean(episodic_scores))

        agent.fit(eps_traj)
        episode_trajectory = []
        episodic_scores.append(score)
        action_history.append(epi_actions)
        print("\nScore: {:7} - Max Score: {:7}".format(score,
                                                       max(episodic_scores)))

    pd.DataFrame(action_history,
                 columns=range(0, env.action_space.n),
                 dtype=np.int).to_csv(output_dir+"actions_hist.csv")

    if plot:
        utils.plot_reward(avg_scores,
                          n_ep,
                          title='Random Policy',
                          output_dir=output_dir)

    # TEST
    print("--- TESTING ---")
    score = 0
    actions = []
    state = env.reset()
    for _ in range(ep_length):
        valid_actions = get_valid_actions(state, env.action_space.n)

        state = processor.process_observation(state)

        action = agent.select_action(state, valid_actions)

        next_state, reward, done, _ = env.step(action)

        score += reward
        state = next_state
        actions.append(action)

        if render:
            env.render()

        if done:
            ac = "-".join(str(act) for act in actions)
            print("\nTest Score: {:7} - Actions: {}".format(score, ac))
            break


if __name__ == "__main__":
    output_dir = 'results/deeprm/'
    utils.clean_or_create_dir(output_dir)

    run(output_dir)
