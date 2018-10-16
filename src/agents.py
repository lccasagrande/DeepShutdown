import numpy as np
import csv
import gym
import utils
from gym_grid.envs.grid_env import GridEnv
from multiprocessing import Process, Manager
from collections import defaultdict, deque


class Agent:
    def __init__(self, env):
        self.env = env

    def train(self, n_episodes, metrics, visualize, nb_steps, verbose):
        raise NotImplementedError()

    def save(self, output_fn):
        raise NotImplementedError()

    def load(self, input_fn):
        raise NotImplementedError()


class MCControlAgent(Agent):
    def __init__(self, env, eps_episodes, alpha=None, gamma=1., eps_max=1, eps_min=.1, metrics=[]):
        super(MCControlAgent, self).__init__(env)
        self.policy = utils.LinearAnnealEpsGreedy(
            eps_max, eps_min, eps_episodes)
        self.gamma = gamma
        self.alpha = alpha
        self.metrics = metrics
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def save(self, output_fn):
        np.save(output_fn, dict(self.Q))

    def load(self, input_fn):
        Q = np.load(input_fn+".npy").item()
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), Q)

    def _get_trajectory(self, select_action, epsilon, max_steps=None):
        state = self.env.reset()
        episode, score, steps, result = [], 0, 0, dict()
        while (steps <= max_steps if max_steps != None else True):
            state = tuple(state)
            action = select_action(state, epsilon)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            score += reward
            steps += 1
            if done:
                result['score'] = score
                result['steps'] = steps
                result['Epsilon'] = epsilon
                for metric in self.metrics:
                    result[metric] = info[metric]
                break
        return episode, result

    def _run(self, select_action, n_episodes, update, save_interval=None, visualize=False, verbose=False, max_steps=None):
        def update_q(episode):
            states, actions, rewards = zip(*episode)
            visited = {}
            for i, state in enumerate(states):
                state_action = (state, actions[i])
                if state_action not in visited:
                    discount = np.array(
                        [self.gamma ** i for i in range(len(states[i:]))])
                    counter[state_action] += 1
                    summ = sum(rewards[i:] * discount)

                    if self.alpha is None:
                        error = (
                            summ - self.Q[state][actions[i]]) / counter[state_action]
                    else:
                        error = self.alpha * (summ - self.Q[state][actions[i]])

                    self.Q[state][actions[i]] += error
                    visited[state_action] = True

        counter = defaultdict(int)
        tmp_scores = deque(maxlen=1000)
        avg_scores = deque(maxlen=n_episodes)
        for ep in range(1, n_episodes+1):
            epsilon = self.policy.get_current_value(ep)
            episode, episodic_results = self._get_trajectory(select_action, epsilon, max_steps)
            episodic_results['Episode'] = ep
            tmp_scores.append(episodic_results['score'])

            if update:
                update_q(episode)

            if save_interval != None and ep % save_interval == 0:
                self.save("weights/mcontrol-"+str(ep))

            if (ep % tmp_scores.maxlen == 0):
                if verbose:
                    utils.print_episode_result("MCControl", episodic_results)
                avg_scores.append(np.mean(tmp_scores))

        return avg_scores

    def test(self, n_episodes, visualize=False, max_steps=None, verbose=False):
        def select_action(state, epsilon):
            return np.argmax(self.Q[state])

        return self._run(select_action, n_episodes, update=False, visualize=visualize, verbose=verbose, max_steps=max_steps)

    def train(self, n_episodes, verbose=False, save_interval=None, max_steps=None, visualize=False):
        def select_action(state, epsilon):
            if np.random.random() <= epsilon:
                return np.random.choice(np.arange(self.env.action_space.n))
            else:
                return np.argmax(self.Q[state])

        return self._run(select_action, n_episodes, update=True, visualize=visualize, save_interval=save_interval, verbose=verbose, max_steps=max_steps)


class MCPredictionAgent(Agent):
    def __init__(self, env, policy, metrics=[], gamma=1.0):
        super(MCPredictionAgent, self).__init__(env)
        self.policy = policy
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.metrics = metrics

    def save(self, output_fn):
        np.save(output_fn, dict(self.Q))

    def load(self, input_fn):
        Q = np.load(input_fn+".npy").item()
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), Q)

    def _get_trajectory(self, visualize=False):
        state = self.env.reset()
        episode = []
        score, steps, result = 0, 0, dict()
        while True:
            if visualize:
                self.env.render()
            action = self.policy.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((tuple(state), action, reward))
            state = next_state
            score += reward
            steps += 1
            if done:
                result['score'] = score
                result['steps'] = steps
                for metric in self.metrics:
                    result[metric] = info[metric]
                break
        return episode, result

    def train(self, n_episodes, visualize=False, verbose=False):
        state_action_values = defaultdict(lambda: [0.0, 0])
        for i in range(1, n_episodes+1):
            episode, ep_result = self._get_trajectory(visualize)
            if verbose:
                print("\rEpisode {}/{}".format(i, n_episodes), end="")
                #utils.print_episode_result("MCPred", i, ep_result, 0)

            visited = {}
            states, actions, rewards = zip(*episode)
            for i, state in enumerate(states):
                state_action = (state, actions[i])
                if state_action not in visited:
                    discount = np.array(
                        [self.gamma ** i for i in range(len(states[i:]))])
                    state_action_values[state_action][0] += sum(
                        rewards[i:] * discount)
                    state_action_values[state_action][1] += 1
                    visited[state_action] = True

        for key, value in state_action_values.items():
            state, action = key
            self.Q[state][action] = value[0] / value[1]


class QLearningAgent(Agent):
    def __init__(self, env, alpha, eps_episodes, gamma=1., eps_max=1, eps_min=.1, metrics=[]):
        super(QLearningAgent, self).__init__(env)
        self.policy = utils.LinearAnnealEpsGreedy(
            eps_max, eps_min, eps_episodes)
        self.gamma = gamma
        self.alpha = alpha
        self.metrics = metrics
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def save(self, output_fn):
        np.save(output_fn, dict(self.Q))

    def load(self, input_fn):
        Q = np.load(input_fn+".npy").item()
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), Q)

    def _run(self, select_action, n_episodes, max_steps=None, update=True, visualize=False, verbose=False, save_interval=None):
        def get_epsilon(epi):
            return self.policy.get_current_value(epi) if update else self.policy.value_min

        tmp_scores = deque(maxlen=1000)
        avg_scores = deque(maxlen=n_episodes)
        episodic_results = defaultdict(float)
        total_steps = 0
        for i in range(1, n_episodes + 1):
            score, episode_steps, state = 0, 0, tuple(self.env.reset())
            while (total_steps <= max_steps if max_steps != None else True):
                if visualize:
                    self.env.render()

                epsilon = get_epsilon(i)
                action = select_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                next_state = tuple(next_state)

                if update:
                    td_error = reward + self.gamma * \
                        self.Q[next_state][np.argmax(
                            self.Q[next_state])] - self.Q[state][action]
                    self.Q[state][action] += self.alpha * td_error

                state = next_state
                total_steps += 1
                episode_steps += 1
                score += reward

                if done:
                    for metric in self.metrics:
                        episodic_results[metric] = info[metric]
                    episodic_results['score'] = score
                    episodic_results['steps'] = episode_steps
                    episodic_results['Episode'] = i
                    episodic_results['Epsilon'] = epsilon
                    tmp_scores.append(score)
                    break

            if update and save_interval != None and i % save_interval == 0:
                self.save("weights/qlearning-{}".format(i))

            if (i % 1000 == 0):
                if verbose:
                    utils.print_episode_result("QLearning", episodic_results)
                avg_scores.append(np.mean(tmp_scores))

        return avg_scores

    def test(self, n_episodes, visualize=False, max_steps=None, verbose=False):
        def select_action(state, epsilon):
            return np.argmax(self.Q[state])

        return self._run(select_action, n_episodes, update=False, visualize=visualize, verbose=verbose, max_steps=max_steps)

    def train(self, n_episodes, save_interval=None, visualize=False, max_steps=None, verbose=False):
        def select_action(state, epsilon):
            if np.random.rand() <= epsilon:
                return np.random.choice(np.arange(self.env.action_space.n))
            else:
                return np.argmax(self.Q[state])

        return self._run(select_action, n_episodes, update=True, visualize=visualize, verbose=verbose, max_steps=max_steps, save_interval=save_interval)


class SARSAAgent(Agent):
    def __init__(self, env, alpha, eps_episodes, gamma=1., eps_max=1, eps_min=.1, metrics=[]):
        super(SARSAAgent, self).__init__(env)
        self.policy = utils.LinearAnnealEpsGreedy(
            eps_max, eps_min, eps_episodes)
        self.gamma = gamma
        self.alpha = alpha
        self.metrics = metrics
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def save(self, output_fn):
        np.save(output_fn, dict(self.Q))

    def load(self, input_fn):
        Q = np.load(input_fn+".npy").item()
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n), Q)

    def _run(self, select_action, n_episodes, max_steps=None, update=True, visualize=False, verbose=False, save_interval=None):
        def get_epsilon(epi):
            return self.policy.get_current_value(epi) if update else self.policy.value_min

        tmp_scores = deque(maxlen=1000)
        avg_scores = deque(maxlen=n_episodes)

        episodic_results = defaultdict(float)
        self.step = 0
        for i in range(1, n_episodes + 1):
            score, steps = 0, 0
            state = tuple(self.env.reset())
            epsilon = get_epsilon(i)
            action = select_action(state, epsilon)

            while (self.step <= max_steps if max_steps != None else True):
                if visualize:
                    self.env.render()

                next_state, reward, done, info = self.env.step(action)
                next_state = tuple(next_state)
                self.step += 1
                steps += 1
                score += reward

                if not done:
                    next_action = select_action(next_state, epsilon)
                    if update:
                        td_error = reward + self.gamma * \
                            self.Q[next_state][next_action] - \
                            self.Q[state][action]
                        self.Q[state][action] += self.alpha * td_error
                    state, action = next_state, next_action
                else:
                    if update:
                        self.Q[state][action] += self.alpha * \
                            (reward - self.Q[state][action])

                    for metric in self.metrics:
                        episodic_results[metric] = info[metric]

                    episodic_results['score'] = score
                    episodic_results['steps'] = steps
                    episodic_results['Episode'] = i
                    episodic_results['Epsilon'] = epsilon

                    tmp_scores.append(score)
                    break

            if update and save_interval != None and i % save_interval == 0:
                self.save("weights/sarsa-{}".format(i))

            if (i % 1000 == 0):
                if verbose:
                    utils.print_episode_result("SARSA", episodic_results)
                avg_scores.append(np.mean(tmp_scores))

        return avg_scores

    def test(self, n_episodes, visualize=False, max_steps=None, verbose=False):
        def select_action(state, epsilon):
            return np.argmax(self.Q[state])

        return self._run(select_action, n_episodes, visualize=visualize, max_steps=max_steps, verbose=verbose, update=False)

    def train(self, n_episodes, visualize=False, max_steps=None, verbose=False, save_interval=None):
        def select_action(state, epsilon):
            if np.random.random() <= epsilon:
                return np.random.choice(np.arange(self.env.action_space.n))
            else:
                return np.argmax(self.Q[state])

        return self._run(select_action, n_episodes, visualize=visualize, max_steps=max_steps, verbose=verbose, save_interval=save_interval, update=True)


def get_worker(agent, episodes, max_steps, save_interval, verbose, results):
    agent_name = agent.__class__.__name__
    weights_fn = "weights/"+agent_name

    avg, _ = agent.train(episodes, verbose=verbose,
                         save_interval=save_interval, max_steps=max_steps)
    agent.save(weights_fn)

    with open(agent_name+'.avg', 'w') as f:
        for item in avg:
            f.write("{}\n".format(item))

    #results[agent_name] = result


def get_agents(alpha, eps_episodes):
    ql = QLearningAgent(env=gym.make('grid-v0'),
                        alpha=alpha,
                        eps_episodes=eps_episodes)
    ql.load("weights/qlearning-13500")
    s = SARSAAgent(env=gym.make('grid-v0'),
                   alpha=alpha,
                   eps_episodes=eps_episodes)
    s.load("weights/sarsa-18900")
    m = MCControlAgent(env=gym.make('grid-v0'),
                       alpha=alpha,
                       eps_episodes=eps_episodes)
    m.load("weights/mcontrol-27900")

    return [ql, s, m]


def run():
    episodes = 50000
    save_interval = 900
    verbose = False
    alpha = 0.9
    eps_episodes = 10000

    agent = QLearningAgent(env=gym.make('grid-v0'),
                           alpha=alpha,
                           eps_episodes=eps_episodes)
    import time as t
    t1 = t.time()
    avg = agent.train(episodes, verbose=True)
    print(t.time() - t1)

    #utils.plot_reward(avg, episodes, "Training", "train")
#
    #agent.save("weights/qlearning")
#
    #avg = agent.test(1000, verbose=True)

    #utils.plot_reward(avg, 500, "Testing", "test")

    #agents = get_agents(alpha, eps_steps)
#
    #manager = Manager()
    #manager_result = manager.dict()
    #process = []
##
    # for agent in agents:
    #    p = Process(target=get_worker, args=(agent, episodes, None,
    #                                         save_interval, verbose, manager_result, ))
    #    p.start()
    #    process.append(p)
##
    # for p in process:
    #    p.join()
#
    # print("Done!")


if __name__ == "__main__":
    run()
