import numpy as np
import csv
from gym_grid.envs.grid_env import GridEnv
import gym
from policies import SJF
import utils
from collections import defaultdict, deque


class GLIE():
    def select_action(self, state):
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        return action


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
    def __init__(self, env, eps_steps, alpha=None, gamma=1., eps_max=1, eps_min=.1, metrics=[]):
        super(MCControlAgent, self).__init__(env)
        self.policy = utils.LinearAnnealEpsGreedy(eps_max, eps_min, eps_steps)
        self.gamma = gamma
        self.alpha = alpha
        self.metrics = metrics
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def save(self, output_fn):
        utils.export_q_values(self.Q, output_fn)

    def load(self, input_fn):
        self.Q = defaultdict(lambda: np.zeros(
            self.env.action_space.n), utils.import_q_values(input_fn))

    def _get_trajectory(self, epsilon):
        def select_action(state, epsilon):
            if np.random.random() <= epsilon:
                return np.random.choice(np.arange(self.env.action_space.n))
            else:
                return np.argmax(self.Q[state])

        state = self.env.reset()
        episode, score, steps, result = [], 0, 0, dict()
        while True:
            action = select_action(state, epsilon)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
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

    def train(self, n_episodes, verbose=False):
        counter = defaultdict(int)
        for i in range(n_episodes):
            epsilon = self.policy.get_current_value(i)
            episode, result = self._get_trajectory(epsilon)
            if verbose:
                print("\rEpisode {}/{} - Eps {}".format(i +
                                                        1, n_episodes, epsilon), end="")
                #utils.print_episode_result("MCControl", i+1, result, epsilon)

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
                        self.Q[state][actions[i]
                                      ] += (summ - self.Q[state][actions[i]]) / counter[state_action]
                    else:
                        self.Q[state][actions[i]] += self.alpha * \
                            (summ - self.Q[state][actions[i]])

                    visited[state_action] = True

        policy = dict((k, np.argmax(v)) for k, v in self.Q.items())
        return policy


class MCPredictionAgent(Agent):
    def __init__(self, env, policy, metrics=[], gamma=1.0):
        super(MCPredictionAgent, self).__init__(env)
        self.policy = policy
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.metrics = metrics

    def save(self, output_fn):
        utils.export_q_values(self.Q, output_fn)

    def load(self, input_fn):
        self.Q = defaultdict(lambda: np.zeros(
            self.env.action_space.n), utils.import_q_values(input_fn))

    def _get_trajectory(self, visualize=False):
        state = self.env.reset()
        episode = []
        score, steps, result = 0, 0, dict()
        while True:
            if visualize:
                self.env.render()
            action = self.policy.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            episode.append((state, action, reward))
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
    def __init__(self, env, alpha, eps_steps, gamma=1., eps_max=1, eps_min=.1, metrics=[]):
        super(QLearningAgent, self).__init__(env)
        self.policy = utils.LinearAnnealEpsGreedy(
            eps_max, eps_min, eps_steps)
        self.gamma = gamma
        self.alpha = alpha
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.step = 0
        self.eps_steps = eps_steps
        self.metrics = metrics
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def save(self, output_fn):
        utils.export_q_values(self.Q, output_fn)

    def load(self, input_fn):
        self.Q = defaultdict(lambda: np.zeros(
            self.env.action_space.n), utils.import_q_values(input_fn))

    def _run(self, select_action, n_episodes, max_steps=None, update=True, visualize=False, verbose=False, save_interval=None):
        tmp_scores = deque(maxlen=100)
        avg_scores = deque(maxlen=n_episodes)
        self.step = 0
        episodic_results = defaultdict(float)
        results = []
        for i in range(1, n_episodes + 1):
            score, episode_steps, state = 0, 0, self.env.reset()
            while (self.step <= max_steps if max_steps != None else True):
                if visualize:
                    self.env.render()

                epsilon = self.policy.get_current_value(self.step) if update else self.policy.value_min
                action = select_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                if update:
                    td_error = reward + self.gamma * \
                        self.Q[next_state][np.argmax(
                            self.Q[next_state])] - self.Q[state][action]
                    self.Q[state][action] += self.alpha * td_error

                self.step += 1
                episode_steps += 1
                score += reward
                state = next_state

                if done:
                    for metric in self.metrics:
                        episodic_results[metric] = info[metric]
                    episodic_results['score'] = score
                    episodic_results['steps'] = episode_steps
                    tmp_scores.append(score)
                    results.append(episodic_results)
                    if verbose:
                        utils.print_episode_result(
                            "QLearning", i, episodic_results, epsilon)
                    break

            if update and save_interval != None and i % save_interval == 0:
                self.save("qlearning-{}.csv".format(i))

            if (i % 100 == 0):
                avg_scores.append(np.mean(tmp_scores))

        return avg_scores, results

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
    def __init__(self, env, alpha, eps_steps, gamma=1., eps_max=1, eps_min=.1, metrics=[]):
        super(SARSAAgent, self).__init__(env)
        self.policy = utils.LinearAnnealEpsGreedy(
            eps_max, eps_min, eps_steps)
        self.gamma = gamma
        self.alpha = alpha
        self.step = 0
        self.metrics = metrics
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def save(self, output_fn):
        utils.export_q_values(self.Q, output_fn)

    def load(self, input_fn):
        self.Q = defaultdict(lambda: np.zeros(
            self.env.action_space.n), utils.import_q_values(input_fn))

    def _run(self, select_action, n_episodes, max_steps=None, update=True, visualize=False, verbose=False, save_interval=None):
        def get_epsilon():
            return self.policy.get_current_value(self.step) if update else self.policy.value_min

        tmp_scores = deque(maxlen=100)
        avg_scores = deque(maxlen=n_episodes)

        self.step = 0
        episodic_results = defaultdict(float)
        results = []
        for i in range(1, n_episodes + 1):
            score = 0
            steps = 0
            state = self.env.reset()
            epsilon = get_epsilon()
            action = select_action(state, epsilon)

            while (self.step <= max_steps if max_steps != None else True):
                if visualize:
                    self.env.render()

                next_state, reward, done, info = self.env.step(action)
                self.step += 1
                steps += 1
                score += reward

                if not done:
                    epsilon = get_epsilon()
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

                    results.append(episodic_results)
                    tmp_scores.append(score)
                    if verbose:
                        utils.print_episode_result(
                            "SARSA", i, episodic_results, epsilon)
                    break

            if update and save_interval != None and i % save_interval == 0:
                self.save("sarsa-{}.csv".format(i))

            if (i % 100 == 0):
                avg_scores.append(np.mean(tmp_scores))

        return avg_scores, results

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


def run():
    env = gym.make('grid-v0')

    agent = QLearningAgent(env, .01, 1)

    avg, _ = agent.train(5000, verbose=True)


if __name__ == "__main__":
    run()
