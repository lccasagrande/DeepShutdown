
import time
import json
import csv
import os.path as osp
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from multiprocessing import Process, Pipe

import numpy as np
import gym
from gym import Wrapper, spaces

from src.utils.commons import tile_images


def _pad_sequences(seqs):
    maxlen = max(len(s) for s in seqs)
    seq, seqs_len = [], []
    for s in seqs:
        seqlen = len(s)
        seqs_len.append(seqlen)
        if seqlen < maxlen:
            seq.append(np.pad(s, ((0, maxlen - seqlen), (0, 0)),
                              mode='constant', constant_values=0))
        else:
            seq.append(s)
    return np.asarray(seq), np.asarray(seqs_len)


def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.action_space, env.observation_space))
            elif cmd == 'render':
                remote.send(env.render(mode='rgb_array'))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        #if self.viewer is None:
        #    from gym.envs.classic_control import rendering
        #    self.viewer = rendering.SimpleImageViewer()
        return self.viewer


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(env_fns), env.observation_space, env.action_space)

    def step(self, actions):
        acts = actions if isinstance(
            actions, (list, np.ndarray)) else [actions]

        results = [self.envs[e].step(acts[e]) for e in range(self.num_envs)]

        obs, rews, dones, infos = map(list, zip(*results))

        obs = [self.envs[e].reset() if done else obs[e]
               for e, done in enumerate(dones)]

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = [self.envs[e].reset() for e in range(self.num_envs)]
        return np.stack(obs)

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)


class SingleVecEnv(DummyVecEnv):
    def __init__(self, env_fn):
        super().__init__([env_fn])

    def step(self, actions):
        acts = actions if isinstance(actions, (list, np.ndarray)) else [actions]
        results = [self.envs[e].step(acts[e]) for e in range(self.num_envs)]
        obs, rews, dones, infos = map(list, zip(*results))
        return np.stack(obs), np.stack(rews), np.stack(dones), infos


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        env_fns: list of environments to run in sub-processes
        """
        self.viewer = None
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(len(env_fns))])
        # self.ps = [Process(target=test_worker, args=(work_remote,), daemon=True) for work_remote in self.work_remotes]
        self.ps = [
            Process(target=worker, args=(work_remote,
                                         CloudpickleWrapper(env_fn)), daemon=True)
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        action_space, observation_space = self.remotes[0].recv()
        super().__init__(len(self.remotes), observation_space, action_space)

    def step(self, actions):
        assert not self.closed
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        assert not self.closed
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        assert not self.closed
        for remote in self.remotes:
            remote.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        low = np.repeat(venv.observation_space.low, self.nstack, axis=-1)
        high = np.repeat(venv.observation_space.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        super().__init__(venv, observation_space=spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype))

    def step(self, action):
        obs, rews, dones, infos = self.venv.step(action)
        for (i, done) in enumerate(dones):
            if done:
                self.stackedobs[i] = 0

        self._stack(obs)
        return self.stackedobs, rews, dones, infos

    def _stack(self, obs):
        self.stackedobs = np.roll(
            self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedobs[..., -obs.shape[-1]:] = obs

    def reset(self):
        self.stackedobs[...] = 0
        self._stack(self.venv.reset())
        return self.stackedobs


class Monitor(Wrapper):
    def __init__(self, env, info_kws=()):
        super().__init__(env)
        self.tstart = time.time()
        self.info_kws = info_kws
        self.actions = None
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.fieldnames = tuple(
            ['score', 'nsteps', 'time'] + [str(i) for i in range(env.action_space.n)]) + tuple(info_kws)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self._update(action, rew, done, info)
        return ob, rew, done, info

    def reset(self, **kwargs):
        self.rewards = []
        self.actions = [0] * self.env.action_space.n
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def _update(self, action, rew, done, info):
        self.rewards.append(rew)
        self.actions[action] += 1
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"score": round(eprew, 6), "nsteps": eplen, "time": round(
                time.time() - self.tstart, 6)}
            for k, v in enumerate(self.actions):
                epinfo[str(k)] = v / eplen
            for k in self.info_kws:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


class FrameStack(Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class CSVMonitor(Monitor):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, info_kws=()):
        super().__init__(env, info_kws)
        if not filename.endswith(CSVMonitor.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, CSVMonitor.EXT)
            else:
                filename = filename + "." + CSVMonitor.EXT
        self.f = open(filename, "wt")
        self.logger = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        self.logger.writeheader()
        self.f.flush()

    def _write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        if 'episode' in info:
            self._write_row(info['episode'])
        return ob, rew, done, info

    def close(self):
        super().close()
        if self.f is not None:
            self.f.close()


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * \
            batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=False, clipob=None, cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.ob_rms = RunningMeanStd(
            shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, actions):
        obs, rews, news, infos = self.venv.step(actions)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var +
                                          self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = (obs - self.ob_rms.mean) / \
                np.sqrt(self.ob_rms.var + self.epsilon)
            if self.clipob is not None:
                obs = np.clip(obs, -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


def make_vec_env(env_id, nenv, seed=None, monitor_dir=None, info_kws=(), wrappers=(), sequential=False):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank if seed is not None else None)
            for wrapper in wrappers:
                env = wrapper(env)
            env = Monitor(env, info_kws) if monitor_dir is None else CSVMonitor(
                env, monitor_dir + str(rank), info_kws)
            return env

        return _thunk

    if nenv == 1:
        return SingleVecEnv(make_env(0))
    elif sequential:
        return DummyVecEnv([make_env(i) for i in range(nenv)])
    else:
        return SubprocVecEnv([make_env(i) for i in range(nenv)])
