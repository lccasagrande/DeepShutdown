import gym
from copy import deepcopy
from collections import defaultdict
from gym import error, spaces, utils
from gym.utils import seeding
from sklearn.preprocessing import MinMaxScaler
from .batsim.batsim import BatsimHandler, InsufficientResourcesError, UnavailableResourcesError
import numpy as np


class GridEnv(gym.Env):
    def __init__(self):
        self.simulator = BatsimHandler(output_freq=1)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.seed()

    @property
    def max_time(self):
        return self.simulator.max_walltime

    def step(self, action):
        assert self.simulator.running_simulation, "Simulation is not running."

        alloc_resources = self._prepare_input(action)

        # get reward metrics
        energy_consumed_est = self.simulator.resource_manager.estimate_energy_consumption(alloc_resources)
        #energy_consumed_est = float(energy_consumed_est) / self.simulator.resource_manager.max_cost_to_compute

        wait_time = self.simulator.sched_manager.get_first_job_wait_time()
       # expected_runtime = self.simulator.sched_manager.get_first_job_walltime()
        #expected_turnaround = max((wait_time + expected_runtime) / expected_runtime, 1)

        # schedule first job
        self.simulator.schedule_job(alloc_resources)

        state = self._update_state()
        
        #nb_res = self.simulator.resource_manager.nb_resources
        jobs_waiting = self.simulator.sched_manager.nb_jobs_waiting
        #load = min(nb_res, jobs_waiting) / nb_res

        done = not self.simulator.running_simulation

        #energy_consumed_est = -1*energy_consumed_est
        #load = -1*load
        #bdslowdown = 1 - min(expected_turnaround, 2)

        reward = -1 * (energy_consumed_est + wait_time + jobs_waiting + 1)
        #reward = (energy_consumed_est + bdslowdown + load) / 3

        return state, reward, done, {}

    def reset(self):
        self.simulator.close()
        self.simulator.start()
        state = self._update_state()
        return state

    def render(self, mode='human'):
        stats = "\rSubmitted: {:5} Completed: {:5} | Running: {:5} In Queue: {:5}".format(
            self.simulator.nb_jobs_submitted,
            self.simulator.nb_jobs_completed,
            self.simulator.nb_jobs_running,
            self.simulator.nb_jobs_in_queue)

        print(stats, end="", flush=True)

    def close(self):
        self.simulator.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _prepare_input(self, action):
        return [] if action == 0 else [action-1]

    def _update_state(self):
        shape = self.simulator.current_state.shape
        state = np.zeros(shape=shape, dtype=np.float)
        simulator_state = self.simulator.current_state
        for row in range(shape[0]):
            if simulator_state[row][0] != None:
                # Get host property
                state[row][0] = simulator_state[row][0].cost_to_compute
                for col in range(1, shape[1]):
                    # Get jobs remaining time
                    job = simulator_state[row][col]
                    state[row][col] = job.remaining_time if job is not None else 0
            else:
                # Add first job in queue and number of waiting jobs
                job = simulator_state[row][1]
                state[row][0] = job.waiting_time if job is not None else 0
                state[row][1] = self.simulator.sched_manager.nb_jobs_waiting

        return state

    def _get_action_space(self):
        return spaces.Discrete(self.simulator.nb_resources+1)

    def _get_observation_space(self):
        obs_space = spaces.Box(low=0,
                               high=self.max_time,
                               shape=self.simulator.state_shape,
                               dtype=np.int16)

        return obs_space
