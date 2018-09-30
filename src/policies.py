from random import choice
from rl.policy import Policy
import numpy as np


class CustomEpsGreedyQPolicy(Policy):
    def __init__(self, nb_res, job_slots, eps=.1):
        super(CustomEpsGreedyQPolicy, self).__init__()
        self.eps = eps
        self.nb_res = nb_res
        self.job_slots = job_slots

    def get_config(self):
        config = super(CustomEpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config

    def select_action(self, q_values, state):
        if np.random.uniform() >= self.eps:
            return self.select_best_action(q_values, state)

        valid_actions = [-1]

        avail_res = 0
        for res in state[-1][0][0:self.nb_res]:
            if res[0] == 0:
                avail_res += 1

        job_slots = state[-1][0][self.nb_res:self.nb_res+self.job_slots]
        for i, job in enumerate(job_slots):
            if job[0] <= 0:
                break

            req_res = int(job[0] * self.nb_res)

            if req_res <= avail_res:
                valid_actions.append(i)

        return choice(valid_actions)

    def select_best_action(self, q_values, state):
        actions = [-1]
        q_max = q_values[0]

        avail_res = 0
        for res in state[-1][0][0:self.nb_res]:
            if res[0] == 0:
                avail_res += 1

        job_slots = state[-1][0][self.nb_res:self.nb_res+self.job_slots]
        for i, job in enumerate(job_slots):
            if job[0] <= 0:
                break
            
            req_res = int(job[0] * self.nb_res)
            if req_res <= avail_res:
                if q_values[i+1] > q_max:
                    q_max = q_values[i+1]
                    actions = [i]
                elif q_values[i+1] == q_max:
                    actions.append(i)

        return choice(actions)


class CustomGreedyQPolicy(CustomEpsGreedyQPolicy):
    def __init__(self, nb_res, job_slots):
        super(CustomGreedyQPolicy, self).__init__(nb_res, job_slots, -1)

    def select_action(self, q_values, state):
        return super(CustomGreedyQPolicy, self).select_best_action(q_values, state)


class RandomPolicy():
    def select_valid_action(self, state):
        gantt = state['gantt']
        jobs = state['job_queue']['jobs']
        available_resources = [
            res_data['resource'].id for res_data in gantt if res_data['resource'].is_available]
        actions = [-1]

        for i, job in enumerate(jobs):
            if len(available_resources) >= job.requested_resources:
                actions.append(i)
                break

        return choice(actions)


class FirstFitPolicy():
    def select_valid_action(self, state):
        gantt = state['gantt']
        jobs = state['job_queue']['jobs']
        available_resources = [
            res_data['resource'].id for res_data in gantt if res_data['resource'].is_available]
        action = -1

        for i, job in enumerate(jobs):
            if len(available_resources) >= job.requested_resources:
                action = i
                break

        return action
