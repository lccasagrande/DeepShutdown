import numpy as np
from src.utils.agent import Agent
from src.utils.common import get_jobs_from_img, get_avail_res_from_img


class UserAgent(Agent):
	def act(self, state):
		return int(input("Action: "))


class RandomAgent(Agent):
	def act(self, state):
		_, slots = get_jobs_from_img(state)
		return np.random.randint(slots + 1)


class SJFAgent(Agent):
	def act(self, state):
		action, shortest_job = 0, np.inf
		nb_res = get_avail_res_from_img(state)
		jobs, _ = get_jobs_from_img(state)
		for j in jobs:
			if j[0] <= nb_res and j[1] < shortest_job:
				shortest_job = j[1]
				action = j[2]

		return action


class LJFAgent(Agent):
	def act(self, state):
		action, largest_job = 0, -1

		nb_res = get_avail_res_from_img(state)
		jobs, _ = get_jobs_from_img(state)
		for j in jobs:
			if j[0] <= nb_res and j[1] > largest_job:
				largest_job = j[1]
				action = j[2]

		return action


class PackerAgent(Agent):
	def act(self, state):
		action, score, = 0, 0

		nb_res = get_avail_res_from_img(state)
		jobs, _ = get_jobs_from_img(state)
		for j in jobs:
			if j[0] <= nb_res and j[0] > score:
				score = j[0]
				action = j[2]

		return action


class TetrisAgent(Agent):
	def __init__(self, knob=0.5, **kwargs):
		super(TetrisAgent, self).__init__(**kwargs)
		self.knob = knob

	def act(self, state):
		action, score, = 0, 0

		nb_res = get_avail_res_from_img(state)
		jobs, _ = get_jobs_from_img(state)
		for j in jobs:
			if j[0] <= nb_res:
				sjf_score = 1 / float(j[1])
				align_score = j[0]

				combined_score = (self.knob * align_score) + ((1 - self.knob) * sjf_score)
				if combined_score > score:
					score = combined_score
					action = j[2]

		return action


class FirstFitAgent(Agent):
	def act(self, state):
		action = 0

		nb_res = get_avail_res_from_img(state)
		jobs, _ = get_jobs_from_img(state)
		for j in jobs:
			if j[0] <= nb_res:
				return j[2]

		return action
