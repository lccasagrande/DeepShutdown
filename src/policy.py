from random import choice

class RandomPolicy():
    def __init__(self, nb_actions):
        self.nb_actions = nb_actions


    def select_valid_action(self, state):
        gantt = state['gantt']
        valid_actions = [0] + [gantt[i]['resource'].id +
                               1 for i in range(self.nb_actions-1) if gantt[i]['resource'].is_available]
        return choice(valid_actions)

