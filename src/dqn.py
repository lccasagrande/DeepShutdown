import numpy as np
import gym
import utils
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense,  Flatten
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


WINDOW_LENGTH = 1

class GridProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 2  # (height, width)
        return observation

    def process_state_batch(self, batch):
        return batch


def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + input_shape))
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_shape, activation='linear'))
    print(model.summary())
    return model


if __name__ == "__main__":
    train = False
    weights_nb = "0"
    name = "dqn"
    seed = 123
    weight_path = "weights/" + name
    log_path = "logs/" + name
    utils.create_dir(weight_path)
    utils.create_dir(log_path)

    env = gym.make('grid-v0')
    np.random.seed(seed)
    env.seed(seed)

    model = build_model(input_shape=env.observation_space.shape, output_shape=env.action_space.n)

    memory = SequentialMemory(limit=50000, window_length=WINDOW_LENGTH)

    train_policy = LinearAnnealedPolicy(inner_policy=EpsGreedyQPolicy(),
                                        attr='eps',
                                        value_max=1.,
                                        value_min=.1,
                                        value_test=.05,
                                        nb_steps=50000)

    dqn = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   policy=train_policy,
                   processor=GridProcessor(),
                   memory=memory,
                   train_interval=1,
                   nb_steps_warmup=100,
                   target_model_update=1000,
                   gamma=1.)

    dqn.compile(Adam(lr=1e-3), metrics=['mae', 'mse'])

    callbacks = [
        ModelIntervalCheckpoint(weight_path + '/weights_{step}.h5f', interval=1e5),
        FileLogger(log_path+'/log.json', interval=100),
        TensorBoard(log_dir=log_path,write_graph=True, write_images=True)
    ]
    if train:
        dqn.fit(env=env,
                callbacks=callbacks,
                nb_steps=1e5,
                log_interval=10000,
                visualize=False,
                verbose=2)

        dqn.save_weights(weight_path+'/weights_0.h5f', overwrite=True)
    else:
        dqn.load_weights(weight_path+'/weights_'+weights_nb+'.h5f')
        dqn.test(env, nb_episodes=1, visualize=False)
