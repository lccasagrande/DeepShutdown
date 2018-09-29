import pandas as pd
import json
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot


with open('log/dqn_keras_5_1_log.json', 'r') as f:
    data = json.load(f)
    loss = data['loss']
    eps = data['episode']
    eps_reward = data['episode_reward']
    mae = data['mean_absolute_error']
    mse = data['mean_squared_error']
    mean_q = data['mean_q']
    mean_eps = data['mean_eps']
    nb_episode_steps = data['nb_episode_steps']
    data = [
        go.Scatter(x=eps, y=loss)
    ]
    plot(data)