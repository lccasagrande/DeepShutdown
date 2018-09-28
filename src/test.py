#%%

import pandas as pd
import json
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot


with open('log/dqn_keras_3/dqn_keras_3_1_log.json', 'r') as f:
    data = json.load(f)
    loss = data['loss']
    eps = data['episode']
    eps_reward = data['episode_reward']
    data = [
        go.Scatter(x=data['episode'], y=data['mean_absolute_error'])
    ]
    plot(data)