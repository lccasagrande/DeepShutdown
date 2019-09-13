import pandas as pd
import numpy as np
import seaborn as sns

#approxkl,clip_frac,clip_value,elapsed_time,eplen_avg,eprew_avg,eprew_avg_max,eprew_max,eprew_max_all,eprew_min,fps,frac,lr,nepisodes,ntimesteps,nupdates,p_entropy,p_loss,v_explained_variance,v_loss

def plot_learning(log_fn):
    log = pd.read_csv(log_fn)
    log['Avg. Reward'] = log['eprew_avg']
    log['Avg. Reward (Max)'] = log['eprew_avg_max']
    log = log[['Avg. Reward', 'Avg. Reward (Max)']].stack().reset_index()
    log['metric'] = log['level_1']
    sns.lineplot(x=log.index, y=log[0], hue=log['metric'])