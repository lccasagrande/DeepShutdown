import numpy as np
import pandas as pd
import random
import plotly.graph_objs as go
from matplotlib.colors import XKCD_COLORS as allcolors
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import math
import plotly.tools as tools
import sys
import time as t
import csv
import os, shutil
#sns.set_style("white")

#def plot_action_values(V):
#	# reshape the state-value function
#	# plot the state-value function
#	fig = plt.figure(figsize=(15,5))
#	ax = fig.add_subplot(111)
#	_ = ax.imshow(V, cmap='cool')
#	for (j,i),label in np.ndenumerate(V):
#	    ax.text(i, j, np.round(label,3), ha='center', va='center', fontsize=14)
#	plt.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')
#	plt.title('State-Value Function')
#	plt.show()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def export_rewards(n_ep, avg_scores, output_dir):
    x, y = np.linspace(0, n_ep, len(avg_scores),
                       endpoint=False), np.asarray(avg_scores)
    dt = pd.DataFrame({'ep': list(x), 'avg_reward': list(y)})
    dt.to_csv(output_dir+'rewards.csv', index=False)


def export_q_values(Q, output_dir):
    q_values = pd.DataFrame.from_dict(Q, orient='index')
    q_values.index.name = "state"
    q_values.to_csv(output_dir+"q_values.csv")


def export_max_q_values(Q, output_dir):
    hist = {}
    for k, value in Q.items():
        actions = [i for i, act in enumerate(value) if act == np.amax(value)]
        actions = "-".join(str(q) for q in actions)
        hist[k] = actions

    q_max_values = pd.DataFrame.from_dict(hist, orient='index')
    q_max_values.index.name = "state"
    q_max_values.to_csv(output_dir+"max_q_values.csv")


def dict_to_csv(data, fn):
    dt = pd.DataFrame(data)
    dt.to_csv(fn, index=False)


def plot_reward(avg_scores, n_ep, title, output_dir):
    x, y = np.linspace(0, n_ep, len(avg_scores),
                       endpoint=False), np.asarray(avg_scores)

    dict_to_csv({'ep': list(x), 'avg_reward': list(y)}, output_dir+'rewards.csv')

    trace = go.Scatter(x=x, y=y)
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Episode'},
        yaxis={'title': 'Avg Reward'},
    )
    fig = go.Figure(data=[trace], layout=layout)
    plot(fig, filename=output_dir+'rewards.html')


def print_progress(t_start, ep, num_episodes):
    total_time = t.time() - t_start
    avg_ep_time = round(total_time / ep, 2)
    rem_time = round(avg_ep_time * (num_episodes - ep), 2)
    print("\nEpisode {}/{}: (Avg:{} s - Remaining: {} s)".format(ep,
                                                                 num_episodes, avg_ep_time, rem_time), flush=True)
    sys.stdout.flush()


def merge_results(path, scheds, output_fn=None):
    results = pd.DataFrame()
    for policy in scheds:
        df = pd.read_csv(
            "{}/{}_schedule.csv".format(path, policy))
        df['policy'] = policy
        results = results.append(df, ignore_index=True)

    if output_fn is not None:
        results.to_csv(output_fn)
    return results


def plot_results(data, metrics, output_fn, main_title, max_cols=3, y_labels=None, colors=None):
    def get_cols_and_rows():
        tot_metrics = len(metrics)
        rows = math.ceil(tot_metrics / max_cols)
        cols = tot_metrics if tot_metrics < max_cols else max_cols
        return rows, cols

    def update_rows_and_cols(row, col):
        if col == max_cols:
            row += 1
            col = 1
        else:
            col += 1
        return row, col

    def calc_diffs(dt):
        min_value = dt.min()
        perc = round(((dt * 100) / min_value) - 100, 2)
        return [str(p) + "%" for p in perc]

    random_colors = colors if colors is not None else random.sample(
        list(allcolors.values()), k=len(data['policy']))
    rows, cols = get_cols_and_rows()
    fig = tools.make_subplots(rows=rows, cols=cols, subplot_titles=metrics)

    row, col = 1, 1
    for metric in metrics:
        graph = go.Bar(
            showlegend=False,
            name=metric,
            text=calc_diffs(data[metric]),
            textposition='auto',
            textfont=dict(size=20, color='#ffffff'),
            x=data['policy'],
            y=data[metric],
            marker={'color': random_colors})

        fig.append_trace(graph, row, col)
        row, col = update_rows_and_cols(row, col)

    if y_labels is not None:
        assert len(y_labels) == len(metrics)
        for i, title in enumerate(y_labels):
            axis_ind = "yaxis" + str(i+1)
            fig['layout'][axis_ind].update(title=title)
    fig['layout'].update(title=main_title)
    plot(fig, filename=output_fn)


def plot_job_stats(data, title, total, metrics, output_fn, colors=None):
    def calc_diffs(dt):
        perc = round(((dt * 100) / total), 2)
        return [str(p) + "%" for p in perc]

    dt = []
    for i, metric in enumerate(metrics):
        color = None if colors is None else colors[i]
        dt.append(
            go.Bar(
                x=data['policy'],
                y=data[metric],
                name=metric,
                text=calc_diffs(data[metric]),
                textfont=dict(size=20, color='#ffffff'),
                marker={'color': color}
            )
        )
    layout = go.Layout(barmode='stack', title=title)
    fig = go.Figure(data=dt, layout=layout)
    plot(fig, filename=output_fn)