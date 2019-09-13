import gym
import gridgym.envs.grid_env as g
from src.utils.networks import *
import tensorflow as tf
from gridgym.envs.off_reservation_env import OffReservationGridEnv
from gridgym.envs.simulator.simulator import SimulatorHandler
from shutil import copyfile, rmtree
from collections import defaultdict
import gym

SIMULATOR_PLATFORM = "files/platforms/lyon2__168.xml"
SIMULATOR_WORKLOAD = "files/workloads/aval"
WORKLOAD_PATH = "../src/GridGym/gridgym/envs/simulator/" + SIMULATOR_WORKLOAD
BATSIM_OUTPUT = "../src/GridGym/gridgym/envs/simulator/files/output"


def plot(fn):
	def plot_reward(data):
		# data['date_ordinal'] = pd.to_datetime(data['date']).apply(lambda date: date.toordinal())
		# sns.regplot(x='date', y='reward', fit_reg=False, data=data[data.policy == 'PPO_2'],  label='DeepShutdown')
		# sns.regplot(x='date', y='reward', fit_reg=False, data=data[data.policy == 'idle_1'],  label='Timeout=1')
		# sns.regplot(x='date', y='reward', fit_reg=False, data=data[data.policy == 'idle_5'],  label='Timeout=5')
		# sns.regplot(x='date', y='reward', fit_reg=False, data=data[data.policy == 'idle_15'],  label='Timeout=15')
		# ax = sns.regplot(x='date', y='reward', fit_reg=False, data=data[data.policy == 'idle_10'],  label='Timeout=10')
		fig, axs = plt.subplots(nrows=3, sharex=True, sharey=False, figsize=(16, 16))
		sns.barplot(x='date', y='nb_switches', hue='policy', data=data, palette='pastel', ax=axs[0])
		axs[0].set(ylabel='Qtd. de Reinicializações', xlabel='')

		sns.barplot(x='date', y='time_idle', hue='policy', data=data, palette='pastel', ax=axs[1])
		axs[1].set(ylabel='Tempo ocioso (min)', xlabel='')
		axs[1].get_legend().remove()

		sns.barplot(x='date', y='reward', hue='policy', data=data, palette='pastel', ax=axs[2])
		axs[2].set(ylabel='Recompensa', xlabel='')
		axs[2].get_legend().remove()

		axs[0].legend(loc='upper right')
		axs[-1].set_xticklabels(axs[-1].get_xticklabels(), rotation=45)

		# for i in axs[0].patches:
		# get_x pulls left or right; get_height pushes up or down
		#	axs[0].text(i.get_x() + .04, i.get_height() + 12000,  str(round((i.get_height()), 2)), fontsize=18, color='black', rotation=45)
		# handles, labels = axs[-1].get_legend_handles_labels()
		# fig.legend(handles, labels, loc='upper center')
		# ax.set_ylim(0, data['amount'].max() + 1)
		# fig.show()
		# plt.show()
		plt.savefig('benchmark.png')

	sns.set_style("ticks", {'axes.grid': True})
	data = pd.read_csv(fn)
	data['date'] = [row['workload'][row['workload'].rfind('_') + 1: row['workload'].rfind('.')] for _, row in
	                data.iterrows()]
	data['s'] = [pd.to_datetime(row['workload'][row['workload'].rfind('_') + 1: row['workload'].rfind('.')]) for _, row
	             in data.iterrows()]
	data = data[data.policy != 'random']
	data = data[data.policy != 'idle_1']
	data.loc[data.policy == 'idle_5', 'policy'] = "Timeout=5"
	data.loc[data.policy == 'idle_10', 'policy'] = "Timeout=10"
	data.loc[data.policy == 'idle_15', 'policy'] = "Timeout=15"
	data.loc[data.policy == 'PPO_2', 'policy'] = "DeepShutdown"
	data = data.sort_values(by='s')
	plot_reward(data)


def plot_gantt(env_id, workload, policies, weights):
	def run_idles():
		all_jobs = []
		for p in policies:
			OffReservationGridEnv.IDLE_TIME = p
			OffReservationGridEnv.SIMULATION_TIME = 1440
			env = gym.make(env_id)
			env.reset()
			while True:
				state, reward, done, info = env.step(0)
				if done: break
			jobs = info['trace']['jobs']
			pd.DataFrame(jobs).to_csv("tmp.csv", index=False)
			jobs = JobSet.from_csv("tmp.csv")
			os.remove("tmp.csv")

			pstates = pd.DataFrame(info['trace']['host_pstates'])
			pstates.drop_duplicates(subset=['time', 'machine_id'], keep='last', inplace=True)
			pstates.to_csv("host_pstates.csv", index=False)
			pstates = PowerStatesChanges("host_pstates.csv")
			os.remove("host_pstates.csv")

			all_jobs.append((jobs, pstates))
		return all_jobs

	def run_ds():
		OffReservationGridEnv.IDLE_TIME = -1
		OffReservationGridEnv.SIMULATION_TIME = 1440
		agent = PPOAgent(env_id, 123, 20, normalize_obs=False, clip_obs=None, is_episodic=False)
		agent.compile(
			p_network=lstm_mlp(128, (20, agent.input_shape[-1] // 20), [64], activation=tf.nn.leaky_relu),
			shared=True,
			batch_size=1,
			epochs=1,
			lr=5e-4,
			end_lr=5e-6,
			ent_coef=0.0,
			vf_coef=1.,
			decay_steps=250,
			max_grad_norm=None,
			summ_dir=None)
		agent.load(weights)

		infos = agent.play(False, False, 1)

		jobs = infos['trace'][-1]['jobs']
		pd.DataFrame(jobs).to_csv("tmp.csv", index=False)
		jobs = JobSet.from_csv("tmp.csv")
		os.remove("tmp.csv")

		pstates = pd.DataFrame(infos['trace'][-1]['host_pstates'])
		pstates.drop_duplicates(subset=['time', 'machine_id'], keep='last', inplace=True)
		pstates.to_csv("host_pstates.csv", index=False)
		pstates = PowerStatesChanges("host_pstates.csv")
		os.remove("host_pstates.csv")

		return (jobs, pstates)

	SimulatorHandler.WORKLOADS = SIMULATOR_WORKLOAD
	SimulatorHandler.PLATFORM = SIMULATOR_PLATFORM
	OffReservationGridEnv.TRACE = True

	sns.set_style("ticks", {'axes.grid': True})
	print("Running {}".format(workload))
	rmtree(WORKLOAD_PATH)
	os.makedirs(WORKLOAD_PATH, exist_ok=True)
	copyfile("workloads/" + workload, WORKLOAD_PATH + "/{}".format(workload))
	bat_jobs = run_idles()
	ds_jobs = run_ds()
	fig, ax_list = plt.subplots(3, sharex=True, sharey=False, figsize=(32, 22))
	# fig.subplots_adjust(bottom=0.05, right=0.95, top=0.95, left=0.05)
	plot_gantt_pstates(
		bat_jobs[0][0],
		bat_jobs[0][1],
		ax_list[0],
		title="",
		labels=False,
		off_pstates=set([0]),
		son_pstates=set([2]),
		soff_pstates=set([3]))
	plot_gantt_pstates(
		bat_jobs[1][0],
		bat_jobs[1][1],
		ax_list[1],
		title="",
		labels=False,
		off_pstates=set([0]),
		son_pstates=set([2]),
		soff_pstates=set([3]))

	plot_gantt_pstates(
		ds_jobs[0],
		ds_jobs[1],
		ax_list[2],
		title="",
		labels=False,
		off_pstates=set([0]),
		son_pstates=set([2]),
		soff_pstates=set([3]))
	# ax_list[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.show()


# fig.savefig("{}_gantt.png".format(workload))


def get_distributions(env_id, clusters, weights):
	#SimulatorHandler.PLATFORM = SIMULATOR_PLATFORM
	SimulatorHandler.WORKLOADS = SIMULATOR_WORKLOAD
	dist = defaultdict(list)
	agent = PPOAgent(env_id, 123, 20, normalize_obs=False, clip_obs=None, is_episodic=True)
	agent.compile(
		p_network=lstm_mlp(128, (20, agent.input_shape[-1] // 20), [64], activation=tf.nn.leaky_relu),
		shared=True,
		batch_size=1,
		epochs=1,
		lr=5e-4,
		end_lr=5e-6,
		ent_coef=0.0,
		vf_coef=1.,
		decay_steps=250,
		max_grad_norm=None,
		summ_dir=None)
	agent.load(weights)
	for cluster, workloads in clusters.items():
		for w_i, workload in enumerate(workloads):
			sns.set_style("ticks", {'axes.grid': True})
			print("Running {}".format(workload))
			rmtree(WORKLOAD_PATH)
			os.makedirs(WORKLOAD_PATH, exist_ok=True)
			copyfile(workload, WORKLOAD_PATH + "/workload.json")
			infos = agent.play(False, False, 2)
			historic = infos['historic'][-1]
			for k, v in historic.items():
				steps = len(v)
				for value in v:
					dist[k].append(value)

			for _ in range(steps):
				dist['cluster'].append(cluster)
				dist['workload'].append(w_i)

		pd.DataFrame(dist).to_csv('history.csv')


def plot_dist(fn):
	sns.set_style("ticks", {'axes.grid': True})
	df = pd.read_csv(fn)
	ax = sns.barplot(x='action', y='freq', hue='cluster', data=df, palette='pastel')
	ax.set(ylabel='Frequência', xlabel='Tamanho da Reserva')
	plt.savefig("distribution.png")


def get_workloads(workloads_path):
	workloads = []
	for w in os.listdir(workloads_path):
		if w.endswith('.json'):
			workloads.append(w)
	return workloads

if __name__ == "__main__":
	# plot("benchmark.csv")
	policies = [1, 5, 10]
	weights = "2600"
	# plot_gantt("off_reservation-v0", "lyon_taurus_2016_2016-02-21.json", policies, weights)
	#workloads = ['lyon_taurus_2015_cluster_7_1.json', 'lyon_taurus_2015_cluster_6_8.json',
	 #            'lyon_taurus_2015_cluster_4_7.json']
	clusters = defaultdict(list)
	i = 0
	for cluster in os.listdir('clusters'):
		for workload in get_workloads('clusters/{}/'.format(cluster)):
			clusters[cluster] .append('clusters/{}/{}'.format(cluster, workload))


	get_distributions("off_reservation-v0", clusters, weights)
	#plot_dist('distributions.csv')	df = pd.read_csv("benchmark.csv")
	#print(df.groupby('policy').reward.mean())
