import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
plt.rc('text', usetex=True)
envs = ['random_0.01',] # 'GP_0.01', # ,'sparse_0.1' # 'GP_0.001' 'GP_0.01', 'GP_0.01', 'random_0.01'
plot_labels = ['regret']
plot_names = {'bandit': 'length scale 0.01', 'base': 'length scale 0.5'} #  'base': 'correlation kernel'
env_names = {'GP': r'$w_{\rm Normal}$', 'random': r'$w_{\rm Uniform}$', 'sparse': r'$w_{\rm Sparse}$'}
algo_names = {'double': 'MAC-DT', 'base': 'MacOpt-SP', 'voronoi': 'Voronoi'}
ylim = {'GP': [0, 1.7],
        'random': [0, 1.5],
        'sparse': [0, 1.3]}
# root_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(root_dir, 'experiments')
data_dir = '/home/mht/PycharmProjects/safemac_data/experiments'
sns.set_style('darkgrid',)
sns.set(font_scale=1.5)
def plot():
    for plot_key, plot_val in plot_names.items():
        for env in envs: #
            if not os.path.isdir(os.path.join(data_dir, env)):
                continue
            print('plotting env {}'.format(env))
            dfs = []
            env_dir = os.path.join(data_dir, env)
            for sub_env in os.listdir(env_dir): # should be env
                if not os.path.isdir(os.path.join(env_dir, sub_env)):
                    continue
                for algo in sorted(os.listdir(os.path.join(env_dir, sub_env))):
                    # try:
                    if not os.path.isdir(os.path.join(os.path.join(env_dir, sub_env), algo)):
                        continue
                    if os.listdir(os.path.join(os.path.join(env_dir, sub_env), algo)) == []:
                        continue
                    if algo.split('_')[1] == plot_key:
                        df = pd.read_csv(os.path.join(os.path.join(os.path.join(env_dir, sub_env), algo), 'data.csv'))
                        df['algorithm'] = algo_names[algo.split('_')[-1]]
                        df['regret'] = df['regret'] / 1000.
                        dfs.append(df)
                    # except:
                    #     pass
            total_df = pd.concat(dfs, ignore_index=True)
            for label in plot_labels:
                fig, ax = plt.subplots(figsize=[4, 3])
                sns.set_style('darkgrid')
                # sns.set(font_scale=1.)
                sns.lineplot(x='iter', y=label, hue='algorithm', data=total_df, ax=ax)
                title = env_names[env.split('_')[0]] + ', ' + plot_val
                ax.set_title(title)
                ax.set_xlabel('Samples')
                ax.set_ylabel(r'Regret ($\times 10^3$)')
                if label == 'instant_regret':
                    # plt.yscale('log')
                    plt.gca().invert_yaxis()
                ax.set_ylim(ylim.get(env.split('_')[0]))
                plt.tight_layout(pad=0.3)
                # plt.xlim(0, 500)
                h, l = ax.get_legend_handles_labels()
                ax.get_legend().remove()
                fig_name = env + '_' + label + '_' + plot_val + '_paper.pdf'
                plt.savefig(os.path.join(data_dir, fig_name))

    legfig, legax = plt.subplots(figsize=(7.5, 0.75))
    legax.set_facecolor('white')
    leg = legax.legend(h, l, loc='center', ncol=len(l), handlelength=1.5,
                       mode="expand", borderaxespad=0., prop={'size': 13})
    legax.xaxis.set_visible(False)
    legax.yaxis.set_visible(False)
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(data_dir, 'legend.pdf'),
                bbox_inches='tight')

if __name__ == '__main__':
    plot()