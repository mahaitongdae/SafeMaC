import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# exps = ['GP_random'] # 'GP',
# algos = ['smcc_MacOpt_GP_random', 'smcc_MacOpt_GP_random_bandit_double', 'smcc_MacOpt_GP_random_bandit', 'smcc_MacOpt_GP_random_double']
envs = ['sparse'] # 'GP',
# algos = ['smcc_MacOpt_GP', 'smcc_MacOpt_GP_bandit', 'smcc_MacOpt_GP_bandit_double', 'smcc_MacOpt_GP_double']
plot_labels = ['instant_regret', 'regret']
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, 'experiments')
def plot():
    for env in envs:
        dfs = []
        env_dir = os.path.join(data_dir, env)
        for sub_env in os.listdir(env_dir): # should be env
            if not os.path.isdir(os.path.join(env_dir, sub_env)):
                continue
            for algo in os.listdir(os.path.join(env_dir, sub_env)):
                try:
                    df = pd.read_csv(os.path.join(os.path.join(os.path.join(env_dir, sub_env), algo), 'data.csv'))
                    df['algo'] = algo
                    dfs.append(df)
                except:
                    pass
        total_df = pd.concat(dfs, ignore_index=True)
        for label in plot_labels:
            fig, ax = plt.subplots(figsize=[6,4])
            sns.set_style('darkgrid')
            sns.lineplot(x='iter', y=label, hue='algo',data=total_df)
            if label == 'instant_regret':
                # plt.yscale('log')
                plt.gca().invert_yaxis()
            fig_name = label + '.pdf'
            plt.savefig(os.path.join(env_dir, fig_name))

if __name__ == '__main__':
    plot()