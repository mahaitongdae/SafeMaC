import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

exps = ['GP']
algos = ['smcc_MacOpt_GP', 'smcc_MacOpt_GP_double']
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, 'experiments')
def plot():
    for exp in exps:
        dfs = []
        exp_dir = os.path.join(data_dir, exp)
        for dir in os.listdir(exp_dir): # should be env
            for algo in algos:
                df = pd.read_csv(os.path.join(os.path.join(os.path.join(exp_dir, dir), algo), 'data.csv'))
                df['algo'] = algo
                dfs.append(df)
        total_df = pd.concat(dfs, ignore_index=True)
        fig, ax = plt.subplots(figsize=[6,4])
        sns.set_style('darkgrid')
        sns.lineplot(x='iter', y='regret', hue='algo',data=total_df)
        plt.yscale('log')
        plt.gca().invert_yaxis()
        plt.show()

if __name__ == '__main__':
    plot()