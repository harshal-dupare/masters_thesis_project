import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, log_name: str):
        self.log_name = log_name
        self.list_R = []
        self.list_L = []
        self.EA_RL_logs = dict()
        pass

    def clear(self):
        self.list_L = []
        self.list_R = []
        pass

    def log_RL(self, R, L):
        self.list_L += [float(L)]
        self.list_R += [float(R)]
        pass
    
    def log_EA_RL(self,gen,R,L):
        self.EA_RL_logs[gen] = self.EA_RL_logs.get(gen,[]) + [(-R,L)]
        pass
    
    def plot_scatter_LR(self, x_scale="linear", y_scale="linear", save=False, file_name="fig.png"):
        sns.set_style("darkgrid")
        sns.set_palette("bright")
        x_label = "Reward"
        y_label = "Loss"
        ax = sns.scatterplot(x=self.list_R, y=self.list_L)
        ax.set(xscale=x_scale, yscale=y_scale, xlabel=x_label, ylabel=y_label)
        if save:
            plt.savefig(file_name)
        plt.show()

    def pair_plot_LR(self, save=False, filename='pairplot.png'):
        """
        Very slow for 1000+ points
        """
        data = pd.DataFrame({'Reward': self.list_R, 'Loss': self.list_L})
        sns.set_style('whitegrid')
        pair_plot = sns.pairplot(data, height=2.5)
        if save:
            pair_plot.savefig(filename)
        plt.show()

    def dist_plot(self, save=False, filename='distplot.png'):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
        sns.set_style('whitegrid')
        sns.distplot(self.list_R, label='Reward', ax=ax1)
        ax1.set(title='Distribution of Rewards')
        sns.distplot(self.list_L, label='Loss', ax=ax2)
        ax2.set(title='Distribution of Loss')
        ax1.legend()
        ax2.legend()
        if save:
            plt.savefig(filename)
        plt.show()
