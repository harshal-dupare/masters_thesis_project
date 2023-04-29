import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import gif
from datahandler import *
from ea_helpers import *

class Logger:Ū
    def __init__(self, log_name: str, data_handler:Data_Handler):
        self.log_name = log_name
        self.list_R = []
        self.list_L = []
        self.list_O = []
        self.EA_RLO_logs = dict()
        self.EA_archive_RLO = dict()

        self.data_handler = data_handler
        pass

    def clear(self):
        self.list_R = []
        self.list_L = []
        self.list_O = []
        self.EA_RLO_logs = dict()
        self.EA_archive_RLO = dict()
        pass

    def log_RLO(self, R, L, O):
        self.list_L += [float(L)]
        self.list_R += [float(R)]
        self.list_O += [float(O)]
        pass
    
    def log_EA_RLO(self,gen,R,L,O):
        R = float(R)
        L = float(L)
        O = float(O)
        if gen in self.EA_RLO_logs.keys():
            self.EA_RLO_logs[gen][0].append(R)
            self.EA_RLO_logs[gen][1].append(L)
            self.EA_RLO_logs[gen][2].append(O)
        else:
            self.EA_RLO_logs[gen] =  [[R],[L],[O]]
        pass  
        
    def log_EA_Population(self, gen, P): 
        R = [float(P[i].R) for i in range(len(P))]
        L = [float(P[i].L) for i in range(len(P))]
        O = [float(self.data_handler.compute_orthogonal_condition_normF(P[i].A)) for i in range(len(P))]
        self.EA_RLO_logs[gen] = [R,L,O]

    def log_archive(self,gen,Arch):
        R = [float(Arch[i].R) for i in range(len(Arch))]
        L = [float(Arch[i].L) for i in range(len(Arch))]
        O = [float(self.data_handler.compute_orthogonal_condition_normF(Arch[i].A)) for i in range(len(Arch))]
        self.EA_archive_RLO[gen] = [R,L,O]
        pass
    
    def dump(self,folder):
        if len(self.list_L) > 0:
            df = pd.DataFrame()
            df['R'] = self.list_R
            df['L'] = self.list_L
            df['O'] = self.list_O
            fname = "RLO_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            df.to_csv(fpath, index = False)
            print(f" {self.log_name} : dumped ROL")
        
        if len(self.EA_RLO_logs) > 0:
            df = pd.DataFrame()
            for k,v in self.EA_RLO_logs.items():
                df[f"{k}_R"] = v[0]
                df[f"{k}_L"] = v[1]
                df[f"{k}_O"] = v[2]
            fname = "EARLO_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            df.to_csv(fpath, index = False)
            print(f" {self.log_name} : dumped EARLO")

        if len(self.EA_archive_RLO) > 0:
            l = []
            for k,v in self.EA_archive_RLO.items():
                l.append((f"{k}_R", pd.Series(v[0])))
                l.append((f"{k}_L", pd.Series(v[1])))
                l.append((f"{k}_O", pd.Series(v[2])))
            df = pd.DataFrame(dict(l))
            fname = "EAarchiveRLO_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            df.to_csv(fpath, index = False)
            print(f" {self.log_name} : dumped EAarchiveROL")
        pass

    def load(self,rlo_file=None, earlo_file=None, eaarchiverl_file=None):
        if rlo_file is not None:
            df = pd.read_csv(rlo_file)
            self.list_R = list(df['R'])
            self.list_L = list(df['L'])
            self.list_O = list(df['O'])
        else:
            self.list_R = []
            self.list_L = []
            self.list_O = []
        
        self.EA_RLO_logs = dict()
        if earlo_file is not None:
            df = pd.read_csv(earlo_file)
            for k in df.columns:
                g = int(k.split('_')[0])
                if g in self.EA_RLO_logs.keys():
                    continue
                self.EA_RLO_logs[g] = [list(df[g+"_R"]),list(df[g+"_L"]),list(df[g+"_O"])]
        
        self.EA_archive_RLO = dict()
        if eaarchiverl_file is not None:
            df = pd.read_csv(eaarchiverl_file)
            for k in df.columns:
                g = int(k.split('_')[0])
                if g in self.EA_archive_RLO.keys():
                    continue
                self.EA_archive_RLO[g] = [list(df[g+"_R"]),list(df[g+"_L"]),list(df[g+"_O"])]
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
    
    def get_gif_from_RL_log_dict(self, duration=200,style='seaborn',xscale='log',yscale='log',dpi=300,lam_margins=1.2, set_boundaries = False, use_archive_instead=False):
        plt.style.use(style)
        gif.options.matplotlib["dpi"] = dpi

        def get_min_max_limits(lam):
            R_max = -float('inf')
            R_min = float('inf')
            L_max = -float('inf')
            L_min = float('inf')
            if use_archive_instead:
                for k,v in self.EA_archive_RLO.items():
                    R_max = max(np.max(v[0]),R_max)
                    R_min = min(np.min(v[0]),R_min)
                    L_max = max(np.max(v[1]),L_max)
                    L_min = min(np.min(v[1]),L_min)
            else:
                for k,v in self.EA_RLO_logs.items():
                    R_max = max(np.max(v[0]),R_max)
                    R_min = min(np.min(v[0]),R_min)
                    L_max = max(np.max(v[1]),L_max)
                    L_min = min(np.min(v[1]),L_min)
            return 0.5*np.asarray([  (1+lam)*R_min+(1-lam)*R_max, (1-lam)*R_min+(1+lam)*R_max, (1+lam)*L_min+(1-lam)*L_max, (1-lam)*L_min+(1+lam)*L_max])
            
        limits = get_min_max_limits(lam_margins)
        @gif.frame
        def plot(key,value):
            # sns.set(style="whitegrid")
            fig, ax = plt.subplots()
            ax.scatter(x=value[0], y=value[1], color='red')
            if set_boundaries:
                ax.set_xlim([limits[0], limits[1]])
                ax.set_ylim([limits[2], limits[3]])
            ax.set_xlabel('Reward')
            ax.set_ylabel('Loss')
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)
            ax.set_title(f"{self.log_name} Generation : {key}")
        frames = []
        if use_archive_instead:
            frames += [ plot(k,v) for k,v in self.EA_archive_RLO.items()]
        else:
            frames += [ plot(k,v) for k,v in self.EA_RLO_logs.items()]
        filename = f"{self.log_name}" + ".gif"
        gif.save(frames, filename, duration=duration)
        pass

    def get_tragectory_gif(self,duration=200,style='seaborn',xscale='log',yscale='log',dpi=300,lam_margins=1.2,set_boundaries = False):
        plt.style.use(style)
        gif.options.matplotlib["dpi"] = dpi

        def get_min_max_limits(lam):
            R_max = -float('inf')
            R_min = float('inf')
            L_max = -float('inf')
            L_min = float('inf')
            for k,v in self.EA_RLO_logs.items():
                R_max = max(np.max(v[0]),R_max)
                R_min = min(np.min(v[0]),R_min)
                L_max = max(np.max(v[1]),L_max)
                L_min = min(np.min(v[1]),L_min)
            return 0.5*np.asarray([  (1+lam)*R_min+(1-lam)*R_max, (1-lam)*R_min+(1+lam)*R_max, (1+lam)*L_min+(1-lam)*L_max, (1-lam)*L_min+(1+lam)*L_max])
        
        limits = get_min_max_limits(lam_margins)
        @gif.frame
        def plot(gen_i):
            fig, ax = plt.subplots()
            for point in range(len(self.EA_RLO_logs[gen_i][0])):
                point_Rs = [self.EA_RLO_logs[gg][0][point] for gg in range(gen_i+1)]
                points_Ls = [self.EA_RLO_logs[gg][1][point] for gg in range(gen_i+1)]
                # ax.plot(point_Rs, points_Ls, label=f'Point {point+1}')
                ax.plot(point_Rs, points_Ls)
                if set_boundaries:
                    ax.set_xlim([limits[0], limits[1]])
                    ax.set_ylim([limits[2], limits[3]])

            # Set plot properties
            ax.set_xlabel('Reward')
            ax.set_ylabel('Loss')
            ax.set_title(f"Trajectories of Points: {gen_i}")
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)

        gen_max = len(self.EA_RLO_logs)
        frames = [ plot(i) for i in range(gen_max)]
        filename = f"{self.log_name}-tragectory" + ".gif"
        gif.save(frames, filename, duration=duration)
        pass

    def show_tragetctory_plot(self,style='seaborn',xscale='log',yscale='log'):
        plt.style.use(style)
        fig, ax = plt.subplots()
        gen_max = len(self.EA_RLO_logs)
        # Plot each point's trajectory as a line
        for point in range(len(self.EA_RLO_logs[0][0])):
            point_Rs = [self.EA_RLO_logs[gg][0][point] for gg in range(gen_max)]
            points_Ls = [self.EA_RLO_logs[gg][1][point] for gg in range(gen_max)]
            ax.plot(point_Rs, points_Ls, label=f'Point {point+1}')

        # Set plot properties
        ax.set_xlabel('Reward')
        ax.set_ylabel('Loss')
        ax.set_title('Trajectories of Points')
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        # ax.legend()
        plt.show()

    def get_grad_gif(self,duration=200,style='seaborn',xscale='log',yscale='log',dpi=300,lam_margins=1.2,set_boundaries = False):
        plt.style.use(style)
        gif.options.matplotlib["dpi"] = dpi

        def get_min_max_limits(lam):
            R_max = -float('inf')
            R_min = float('inf')
            L_max = -float('inf')
            L_min = float('inf')
            for k,v in self.EA_RLO_logs.items():
                R_max = max(np.max(v[0]),R_max)
                R_min = min(np.min(v[0]),R_min)
                L_max = max(np.max(v[1]),L_max)
                L_min = min(np.min(v[1]),L_min)
            return 0.5*np.asarray([  (1+lam)*R_min+(1-lam)*R_max, (1-lam)*R_min+(1+lam)*R_max, (1+lam)*L_min+(1-lam)*L_max, (1-lam)*L_min+(1+lam)*L_max])
        
        limits = get_min_max_limits(lam_margins)
        @gif.frame
        def plot(gen_i):
            fig, ax = plt.subplots()
            for point in range(len(self.EA_RLO_logs[gen_i][0])):
                point_Rs = [self.EA_RLO_logs[gg][0][point] for gg in range(gen_i-1,gen_i+1)]
                points_Ls = [self.EA_RLO_logs[gg][1][point] for gg in range(gen_i-1,gen_i+1)]
                # ax.plot(point_Rs, points_Ls, label=f'Point {point+1}')
                ax.plot(point_Rs, points_Ls)
                if set_boundaries:
                    ax.set_xlim([limits[0], limits[1]])
                    ax.set_ylim([limits[2], limits[3]])

            # Set plot properties
            ax.set_xlabel('Reward')
            ax.set_ylabel('Loss')
            ax.set_title(f"Trajectories of Points: {gen_i}")
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)

        gen_max = len(self.EA_RLO_logs)
        frames = [ plot(i) for i in range(1,gen_max)]
        filename = f"{self.log_name}-directiongrad" + ".gif"
        gif.save(frames, filename, duration=duration)
        pass


    def pareto_front_plot(self):
        """
        using all the solution ever generated in a method
        """
        pass

    def get_3d_graph(self):
        pass

    def compute_core_statistics(folder, self, last_from_list__=200, beyond_gen=10):
        if len(self.list_L) > 0:
            df = pd.DataFrame()
            df['R'] = self.list_R[-last_from_list__:]
            df['L'] = list(-np.log(self.list_L[-last_from_list__:]))
            df['O'] = list(-np.log(self.list_O[-last_from_list__:]))
            dfd = df.describe()
            print(dfd)

            fname = f"CORE_STATS_{last_from_list__}_RLO_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            dfd.to_csv(fpath, index = False)
            print(f" {fname} : dumped ROL")
        
        if len(self.EA_RLO_logs) > 0:
            from_dict = self.EA_RLO_logs

            df = pd.DataFrame()
            df['R'] = []
            df['L'] = []
            df['O'] = []
            df['gen'] = []
            for k,v in from_dict.items():
                if k < beyond_gen:
                    continue
                df1 = pd.DataFrame()
                df1["R"] = v[0]
                df1["L"] = list(-np.log(v[1]))
                df1["O"] = list(-np.log(v[2]))
                df1['gen'] = len(v[0])*[k]
                df = pd.concat([df, df1], axis=0)
            dfd = df.describe()
            grouped_df = df.groupby('gen').mean()
            print(dfd)
            print(grouped_df)

            fname = "CORE_STATS_EARLO_{beyond_gen}_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            dfd.to_csv(fpath, index = False)
            print(f" {fname} : dumped EARLO")

            fname = "gen_change_EARLO_{beyond_gen}_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            dfd.to_csv(fpath, index = False)
            print(f" {fname} : dumped change gen EARLO")

        if len(self.EA_archive_RLO) > 0:
            from_dict = self.EA_archive_RLO

            df = pd.DataFrame()
            df['R'] = []
            df['L'] = []
            df['O'] = []
            df['gen'] = []
            for k,v in from_dict.items():
                if k < beyond_gen:
                    continue
                df1 = pd.DataFrame()
                df1["R"] = v[0]
                df1["L"] = list(-np.log(v[1]))
                df1["O"] = list(-np.log(v[2]))
                df1['gen'] = len(v[0])*[k]
                df = pd.concat([df, df1], axis=0)
            dfd = df.describe()
            grouped_df = df.groupby('gen').mean()
            print(dfd)
            print(grouped_df)

            fname = "CORE_STATS_EAarchiveROL_{beyond_gen}_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            dfd.to_csv(fpath, index = False)
            print(f" {fname} : dumped EAarchiveROL")

            fname = "gen_change_EAarchiveROL_{beyond_gen}_"+self.log_name+".csv"
            fpath = os.path.join(folder,fname)
            dfd.to_csv(fpath, index = False)
            print(f" {fname} : dumped change gen EAarchiveROL")
        pass
    