import os
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import gif
import copy
from datahandler import *
from ea_helpers import *


class PlotSettings:
    def __init__(self, x_axis = 'R',y_axis = '-logO',z_axis = '-logL', alpha=0.5, s=100, ec='none', use_xyz_limits=False, lam=1.2, remove_outliers=False,
                 outlier_tail_lower=0.025, outlier_tail_upper=0.025, cmap='cool', plot_style='seaborn-v0_8-whitegrid', dpi=300,
                 x_scale='linear', y_scale='linear', duration=200, traj_font_size=14, plot_size=(20,16), 
                 three_d_plot_elev=20, three_d_plot_azim=45, traj_color='blue', traj_arrow_size=1,
                 traj_arrow_length_ratio=0.1, traj_alpha=0.8, traj_linewidths=0.5, traj_edgecolors='k', 
                 traj_pivot='tip', traj_lines_linewidth=2, two_d_traj_scale=1, two_d_traj_width=0.005, use_quiver = True,add_rud_at_start=''):
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._z_axis = z_axis
        self._alpha = alpha
        self._add_rud_at_start = add_rud_at_start
        self._s = s
        self._ec = ec
        self._use_xyz_limits = use_xyz_limits
        self._lam = lam
        self._remove_outliers = remove_outliers
        self._outlier_tail_lower = outlier_tail_lower
        self._outlier_tail_upper = outlier_tail_upper
        self._cmap = cmap
        self._plot_style = plot_style
        self._dpi = dpi
        self._x_scale = x_scale
        self._y_scale = y_scale
        self._duration = duration
        self._traj_font_size = traj_font_size
        self._plot_size = plot_size
        self._3d_plot_elev = three_d_plot_elev
        self._3d_plot_azim = three_d_plot_azim
        self._traj_color = traj_color
        self._traj_arrow_size = traj_arrow_size
        self._traj_arrow_length_ratio = traj_arrow_length_ratio
        self._traj_alpha = traj_alpha
        self._traj_linewidths = traj_linewidths
        self._traj_edgecolors = traj_edgecolors
        self._traj_pivot = traj_pivot
        self._traj_lines_linewidth = traj_lines_linewidth
        self._2d_traj_scale = two_d_traj_scale
        self._2d_traj_width = two_d_traj_width
        self._use_quiver = use_quiver 
        tab20 = plt.get_cmap('tab20')
        self.color_names = [tab20.colors[i] for i in range(tab20.N)]
        self.color_names_proper = ['g','b','r','c','m','y','k','w']
        self.swap_buffer = None
        self.save_combined_data = False

    def store_in_buffer(self,var):
        self.swap_buffer = var
        return

class Logger:
    def __init__(self, log_name: str, data_handler: Data_Handler, _img_save_folder=None,_add_rud_at_start=''):
        self.log_name = log_name
        self.list_R = []
        self.list_L = []
        self.list_O = []
        self.EA_RLO_logs = dict()
        self.EA_archive_RLO = dict()
        self.RLO_iter_logs = dict()
        self._add_rud_at_start = _add_rud_at_start

        self.data_handler = data_handler
        self.IMAGE_SAVE_FOLDER = _img_save_folder
        pass

    def clear(self):
        self.list_R = []
        self.list_L = []
        self.list_O = []
        self.EA_RLO_logs = dict()
        self.EA_archive_RLO = dict()
        self.RLO_iter_logs = dict()
        pass

    def log_RLO(self, R, L, O):
        self.list_L += [float(L)]
        self.list_R += [float(R)]
        self.list_O += [float(O)]
        pass

    def log_EA_RLO(self, gen, R, L, O):
        R = float(R)
        L = float(L)
        O = float(O)
        if gen in self.EA_RLO_logs.keys():
            self.EA_RLO_logs[gen][0].append(R)
            self.EA_RLO_logs[gen][1].append(L)
            self.EA_RLO_logs[gen][2].append(O)
        else:
            self.EA_RLO_logs[gen] = [[R], [L], [O]]
        pass

    def log_EA_Population(self, gen, P):
        R = [float(P[i].R) for i in range(len(P))]
        L = [float(P[i].L) for i in range(len(P))]
        O = [float(self.data_handler.compute_orthogonal_condition_normF(P[i].A))
             for i in range(len(P))]
        self.EA_RLO_logs[gen] = [R, L, O]

    def log_archive(self, gen, Arch):
        R = [float(Arch[i].R) for i in range(len(Arch))]
        L = [float(Arch[i].L) for i in range(len(Arch))]
        O = [float(self.data_handler.compute_orthogonal_condition_normF(
            Arch[i].A)) for i in range(len(Arch))]
        self.EA_archive_RLO[gen] = [R, L, O]
        pass

    def dump(self, folder):
        if len(self.list_L) > 0:
            df = pd.DataFrame()
            df['R'] = self.list_R
            df['L'] = self.list_L
            df['O'] = self.list_O
            fname = "RLO_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            df.to_csv(fpath, index=False)
            print(f" {self.log_name} : dumped ROL")

        if len(self.EA_RLO_logs) > 0:
            df = pd.DataFrame()
            for k, v in self.EA_RLO_logs.items():
                df[f"{k}_R"] = v[0]
                df[f"{k}_L"] = v[1]
                df[f"{k}_O"] = v[2]
            fname = "EARLO_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            df.to_csv(fpath, index=False)
            print(f" {self.log_name} : dumped EARLO")

        if len(self.EA_archive_RLO) > 0:
            l = []
            for k, v in self.EA_archive_RLO.items():
                l.append((f"{k}_R", pd.Series(v[0])))
                l.append((f"{k}_L", pd.Series(v[1])))
                l.append((f"{k}_O", pd.Series(v[2])))
            df = pd.DataFrame(dict(l))
            fname = "EAarchiveRLO_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            df.to_csv(fpath, index=False)
            print(f" {self.log_name} : dumped EAarchiveROL")
        pass

    def load(self, rlo_file=None, earlo_file=None, eaarchiverl_file=None, iter_files=[]):
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
                self.EA_RLO_logs[g] = [list(df[f"{g}_R"]), list(df[f"{g}_L"]), list(df[f"{g}_O"])]

        self.EA_archive_RLO = dict()
        if eaarchiverl_file is not None:
            df = pd.read_csv(eaarchiverl_file)
            for k in df.columns:
                g = int(k.split('_')[0])
                if g in self.EA_archive_RLO.keys():
                    continue
                self.EA_archive_RLO[g] = [list(df[f"{g}_R"]), list(df[f"{g}_L"]), list(df[f"{g}_O"])]

        if len(iter_files) > 0:
            for i, f in enumerate(iter_files):
                df = pd.read_csv(f)
                self.RLO_iter_logs[i] = [[val for val in list(df['R']) if str(val) != 'nan'],
                                         [val for val in list(df['L']) if str(val) != 'nan'],
                                         [val for val in list(df['O']) if str(val) != 'nan']]
        return

    def compute_core_statistics(self, folder, last_from_list__=200, beyond_gen_iter=10):
        if len(self.list_L) > 0:
            df = pd.DataFrame()
            df['R'] = self.list_R[-last_from_list__:]
            df['-logL'] = list(-np.log(self.list_L[-last_from_list__:]))
            df['-logO'] = list(-np.log(self.list_O[-last_from_list__:]))
            dfd = df.describe()
            print(dfd)

            fname = f"CORE_STATS_{last_from_list__}_RLO_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            dfd.to_csv(fpath)
            print(f" {fname} : dumped ROL")

        if len(self.RLO_iter_logs) > 0:
            from_dict = self.RLO_iter_logs

            df = pd.DataFrame()
            df['R'] = []
            df['-logL'] = []
            df['-logO'] = []
            df['index'] = []
            for attempt_i, v in from_dict.items():
                df1 = pd.DataFrame()
                df1["R"] = v[0][beyond_gen_iter:]
                df1["-logL"] = list(-np.log(v[1][beyond_gen_iter:]))
                df1["-logO"] = list(-np.log(v[2][beyond_gen_iter:]))
                df1['index'] = list(
                    range(beyond_gen_iter, beyond_gen_iter+len(v[0][beyond_gen_iter:])))
                df = pd.concat([df, df1], axis=0)
            dfd = df.describe()
            grouped_df = df.groupby('index').mean()
            print(dfd)
            print(grouped_df)

            fname = f"CORE_STATS_EARLO_{beyond_gen_iter}_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            dfd.to_csv(fpath)
            print(f" {fname} : dumped EARLO")

            fname = f"index_change_EARLO_{beyond_gen_iter}_" + \
                self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            grouped_df.to_csv(fpath)
            print(f" {fname} : dumped change gen EARLO")

        if len(self.EA_RLO_logs) > 0:
            from_dict = self.EA_RLO_logs

            df = pd.DataFrame()
            df['R'] = []
            df['-logL'] = []
            df['-logO'] = []
            df['gen'] = []
            for k, v in from_dict.items():
                if k < beyond_gen_iter:
                    continue
                df1 = pd.DataFrame()
                df1["R"] = v[0]
                df1["-logL"] = list(-np.log(v[1]))
                df1["-logO"] = list(-np.log(v[2]))
                df1['gen'] = len(v[0])*[k]
                df = pd.concat([df, df1], axis=0)
            dfd = df.describe()
            grouped_df = df.groupby('gen').mean()
            print(dfd)
            print(grouped_df)

            fname = f"CORE_STATS_EARLO_{beyond_gen_iter}_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            dfd.to_csv(fpath)
            print(f" {fname} : dumped EARLO")

            fname = f"gen_change_EARLO_{beyond_gen_iter}_"+self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            grouped_df.to_csv(fpath)
            print(f" {fname} : dumped change gen EARLO")

        if len(self.EA_archive_RLO) > 0:
            from_dict = self.EA_archive_RLO

            df = pd.DataFrame()
            df['R'] = []
            df['-logL'] = []
            df['-logO'] = []
            df['gen'] = []
            for k, v in from_dict.items():
                if k < beyond_gen_iter:
                    continue
                df1 = pd.DataFrame()
                df1["R"] = v[0]
                df1["-logL"] = list(-np.log(v[1]))
                df1["-logO"] = list(-np.log(v[2]))
                df1['gen'] = len(v[0])*[k]
                df = pd.concat([df, df1], axis=0)
            dfd = df.describe()
            grouped_df = df.groupby('gen').mean()
            print(dfd)
            print(grouped_df)

            fname = f"CORE_STATS_EAarchiveROL_{beyond_gen_iter}_" + \
                self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            dfd.to_csv(fpath)
            print(f" {fname} : dumped EAarchiveROL")

            fname = f"gen_change_EAarchiveROL_{beyond_gen_iter}_" + \
                self.log_name+".csv"
            fpath = os.path.join(folder, fname)
            grouped_df.to_csv(fpath)
            print(f" {fname} : dumped change gen EAarchiveROL")

    def get_save_name(self,fname):
        return os.path.join(self.IMAGE_SAVE_FOLDER,self._add_rud_at_start+fname)
    # MAIN
    def plot_and_save_figures(self, plotsetting, beyond_gen_iter=10):
        if len(self.list_L) > 0:
            df = pd.DataFrame()
            df['R'] = self.list_R
            df['-logL'] = list(-np.log(self.list_L))
            df['-logO'] = list(-np.log(self.list_O))
            df = self.remove_duplicates_lossy(df)

            self.scatter_plot_2d_3d(df,plotsetting)
            self.dist_plot(df,plotsetting)

        if len(self.RLO_iter_logs) > 0:
            from_dict = self.RLO_iter_logs

            df = pd.DataFrame()
            df['R'] = []
            df['-logL'] = []
            df['-logO'] = []
            df['index'] = []
            for attempt_i, v in from_dict.items():
                df1 = pd.DataFrame()
                df1["R"] = v[0][beyond_gen_iter:]
                df1["-logL"] = list(-np.log(v[1][beyond_gen_iter:]))
                df1["-logO"] = list(-np.log(v[2][beyond_gen_iter:]))
                df1['index'] = len(v[0][beyond_gen_iter:])*[attempt_i]
                df = pd.concat([df, df1], axis=0)
            
            
            if plotsetting.save_combined_data:
                self.progress_line_plots_3d(df, 'index', plotsetting)
                exit()

            # for all the attempts it plots in 1 plot all the paths taken by the point
            self.progress_line_plots_3d(df, 'index', plotsetting)
            self.progress_line_plots_2d(df, 'index', plotsetting)
            self.dist_plot(df, plotsetting)

        if len(self.EA_RLO_logs) > 0:
            from_dict = self.EA_RLO_logs

            df = pd.DataFrame()
            df['R'] = []
            df['-logL'] = []
            df['-logO'] = []
            df['gen'] = []
            for k, v in from_dict.items():
                if k < beyond_gen_iter:
                    continue
                df1 = pd.DataFrame()
                df1["R"] = v[0]
                df1["-logL"] = list(-np.log(v[1]))
                df1["-logO"] = list(-np.log(v[2]))
                df1['gen'] = len(v[0])*[k]
                df = pd.concat([df, df1], axis=0)

            if plotsetting.save_combined_data:
                self.progress_line_plots_3d(df, 'gen', plotsetting)
                exit()

            # including tragectroy/gradient plot and gif
            self.population_gif_and_plots(df,plotsetting)
            # merges takes avg of all the points in gen and plots them againt gen as the time arrow or maybe max
            self.progress_line_plots_3d(df, 'gen', plotsetting)
            self.progress_line_plots_2d(df, 'gen', plotsetting)
            self.dist_plot(df,plotsetting)

        if len(self.EA_archive_RLO) > 0:
            from_dict = self.EA_archive_RLO
            df = pd.DataFrame()
            df['R'] = []
            df['-logL'] = []
            df['-logO'] = []
            df['gen'] = []
            for k, v in from_dict.items():
                if k < beyond_gen_iter:
                    continue
                df1 = pd.DataFrame()
                df1["R"] = v[0]
                df1["-logL"] = list(-np.log(v[1]))
                df1["-logO"] = list(-np.log(v[2]))
                df1['gen'] = len(v[0])*[k]
                df = pd.concat([df, df1], axis=0)
            
            if plotsetting.save_combined_data:
                self.progress_line_plots_3d(df, 'gen', plotsetting)
                exit()

            self.progress_line_plots_3d(df, 'gen',plotsetting)
            self.progress_line_plots_2d(df, 'gen',plotsetting)
            self.dist_plot(df,plotsetting)

        pass

    def remove_duplicates_lossy(self, df, epsilon=1e-5):
        df_new = df.copy()
        df_new = df_new.applymap(lambda x: round(
            x, int(-math.log10(epsilon))) if isinstance(x, float) else x)
        return df_new.drop_duplicates()

    def get_plotting_limits(self, df, plotsetting):
        limits = {'low': {}, 'high': {}}
        for c in df.columns:
            df_col = df[c].copy()
            if plotsetting._remove_outliers:
                # Calculate the lower and upper tails of the distribution
                lower_tail, upper_tail = df_col.quantile(
                    [plotsetting._outlier_tail_lower, 1 - plotsetting._outlier_tail_upper])
                # Remove the outliers from the dataframe
                df_col = df_col[(df_col >= lower_tail) &
                                (df_col <= upper_tail)]
            # Calculate the maximum and minimum values of the dataframe
            max_v, min_v = df_col.max(), df_col.min()
            # Calculate the lower and upper limits for the column
            limits['low'][c] = 0.5*((1+plotsetting._lam)*min_v + (1-plotsetting._lam)*max_v)
            limits['high'][c] = 0.5*((1-plotsetting._lam)*min_v + (1+plotsetting._lam)*max_v)
        return limits

    def scatter_plot_2d_3d(self, df, plotsetting):
        # Define a color map based on the z axis values
        cmap = plt.get_cmap(plotsetting._cmap)
        norm = plt.Normalize(df[plotsetting._z_axis].min(), df[plotsetting._z_axis].max())
        colors = cmap(norm(df[plotsetting._z_axis]))

        # Create a 2D scatter plot for each possible pair of columns
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                if i < j:
                    fig, ax = plt.subplots(figsize=plotsetting._plot_size)
                    title_name = self.log_name + ' scatter plot'
                    ax.set_title(title_name)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.scatter(df[col1], df[col2], c=colors, alpha=plotsetting._alpha, s=plotsetting._s, edgecolors=plotsetting._ec)
                    plt.style.use(plotsetting._plot_style)
                    fname = f"{title_name}_{col1}_vs_{col2}"+".png"
                    fname = self.get_save_name(fname)
                    plt.savefig(fname, dpi =plotsetting._dpi)
                    plt.show()

        # Create a 3D scatter plot
        fig = plt.figure(figsize=plotsetting._plot_size)
        ax = fig.add_subplot(111, projection='3d')
        title_name =  self.log_name + " 3D scatter plot"
        ax.set_title(title_name)
        ax.set_xlabel(plotsetting._x_axis)
        ax.set_ylabel(plotsetting._y_axis)
        ax.set_zlabel(plotsetting._z_axis)

        if plotsetting._use_xyz_limits:
            limits = self.get_plotting_limits(df, plotsetting)
            ax.set_xlim((limits['low'][plotsetting._x_axis], limits['high'][plotsetting._x_axis]))
            ax.set_ylim((limits['low'][plotsetting._y_axis], limits['high'][plotsetting._y_axis]))
            ax.set_zlim((limits['low'][plotsetting._z_axis], limits['high'][plotsetting._z_axis]))

        ax.scatter(df[plotsetting._x_axis], df[plotsetting._y_axis], df[plotsetting._z_axis],c=colors, alpha=plotsetting._alpha, s=plotsetting._s, edgecolors=plotsetting._ec)
        plt.style.use(plotsetting._plot_style)
        fname = title_name+".png"
        fname = os.path.join(self.IMAGE_SAVE_FOLDER, fname)
        plt.savefig(fname, dpi =plotsetting._dpi)
        plt.show()
        pass

    def dist_plot(self, df, plotsetting):
        pass

    def plot_trajectory_3d(self, df, ax, plotsetting):
        # Extract x, y, z coordinates from DataFrame
        x = df[plotsetting._x_axis]
        y = df[plotsetting._y_axis]
        z = df[plotsetting._z_axis]
        # Plot trajectory line
        ax.plot(x, y, z, color=plotsetting._traj_color, linewidth=plotsetting._traj_lines_linewidth)
        # Plot arrows for each step in the trajectory
        if plotsetting._use_quiver:
            for i in range(len(x)-1):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                dz = z[i+1] - z[i]
                ax.quiver(x[i], y[i], z[i], dx, dy, dz, color=plotsetting._traj_color, arrow_length_ratio=plotsetting._traj_arrow_length_ratio,
                        pivot=plotsetting._traj_pivot, linewidths=plotsetting._traj_linewidths, edgecolors=plotsetting._traj_edgecolors,
                        alpha=plotsetting._traj_alpha, length=plotsetting._traj_arrow_size)

    def progress_line_plots_3d(self, df, grouping_type, plotsetting:PlotSettings):
        def modify_ax(_ax,_title_name, _legend_list=[]):
            if plotsetting._use_xyz_limits:
                limits = self.get_plotting_limits(df, plotsetting)
                _ax.set_xlim((limits['low'][plotsetting._x_axis], limits['high'][plotsetting._x_axis]))
                _ax.set_ylim((limits['low'][plotsetting._y_axis], limits['high'][plotsetting._y_axis]))
                _ax.set_zlim((limits['low'][plotsetting._z_axis], limits['high'][plotsetting._z_axis]))

            _ax.set_title(_title_name, fontsize=plotsetting._traj_font_size)
            _ax.set_xlabel(plotsetting._x_axis, fontsize=plotsetting._traj_font_size)
            _ax.set_ylabel(plotsetting._y_axis, fontsize=plotsetting._traj_font_size)
            _ax.set_zlabel(plotsetting._z_axis, fontsize=plotsetting._traj_font_size)

            # Customize legend
            if len(_legend_list) > 0:
                _ax.legend(legends_list, fontsize=int(plotsetting._traj_font_size*0.5))

            # customize the viewing angle
            _ax.view_init(elev=plotsetting._3d_plot_elev, azim=plotsetting._3d_plot_azim)

            # Set plot size
            if plotsetting._plot_size:
                fig.set_size_inches(plotsetting._plot_size[0], plotsetting._plot_size[1])

        if grouping_type == 'index':
            # for all the attempts it plots in 1 plot all the paths taken by the point
            df_dict = dict()
            grp = df.groupby(grouping_type)
            for grp_key, grp_df in grp:
                df_dict[grp_key] = grp_df.reset_index().drop('index', axis=1)

            _temp_key = None
            for grp_key, _ in df_dict.items():
                _temp_key = grp_key
                break
            mean_df = copy.deepcopy(df_dict[_temp_key])
            for grp_key, k_val in df_dict.items():
                if grp_key == _temp_key:
                    continue
                mean_df = mean_df + k_val
            mean_df/=len(df_dict)

            df_dict['mean'] = mean_df

            if plotsetting.save_combined_data:
                fname = f"{self.log_name} + mean_df.csv"
                mean_df.to_csv(fname,index=False)
                exit()
                
            # individual plots
            for k, dfv in df_dict.items():
                fig = plt.figure(figsize=plotsetting._plot_size)
                ax = fig.add_subplot(111, projection='3d')
                self.plot_trajectory_3d(dfv, ax, plotsetting)
                
                # Set plot title and labels
                title_name = f"{self.log_name} attempt {k} trajectory"
                if k == 'mean':
                    title_name = f"{self.log_name} mean trajectory"
                
                modify_ax(ax,title_name)

                plt.style.use(plotsetting._plot_style)
                plt.tight_layout()
                fname = f"{title_name}.png"
                fname = self.get_save_name(fname)
                plt.savefig(fname, dpi=plotsetting._dpi)
                plt.show()

            # all combined plot
            fig = plt.figure(figsize=plotsetting._plot_size)
            ax = fig.add_subplot(111, projection='3d')
            legends_list = []
            counter_i = 0
            for k, dfv in df_dict.items():
                # need way too increment color # TODO
                plotsetting.store_in_buffer(plotsetting._traj_color)
                plotsetting._traj_color = plotsetting.color_names[counter_i%len(plotsetting.color_names)]
                self.plot_trajectory_3d(dfv, ax, plotsetting)
                plotsetting._traj_color = plotsetting.swap_buffer
                legends_list += [f"attempt {k}"]
                counter_i+=1

            # Set plot title and labels
            title_name = f"{self.log_name} trajectories"
            modify_ax(ax,title_name,legends_list)

            plt.style.use(plotsetting._plot_style)
            plt.tight_layout()
            fname = self.get_save_name(fname)
            plt.savefig(fname, dpi=plotsetting._dpi)
            plt.show()
        elif grouping_type == 'gen':
            mean_df = df.groupby(grouping_type).mean()
            if plotsetting.save_combined_data:
                fname = f"{self.log_name} + mean_df.csv"
                mean_df.to_csv(fname,index=False)
                exit()

            fig = plt.figure(figsize=plotsetting._plot_size)
            ax = fig.add_subplot(111, projection='3d')
            self.plot_trajectory_3d(mean_df, ax, plotsetting)

            # Set plot title and labels
            title_name = f"{self.log_name} generation mean trajectory"
            modify_ax(ax,title_name)

            plt.style.use(plotsetting._plot_style)
            plt.tight_layout()
            fname = f"{title_name}.png"
            fname = self.get_save_name(fname)
            plt.savefig(fname, dpi=plotsetting._dpi)
            plt.show()
        pass

    def plot_trajectory_2d(self, df, ax, plotsetting):

        # Extract x, y, z coordinates from DataFrame
        x = df[plotsetting._x_axis]
        y = df[plotsetting._y_axis]
        # Plot trajectory line
        ax.plot(x, y, color=plotsetting._traj_color, linewidth=plotsetting._traj_lines_linewidth)
        # Plot arrows for each step in the trajectory
        if plotsetting._use_quiver:
            for i in range(len(x)-1):
                dx = x[i+1] - x[i]
                dy = y[i+1] - y[i]
                ax.quiver(x[i], y[i], dx, dy, color=plotsetting._traj_color, angles='xy', scale_units='xy', scale=plotsetting._2d_traj_scale, width=plotsetting._2d_traj_width)

    def progress_line_plots_2d(self, df, grouping_type,plotsetting):
        def modify_ax(_ax,_title_name, _legend_list=[]):
            if plotsetting._use_xyz_limits:
                limits = self.get_plotting_limits(df, plotsetting)
                _ax.set_xlim((limits['low'][plotsetting._x_axis], limits['high'][plotsetting._x_axis]))
                _ax.set_ylim((limits['low'][plotsetting._y_axis], limits['high'][plotsetting._y_axis]))

            ax.set_title(_title_name, fontsize=plotsetting._traj_font_size)
            ax.set_xlabel(plotsetting._x_axis, fontsize=plotsetting._traj_font_size)
            ax.set_ylabel(plotsetting._y_axis, fontsize=plotsetting._traj_font_size)

            # Customize legend
            if len(_legend_list) > 0:
                _ax.legend(legends_list, fontsize=int(plotsetting._traj_font_size*0.5))

            # Set plot size
            if plotsetting._plot_size:
                fig.set_size_inches(plotsetting._plot_size[0], plotsetting._plot_size[1])

        mean_df = df.groupby(grouping_type).mean()
        df_dict = dict()
        grp = df.groupby(grouping_type)
        for grp_key, grp_df in grp:
            df_dict[grp_key] = grp_df.reset_index().drop('index', axis=1)

        _temp_key = None
        for grp_key, _ in df_dict.items():
            _temp_key = grp_key
            break
        mean_df = copy.deepcopy(df_dict[_temp_key])
        for grp_key, k_val in df_dict.items():
            if grp_key == _temp_key:
                continue
            mean_df = mean_df + k_val
        mean_df/=len(df_dict)
        old_x_axis = plotsetting._x_axis
        old_y_axis = plotsetting._y_axis
        df_dict['mean'] = mean_df
        clst = ['R', '-logL', '-logO']
        for i in range(len(clst)):
            for j in range(i):
                plotsetting._x_axis = clst[j]
                plotsetting._y_axis = clst[i]

                if grouping_type == 'index':
                    # for all the attempts it plots in 1 plot all the paths taken by the point
                    # individual plots
                    for k, dfv in df_dict.items():
                        fig = plt.figure(figsize=plotsetting._plot_size)
                        ax = fig.add_subplot(111)
                        self.plot_trajectory_2d(dfv, ax, plotsetting)

                        title_name =  f"{self.log_name} attempt {k} trajectory {plotsetting._x_axis} {plotsetting._y_axis}"
                        if k == 'mean':
                            title_name = f"{self.log_name} mean trajectory {plotsetting._x_axis} {plotsetting._y_axis}"
                        modify_ax(ax,title_name)

                        plt.style.use(plotsetting._plot_style)
                        plt.tight_layout()
                        fname = f"{title_name}.png"
                        fname = self.get_save_name(fname)
                        plt.savefig(fname, dpi=plotsetting._dpi)
                        plt.show()

                    # all combined plot
                    fig = plt.figure(figsize=plotsetting._plot_size)
                    ax = fig.add_subplot(111)
                    legends_list = []
                    counter_i = 0
                    for k, dfv in df_dict.items():
                        # need way too increment color # TODO
                        plotsetting.store_in_buffer(plotsetting._traj_color)
                        plotsetting._traj_color = plotsetting.color_names[counter_i%len(plotsetting.color_names)]
                        self.plot_trajectory_2d(dfv, ax, plotsetting)
                        plotsetting._traj_color = plotsetting.swap_buffer
                        legends_list += [f"attempt {k}"]
                        counter_i+=1

                    title_name =  f"{self.log_name} trajectories {plotsetting._x_axis} {plotsetting._y_axis}"
                    modify_ax(ax,title_name,legends_list)

                    plt.style.use(plotsetting._plot_style)
                    plt.tight_layout()
                    fname = f"{title_name}.png"
                    fname = self.get_save_name(fname)
                    plt.savefig(fname, dpi=plotsetting._dpi)
                    plt.show()
                elif grouping_type == 'gen':
                    fig = plt.figure(figsize=plotsetting._plot_size)
                    ax = fig.add_subplot(111)
                    self.plot_trajectory_2d(mean_df, ax, plotsetting)

                    title_name =  f"{self.log_name} generation mean trajectory {plotsetting._x_axis} {plotsetting._y_axis} "
                    modify_ax(ax,title_name)

                    plt.style.use(plotsetting._plot_style)
                    plt.tight_layout()
                    fname = f"{title_name}.png"
                    fname = self.get_save_name(fname)
                    plt.savefig(fname, dpi=plotsetting._dpi)
                    plt.show()
        plotsetting._x_axis = old_x_axis
        plotsetting._y_axis = old_y_axis
        return


    def population_gif_and_plots(self, df,plotsetting):
        # plot grad gif and individual image for the ppl forall 2 d pairs
        # plot point gif and individual image for the ppl forall 2 d pairs
        clst = ['R', '-logL', '-logO']
        old_x_axis = plotsetting._x_axis
        old_y_axis = plotsetting._y_axis
        old_z_axis = plotsetting._z_axis
        for i in range(len(clst)):
            for j in range(i):
                plotsetting._x_axis = clst[j]
                plotsetting._y_axis = clst[i]
                plotsetting._z_axis = clst[(6-i-j)%3]
                self.grad_plot_and_gif(df,plotsetting)
                self.point_plot_and_gif(df,plotsetting)
        plotsetting._x_axis = old_x_axis
        plotsetting._y_axis = old_y_axis
        plotsetting._z_axis = old_z_axis
        return

    def point_plot_and_gif(self, df,plotsetting):

        plt.style.use(plotsetting._plot_style)
        gif.options.matplotlib["dpi"] = plotsetting._dpi
        limits = None
        if plotsetting._use_xyz_limits:
            limits = self.get_plotting_limits(df, plotsetting)

        def _save_plot(title):
            fname = title + f"-{plotsetting._x_axis}-{plotsetting._y_axis}-point"+ ".png"
            fname = self.get_save_name(fname)
            plt.savefig(fname, dpi = plotsetting._dpi)
            plt.show()

        @gif.frame
        def plot(gen, x, y,to_save_plot=False):
            fig, ax = plt.subplots()
            ax.scatter(x=x, y=y, color='red')
            if plotsetting._use_xyz_limits:
                ax.set_xlim((limits['low'][plotsetting._x_axis], limits['high'][plotsetting._x_axis]))
                ax.set_ylim((limits['low'][plotsetting._y_axis], limits['high'][plotsetting._y_axis]))
            ax.set_xlabel(plotsetting._x_axis)
            ax.set_ylabel(plotsetting._y_axis)
            ax.set_yscale(plotsetting._y_scale)
            ax.set_xscale(plotsetting._x_scale)
            title = f"{self.log_name} Generation : {gen}"
            ax.set_title(title)
            if to_save_plot:
                _save_plot(title)

        frames = []
        grp = df.groupby('gen')
        for grp_key,grp_df in grp:
            df_grp_key = grp_df.reset_index().drop('index', axis=1)
            _x = df_grp_key[plotsetting._x_axis]
            _y = df_grp_key[plotsetting._y_axis]
            frames += [plot(grp_key, _x, _y)]
            plot(grp_key,_x,_y,to_save_plot=True)

        fname = f"{self.log_name}-{plotsetting._x_axis}-{plotsetting._y_axis}-point" + ".gif"
        fname = self.get_save_name(fname)
        gif.save(frames, fname, duration=plotsetting._duration)
        pass

    def grad_plot_and_gif(self, df, plotsetting:PlotSettings):

        plt.style.use(plotsetting._plot_style)
        gif.options.matplotlib["dpi"] = plotsetting._dpi
        limits = None
        if plotsetting._use_xyz_limits:
            limits = self.get_plotting_limits(df, plotsetting)

        df_list = []
        grp = df.groupby('gen')
        for grp_key, grp_df in grp:
            df_list.append(grp_df.reset_index().drop('index', axis=1))
        gen_max = len(df_list)
        points_count = len(df_list[0])
        color_of_point = []
        for i in range(points_count):
            # give different color for each point
            color_of_point += plotsetting.color_names_proper[i%len(plotsetting.color_names_proper)]

        def _save_plot(title):
            fname = title + f"-{plotsetting._x_axis}-{plotsetting._y_axis}-gard"+ ".png"
            fname = self.get_save_name(fname)
            plt.savefig(fname, dpi = plotsetting._dpi)
            plt.show()

        @gif.frame
        def plot(gen,to_save_plot=False):
            fig, ax = plt.subplots()
            list_point_xs = []
            list_point_ys = []
            metric = []
            for point in range(points_count):
                list_point_xs += [[df_list[gg][plotsetting._x_axis][point]for gg in range(gen-1, gen+1)]]
                list_point_ys += [[df_list[gg][plotsetting._y_axis][point] for gg in range(gen-1, gen+1)]]
                metric += [((list_point_xs[-1][1]-list_point_xs[-1][0])**2+(list_point_ys[-1][1]-list_point_ys[-1][0])**2)**0.5]
            q25q75 = pd.DataFrame(metric).quantile([0.25,0.75])

            for p in range(points_count):
                if metric[p] > 2.5*q25q75[0][0.75]-1.5*q25q75[0][0.25]:
                    continue
                ax.plot(list_point_xs[p], list_point_ys[p],color=color_of_point[p])
                if plotsetting._use_xyz_limits:
                    ax.set_xlim((limits['low'][plotsetting._x_axis],limits['high'][plotsetting._x_axis]))
                    ax.set_ylim((limits['low'][plotsetting._y_axis],limits['high'][plotsetting._y_axis]))
            ax.set_xlabel(plotsetting._x_axis)
            ax.set_ylabel(plotsetting._y_axis)
            ax.set_yscale(plotsetting._y_scale)
            ax.set_xscale(plotsetting._x_scale)
            title = f"{self.log_name} motion of solutions : {gen}"
            ax.set_title(title)
            if to_save_plot:
                _save_plot(title)

        frames = []
        for gen_i in range(1, gen_max-1):
            frames += [plot(gen_i)]
        
        for gen_i in range(1, gen_max-1):
            plot(gen_i,to_save_plot=True)

        fname = f"{self.log_name}-{plotsetting._x_axis}-{plotsetting._y_axis}-grad" + ".gif"
        fname = self.get_save_name(fname)
        gif.save(frames, fname, duration=plotsetting._duration)
        pass

    def combined_progress_plot(self, df_dict, plotsetting:PlotSettings):
        # all combined plot
        fig = plt.figure(figsize=plotsetting._plot_size)
        ax = fig.add_subplot(111, projection='3d')
        legends_list = []
        counter_i = 0
        for k, dfv in df_dict.items():
            plotsetting.store_in_buffer(plotsetting._traj_color)
            plotsetting._traj_color = plotsetting.color_names[counter_i%len(plotsetting.color_names)]
            self.plot_trajectory_3d(dfv, ax, plotsetting)
            plotsetting._traj_color = plotsetting.swap_buffer
            legends_list += [f"{k}"]
            counter_i+=1

        # Set plot title and labels
        title_name = f"All Methods Trajectories"

        ax.set_title(title_name, fontsize=plotsetting._traj_font_size)
        ax.set_xlabel(plotsetting._x_axis, fontsize=plotsetting._traj_font_size)
        ax.set_ylabel(plotsetting._y_axis, fontsize=plotsetting._traj_font_size)
        ax.set_zlabel(plotsetting._z_axis, fontsize=plotsetting._traj_font_size)

        # Customize legend
        if len(legends_list) > 0:
            ax.legend(legends_list, fontsize=int(plotsetting._traj_font_size*0.5))

        # customize the viewing angle
        ax.view_init(elev=plotsetting._3d_plot_elev, azim=plotsetting._3d_plot_azim)

        # Set plot size
        if plotsetting._plot_size:
            fig.set_size_inches(plotsetting._plot_size[0], plotsetting._plot_size[1])

        plt.style.use(plotsetting._plot_style)
        fname = title_name+".png"
        fname = self.get_save_name(fname)
        plt.savefig(fname, dpi=plotsetting._dpi)
        plt.show()

        clst = ['R', '-logL', '-logO']
        old_x_axis = plotsetting._x_axis
        old_y_axis = plotsetting._y_axis
        old_z_axis = plotsetting._z_axis
        for i in range(len(clst)):
            for j in range(i):
                plotsetting._x_axis = clst[j]
                plotsetting._y_axis = clst[i]
                plotsetting._z_axis = clst[(6-i-j)%3]

                # all combined plot
                fig = plt.figure(figsize=plotsetting._plot_size)
                ax = fig.add_subplot(111)
                legends_list = []
                counter_i = 0
                for k, dfv in df_dict.items():
                    plotsetting.store_in_buffer(plotsetting._traj_color)
                    plotsetting._traj_color = plotsetting.color_names[counter_i%len(plotsetting.color_names)]
                    self.plot_trajectory_2d(dfv, ax, plotsetting)
                    plotsetting._traj_color = plotsetting.swap_buffer
                    legends_list += [f"{k}"]
                    counter_i+=1

                title_name =  f"All Methods Trajectories {plotsetting._x_axis} {plotsetting._y_axis}"


                ax.set_title(title_name, fontsize=plotsetting._traj_font_size)
                ax.set_xlabel(plotsetting._x_axis, fontsize=plotsetting._traj_font_size)
                ax.set_ylabel(plotsetting._y_axis, fontsize=plotsetting._traj_font_size)

                # Customize legend
                if len(legends_list) > 0:
                    ax.legend(legends_list, fontsize=int(plotsetting._traj_font_size*0.5))

                # Set plot size
                if plotsetting._plot_size:
                    fig.set_size_inches(plotsetting._plot_size[0], plotsetting._plot_size[1])

                plt.style.use(plotsetting._plot_style)
                plt.tight_layout()
                fname = f"{title_name}.png"
                fname = self.get_save_name(fname)
                plt.savefig(fname, dpi=plotsetting._dpi)
                plt.show()
        plotsetting._x_axis = old_x_axis
        plotsetting._y_axis = old_y_axis
        plotsetting._z_axis = old_z_axis
