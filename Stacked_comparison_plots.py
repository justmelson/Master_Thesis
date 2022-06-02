#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 11:04:59 2022

@author: frederikmelson
"""

import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size': 22})

open_file = open("colors.pkl", "rb")
colors = pickle.load(open_file)
open_file.close()

colors = ["dodgerblue","lightgreen","gold", 'coral',"peru","grey","plum","brown",'darkkhaki']
colors_disp = ["dodgerblue","lightgreen","gold", 'coral',"peru","grey","plum","brown"]

cmap = ListedColormap(colors)

#%%

def plot_clustered_stacked(dfall, labels=None, title="Installed capacites",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      figsize = [12, 6],

                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel("GW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.savefig('Results/Exp1/Comparison_figures/capacities_stacked_LR_comp.png', dpi=200,bbox_inches='tight',transparent=True)
    return axe


NoLR    = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-no_LRcapacities_2050CO2.pkl")
LowLR   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-lowLRcapacities_2050CO2.pkl")
NomLR   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-nomLRcapacities_2050CO2.pkl")
HighLR  = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-highLRcapacities_2050CO2.pkl")

plot_clustered_stacked([NoLR,LowLR, NomLR, HighLR],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)


def plot_buildyears_stacked(dfall, labels=None, title="New capacites built",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe2 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe2 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe2,
                      legend=False,
                      figsize = [12, 6],
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe2.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe2.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe2.set_xticklabels(df.index, rotation = 0)
    axe2.set_title(title)
    axe2.set_ylabel("GW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe2.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe2.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe2.add_artist(l1)
    plt.savefig('Results/Exp1/Comparison_figures/built_capacities_stacked_LR_comp.png', dpi=200,bbox_inches='tight',transparent=True)
    return axe2


NoLR_build    = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-no_LRbuild_years_2050CO2.pkl")
LowLR_build   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-lowLRbuild_years_2050CO2.pkl")
NomLR_build   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-nomLRbuild_years_2050CO2.pkl")
HighLR_build   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-highLRbuild_years_2050CO2.pkl")

plot_buildyears_stacked([NoLR_build,LowLR_build, NomLR_build, HighLR_build],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)



def plot_dispatch_stacked(dfall, labels=None, title="Dispatched power over 4 days",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe3 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe3 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe3,
                      legend=False,
                      figsize = [12, 6],
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe3.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe3.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe3.set_xticklabels(df.index, rotation = 0)
    axe3.set_title(title)
    axe3.set_ylabel("GWh")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe3.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe3.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe3.add_artist(l1)
    plt.savefig('Results/Exp1/Comparison_figures/dispatched_power_stacked_LR_comp.png', 
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    return axe3

cmap = ListedColormap(colors_disp)
a = 0
b=8
NoLR_dispatch    = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-no_LRdispatch_sum_2050CO2.pkl").iloc[a:b]
LowLR_dispatch   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-lowLRdispatch_sum_2050CO2.pkl").iloc[a:b]
NomLR_dispatch   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-nomLRdispatch_sum_2050CO2.pkl").iloc[a:b]
HighLR_dispatch  = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-highLRdispatch_sum_2050CO2.pkl").iloc[a:b]


plot_dispatch_stacked([NoLR_dispatch,LowLR_dispatch, NomLR_dispatch, HighLR_dispatch],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)

def plot_PV_stacked(dfall, labels=None, title="Capital cost for PV",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = 1#len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe4 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe4 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe4,
                      legend=False,
                      grid=True,
                      figsize = [12, 4],
                      **kwargs)  # make bar plots

    h,l = axe4.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe4.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe4.set_xticklabels(df.index, rotation = 0)
    axe4.set_title(title)
    axe4.set_ylabel("2022\N{euro sign}/MW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe4.bar(0, 0, color="gray", hatch=H * i))

    # l1 = axe4.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    # axe4.add_artist(l1)
    plt.savefig('Results/Exp1/Comparison_figures/capex_PV_comp.pdf', 
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    return axe4

a = 0
b=8
NoLR_capex_pv   = pd.read_pickle("Results/Exp1/capex_pv_Grco2_2050co2_constraint-no_LR.pkl").iloc[a:b]
LowLR_capex_pv   = pd.read_pickle("Results/Exp1/capex_pv_Grco2_2050co2_constraint-LR-lowLR.pkl").iloc[a:b]
NomLR_capex_pv   = pd.read_pickle("Results/Exp1/capex_pv_Grco2_2050co2_constraint-LR-nomLR.pkl").iloc[a:b]
HighLR_capex_pv  = pd.read_pickle("Results/Exp1/capex_pv_Grco2_2050co2_constraint-LR-highLR.pkl").iloc[a:b]



cmap = ListedColormap("gold")
plot_PV_stacked([NoLR_capex_pv,LowLR_capex_pv , NomLR_capex_pv, HighLR_capex_pv ],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)


def plot_onwind_stacked(dfall, labels=None, title="Capital cost for Onshore wind",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""
    n_df = len(dfall)
    n_col = 1#len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe5 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe5 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe5,
                      legend=False,
                      grid=True,
                      figsize = [12, 4],
                      **kwargs)  # make bar plots

    h,l = axe5.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe5.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe5.set_xticklabels(df.index, rotation = 0)
    axe5.set_title(title)
    axe5.set_ylabel("2022\N{euro sign}/MW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe5.bar(0, 0, color="gray", hatch=H * i))

    # l1 = axe5.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    # axe5.add_artist(l1)
    plt.savefig('Results/Exp1/Comparison_figures/capex_onshore_comp.pdf', 
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    plt.rcParams.update({'font.size': 22})

    return axe5

a = 0
b=8
NoLR_capex_onwind   = pd.read_pickle("Results/Exp1/capex_onwindGrco2_2050co2_constraint-no_LR.pkl").iloc[a:b]
LowLR_capex_onwind   = pd.read_pickle("Results/Exp1/capex_onwindGrco2_2050co2_constraint-LR-lowLR.pkl").iloc[a:b]
NomLR_capex_onwind   = pd.read_pickle("Results/Exp1/capex_onwindGrco2_2050co2_constraint-LR-nomLR.pkl").iloc[a:b]
HighLR_capex_onwind  = pd.read_pickle("Results/Exp1/capex_onwindGrco2_2050co2_constraint-LR-highLR.pkl").iloc[a:b]

cmap = ListedColormap("lightgreen")
plot_onwind_stacked([NoLR_capex_onwind,LowLR_capex_onwind , NomLR_capex_onwind, HighLR_capex_onwind ],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)


def plot_offwind_stacked(dfall, labels=None, title="Capital cost for Offshore wind",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = 1#len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe5 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe5 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe5,
                      legend=False,
                      grid=True,
                      figsize = [12, 4],
                      **kwargs)  # make bar plots

    h,l = axe5.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe5.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe5.set_xticklabels(df.index, rotation = 0)
    axe5.set_title(title)
    axe5.set_ylabel("2022\N{euro sign}/MW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe5.bar(0, 0, color="gray", hatch=H * i))

    # l1 = axe5.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    # axe5.add_artist(l1)

    plt.savefig('Results/Exp1/Comparison_figures/capex_offshore_comp.png', 
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    return axe5

a = 0
b=8
NoLR_capex_offwind   = pd.read_pickle("Results/Exp1/capex_offwindGrco2_2050co2_constraint-no_LR.pkl").iloc[a:b]
LowLR_capex_offwind   = pd.read_pickle("Results/Exp1/capex_offwindGrco2_2050co2_constraint-LR-lowLR.pkl").iloc[a:b]
NomLR_capex_offwind   = pd.read_pickle("Results/Exp1/capex_offwindGrco2_2050co2_constraint-LR-nomLR.pkl").iloc[a:b]
HighLR_capex_offwind  = pd.read_pickle("Results/Exp1/capex_offwindGrco2_2050co2_constraint-LR-highLR.pkl").iloc[a:b]


cmap = ListedColormap("dodgerblue")
plot_offwind_stacked([NoLR_capex_offwind,LowLR_capex_offwind , NomLR_capex_offwind, HighLR_capex_offwind ],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)


def plot_capex_stacked(dfall, labels=None, title="Capital cost",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe3 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe3 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=False,
                      ax=axe3,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe3.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe3.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe3.set_xticklabels(df.index, rotation = 0)
    axe3.set_title(title)
    axe3.set_ylabel("GWh")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe3.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe3.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe3.add_artist(l1)
    plt.savefig('Results/Exp1/Comparison_figures/dispatched_power_stacked_LR_comp.png', 
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    return axe3
cmap = ListedColormap(["lightgreen","gold"])

NoLR_capex = pd.concat([NoLR_capex_onwind,NoLR_capex_pv],axis=1)
LowLR_capex = pd.concat([LowLR_capex_onwind,LowLR_capex_pv],axis=1)
NomLR_capex =pd.concat([NomLR_capex_onwind,NomLR_capex_pv],axis=1)
HighLR_capex= pd.concat([HighLR_capex_onwind,HighLR_capex_pv],axis=1)

# plot_capex_stacked([NoLR_capex,LowLR_capex , NomLR_capex, HighLR_capex],["NoLR","LowLR", "NomLR", "HighLR"],cmap=cmap)


#%% Exp3


def plot_clustered_stacked(dfall, labels=None, title="Installed capacites",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      figsize = [12, 6],

                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel("GW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.savefig('Results/Exp3/figures/capacities_stacked_floorcost_comp.png', dpi=200,bbox_inches='tight',transparent=True)
    return axe

a = 0
b=8
No_floor   = pd.read_pickle("Results/Exp1/Grco2_2050co2_constraint-LR-nomLRcapacities_2050CO2.pkl")
With_floor  = pd.read_pickle("Results/Exp3/Grexp3_co2_constraint-LR-nomLRcapacities_2050CO2.pkl")

plot_clustered_stacked([No_floor,With_floor],["No_floor","With_floor"],cmap=cmap)

#%% Exp5
indx = [2020,2025,2030,2035,2040,2045,2050,2055]

def plot_clustered_stackedExp5(dfall, labels=None, title="Installed capacites",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)+1
    axe = plt.subplot(111)
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      figsize = [12, 6],

                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2.2 * n_ind, 2.2) + 1 / float(n_df + 2.2)) / 2.2)
    axe.set_xticklabels(indx, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel("GW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.savefig('Results/Exp5/figures/capacities_Exp5.png', dpi=200,bbox_inches='tight',transparent=True)
    return axe


R1   = pd.read_pickle("Results/Exp5/Grdiscount0.01no_co2-LR-nomLRcapacities_2050CO2.pkl")
R2  = pd.read_pickle("Results/Exp5/Grdiscount0.02no_co2-LR-nomLRcapacities_2050CO2.pkl")
R3  = pd.read_pickle("Results/Exp5/Grdiscount0.03no_co2-LR-nomLRcapacities_2050CO2.pkl")
R4  = pd.read_pickle("Results/Exp5/Grdiscount0.04no_co2-LR-nomLRcapacities_2050CO2.pkl")
R5  = pd.read_pickle("Results/Exp5/Grdiscount0.05no_co2-LR-nomLRcapacities_2050CO2.pkl")
R6  = pd.read_pickle("Results/Exp5/Grdiscount0.06no_co2-LR-nomLRcapacities_2050CO2.pkl")
R7  = pd.read_pickle("Results/Exp5/Grdiscount0.07no_co2-LR-nomLRcapacities_2050CO2.pkl")



plot_clustered_stackedExp5([R1,R2,R3,R4,R5,R6,R7],['R1','R2','R3','R4','R5','R6','R7'],cmap=cmap)


def plot_dispatch_Exp5(dfall, labels=None, title="Dispatched power",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)+1
    axe = plt.subplot(111)
    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      figsize = [12, 6],

                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2.2 * n_ind, 2.2) + 1 / float(n_df + 2.2)) / 2.2)
    axe.set_xticklabels(indx, rotation = 0)
    axe.set_title(title)
    axe.set_ylabel("GW")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    plt.savefig('Results/Exp5/figures/Dispatched_power_Exp5.png', dpi=200,bbox_inches='tight',transparent=True)
    return axe


R1   = pd.read_pickle("Results/Exp5/Grdiscount0.01no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")
R2  = pd.read_pickle("Results/Exp5/Grdiscount0.02no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")
R3  = pd.read_pickle("Results/Exp5/Grdiscount0.03no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")
R4  = pd.read_pickle("Results/Exp5/Grdiscount0.04no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")
R5  = pd.read_pickle("Results/Exp5/Grdiscount0.05no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")
R6  = pd.read_pickle("Results/Exp5/Grdiscount0.06no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")
R7  = pd.read_pickle("Results/Exp5/Grdiscount0.07no_co2-LR-nomLRdispatch_allyears_2050CO2.pkl")



plot_clustered_stackedExp5([R1,R2,R3,R4,R5,R6,R7],['R1','R2','R3','R4','R5','R6','R7'],cmap=cmap)
#%% System cost
# scenarios = ["NoLR","LowLR","NomLR","HighLR"]
# cost_noLR = pd.DataFrame(1945,
#                    index=["NoLR"],
#                    columns=["System cost"])

# cost_lowLR = pd.DataFrame(1777,
#                    index=["LowLR"],
#                    columns=["System cost"])

# cost_nomLR = pd.DataFrame(1586,
#                    index=["NomLR"],
#                    columns=["System cost"])

# cost_highLR = pd.DataFrame(1277,
#                    index=["HighLR"],
#                    columns=["System cost"])


# costs=[1945,1777,1586,1277]

# rel_cost = np.ones(4)
# for i in range(len(costs)):
#     rel_cost[i] = costs[i]/costs[0] *100

# def valuelabel(scenario,cost):
#     for i in range(len(scenario)):
#         plt.text(i,cost[i],cost[i], ha = 'center',
#                  bbox = dict(facecolor = 'cyan', alpha =0.8))
        
        
# y_pos = np.arange(len(scenarios))

# plt.bar(y_pos,rel_cost, align='center')
# valuelabel(scenarios,rel_cost)       

# plt.xticks(y_pos, scenarios)
# plt.ylabel('Relative cost')
# plt.title('Relative system cost')
# plt.bar_label(label_type='center')


# def plot_cost(dfall, labels=None, title="System cost",  H="/", **kwargs):
#     """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
# labels is a list of the names of the dataframe, used for the legend
# title is a string for the title of the plot
# H is the hatch used for identification of the different dataframe"""

#     n_df = len(dfall)
#     n_col = len(dfall[0].columns) 
#     n_ind = len(dfall[0].index)
#     axe = plt.subplot(111)
#     for df in dfall : # for each data frame
#         axe = df.plot(kind="bar",
#                       linewidth=0,
#                       stacked=True,
#                       ax=axe,
#                       legend=False,
#                       grid=False,
#                       **kwargs)  # make bar plots

#     h,l = axe.get_legend_handles_labels() # get the handles we want to modify
#     for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
#         for j, pa in enumerate(h[i:i+n_col]):
#             for rect in pa.patches: # for each index
#                 rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
#                 rect.set_hatch(H * int(i / n_col)) #edited part     
#                 rect.set_width(1 / float(n_df + 1))

#     axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
#     axe.set_xticklabels(df.index, rotation = 0)
#     axe.set_title(title)
#     axe.set_ylabel("Cost in billion EUR")
#     axe.set_xlabel("Scenarios")

#     # Add invisible data to add another legend
#     n=[]        
#     for i in range(n_df):
#         n.append(axe.bar(0, 0, color="gray", hatch=H * i))

#     l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
#     if labels is not None:
#         l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
#     axe.add_artist(l1)
#     plt.savefig('systemcost_LR_comp.png', dpi=400,bbox_inches='tight')
#     return axe

# plot_cost([cost_noLR,cost_lowLR, cost_nomLR, cost_highLR],["NoLR","LowLR", "NomLR", "HighLR"])


# systemcost.plot(kind="bar")
# fig, ax = plt.subplots()
# fig.set_dpi((400))
# systemcost.plot(kind="bar")
# fig.tight_layout()
# ax.set_ylabel("Cost in billion EUR")

