#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 14:19:57 2022

@author: frederikmelson
"""

import math 
from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint
from pyomo.environ import *
from pyomo.contrib.sensitivity_toolbox.sens import sensitivity_calculation
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
# from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
# import pyomo.contrib.preprocessing.plugins.induced_linearity as induced_linearity
import time
import pyomo.contrib.appsi.solvers.ipopt as ipo
startTime = time.time()
plt.style.use("seaborn")

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# import seaborn as sns
# import kaleido
# from IPython.display import Image
pio.renderers.default='svg'


years = [2020,2025,2030,2035,2040,2045,2050]

techs_file          = "techs.pkl"
fossil_techs_file   = "fossil_techs.pkl"
renewables_file     = "renewables.pkl"
wind_file           = "wind.pkl"
colors_file         = "colors.pkl"
storage_file        = "storage.pkl"
color_storage_file  = "color_storage.pkl"
CC_file             = "CC.pkl"

files = [techs_file,fossil_techs_file,renewables_file,wind_file,colors_file,storage_file,color_storage_file,CC_file]
lists = ["techs","fossil_techs","renewables","wind","colors","storage","color_storage","CC_techs"]


for i in range(len(files)):
    open_file = open(files[i], "rb")
    lists[i] = pickle.load(open_file)
    open_file.close()


techs           = lists[0]
fossil_techs    = lists[1]
renewables      = lists[2]
wind            = lists[3]
colors          = lists[4]
storage         = lists[5]
color_storage   = lists[6]
CC_techs        = lists[7]

level_costfloor = pd.read_excel("Results/Exp3/LCOE_Grexp3_co2_constraint-LR-nomLR.xlsx",index_col=0)
fixed_costfloor = pd.read_excel("Results/Exp3/annuity_costs_Grexp3_co2_constraint-LR-nomLR.xlsx",index_col=0)

level_costnofloor = pd.read_excel("Results/Exp3/LCOE_Grco2_2050co2_constraint-LR-nomLR.xlsx",index_col=0)
fixed_costnofloor = pd.read_excel("Results/Exp3/annuity_costs_Grco2_2050co2_constraint-LR-nomLR.xlsx",index_col=0)

capcostfloor = pd.read_excel("Results/Exp3/capital_costs_Grexp3_co2_constraint-LR-nomLR.xlsx",index_col=0)
capcostnofloor = pd.read_excel("Results/Exp3/capital_costs_Grco2_2050co2_constraint-LR-nomLR.xlsx",index_col=0)

fig, ax = plt.subplots(figsize = [12, 6], dpi = 200)
level_costfloor.plot(kind="line",ax=ax,linewidth=3,color=colors,label='Storage discharged',linestyle='--')#cmap=colormap)
level_costnofloor.plot(kind="line",ax=ax,linewidth=3,color=colors,label='Storage charged')#cmap=colormap)
ax.set_xlabel("Year")
ax.set_ylabel("LCOE [EUR/MWh]")
# plt.title(str(plotyear))
# ax.set_yscale("log")
ax.set_ylim([10,150])
# plt.xticks(ticks=[0, 32/4, 2*32/4, 3*32/4],labels=['1','2','3','4'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1.0, 1.0), ncol=1, fancybox=False, shadow=False)
textstr = '\n'.join((
    r'Dotted line = With floor',
    r'Solid line = Without floor'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.80, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
fig.savefig("Results/Exp3/figures/LCOE_comp_exp3.png",transparent=True,dpi=200,bbox_inches='tight')

fig, ax = plt.subplots(figsize = [12, 6], dpi = 200)
fixed_costfloor.plot(kind="line",ax=ax,linewidth=3,color=colors,label='Storage discharged',linestyle='--')#cmap=colormap)
fixed_costnofloor.plot(kind="line",ax=ax,linewidth=3,color=colors,label='Storage charged')#cmap=colormap)
ax.set_xlabel("Year")
ax.set_ylabel("Annuity [EUR/MW/a]")
# plt.title(str(plotyear))
# ax.set_yscale("log")
ax.set_ylim([10,80])
# plt.xticks(ticks=[0, 32/4, 2*32/4, 3*32/4],labels=['1','2','3','4'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1.0, 1.0), ncol=1, fancybox=False, shadow=False)
textstr = '\n'.join((
    r'Dotted line = With floor',
    r'Solid line = Without floor'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.80, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
fig.savefig("Results/Exp3/figures/annuity_comp_exp3.png",transparent=True,dpi=200,bbox_inches='tight')


fig, ax = plt.subplots(figsize = [12, 6], dpi = 200)
capcostfloor.plot(kind="line",ax=ax,linewidth=3,color=colors,linestyle='--')#cmap=colormap)
capcostnofloor.plot(kind="line",ax=ax,linewidth=3,color=colors)#cmap=colormap)
ax.set_xlabel("Year")
ax.set_ylabel("Capital cost [EUR/MW]")
# plt.title(str(plotyear))
# ax.set_yscale("log")
ax.set_ylim([100000,2000000])
# plt.xticks(ticks=[0, 32/4, 2*32/4, 3*32/4],labels=['1','2','3','4'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1.0, 1.0), ncol=1, fancybox=False, shadow=False)
textstr = '\n'.join((
    r'Dotted line = With floor',
    r'Solid line = Without floor'))
textstr2 = '\n'.join((
    r'Nuclear ~ 7940000',
    r'Coal ~3845000'))
ax.ticklabel_format(axis='both', style='plain')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.80, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.text(0.40, 0.95, textstr2, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
fig.savefig("Results/Exp3/figures/capcost_comp_exp3.png",transparent=True,dpi=200,bbox_inches='tight')