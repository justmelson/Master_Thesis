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

r01e = pd.read_excel("Results/Exp5/Comparison/Exp5_0.01direct_airrevCCco2_constraint-LR-nomLRemissions_allyears.xlsx",index_col=0)


fig = px.line(fixed_cost_all, x="Year", y="solar_PV",color="LR", line_group="LR", hover_name="LR",
         render_mode="svg")
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        gridwidth=0.5,
    ),
    yaxis=dict(
        title='Capital cost of PV [EUR2022/MW/a]',
        gridcolor='grey',
        # type='log',
        gridwidth=0.5,
        
    ),)
fig.update_yaxes(tickcolor='crimson', ticklen=10)
fig.show()
fig.write_image("Results/Exp2/solarLR_capcost_log.png",width=800, height=500,scale=1.5)
