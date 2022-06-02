#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:36:01 2022

@author: frederikmelson
"""



#%% Packages
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

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
# import seaborn as sns
# import kaleido
# from IPython.display import Image
pio.renderers.default='svg'

startTime = time.time()
plt.style.use("seaborn")


#%% Scenarios and settings


# scenario = "no_co2-no_LR"
# scenario = "co2_constraint-no_LR"
scenario = "co2_constraint-LR" 
# scenario = "no_co2-LR"

# learning_scenarios = ["highLR","nomLR","lowLR"]

# learning rate assumptions
learning_scenario = "highLR"
# learning_scenario = "lowLR"
# learning_scenario = "nomLR"


# CO2 budget for 2050 global warming goals
co2_until_2050 = 10500 # 10.5 gigaton CO2

co2_budget = {2020:2500,2025:2500,2030:2,2035:1500,2040:1000,2045:1000,2050:0}

# Greenfield scenario 
Greenfield = True

# Years
years = [2020,2025,2030,2035,2040,2045,2050]
interval = years[1]-years[0]
hour_interval = 3

# legend on/off when plotting
lgnd = True

r = 0.01 # discount rate

hours = list(range(32))
 
dty = 365/(len(hours)/8) # Number of days modeled op to 365 days


parameters  = pd.read_pickle("parameters.pkl")
store_param = pd.read_pickle("store_param.pkl")
CC_param    = pd.read_pickle("CC_param.pkl")

a_file = open("demand_elec5y.pkl", "rb")
demand = pickle.load(a_file) #GWh
PV_learning = [0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.45]
iterations = [0,1,2,3,4,5,6,7]


# CF_solar_one    = pd.read_pickle("CF_solar_one.pkl")
# CF_onwind_one   = pd.read_pickle("CF_onwind_one.pkl")
# CF_offwind_one  = pd.read_pickle("CF_offwind_one.pkl")

# Cf_solar    = pd.read_pickle("Cf_solar.pkl")
# Cf_onshore  = pd.read_pickle("Cf_onshore.pkl")
# Cf_offshore = pd.read_pickle("Cf_offshore.pkl")


Cf_solar      = pd.read_pickle("cf_solar3h4d.pkl")
Cf_onshore    = pd.read_pickle("cf_onshore3h4d.pkl")
Cf_offshore   = pd.read_pickle("cf_offshore3h4d.pkl")
# demand6d3h      = pd.read_pickle("demand6d3h.pkl")
# demand6d3h.reset_index()




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

storage = ['battery_store']
storage_techs = ['battery_store','battery_inverter']
greenfield_df = {}
tech_costs =  ["offshore_wind","onshore_wind","CCGT","OCGT","coal","nuclear"]
fossil_techs = ["CCGT","OCGT","coal"]

# colors = ["dodgerblue","lightgreen", 'coral',"peru","grey","plum","brown","darkkhaki"]
results_data = {}
system_costs={}
build_years_data = {}
capacities_data = {}
annuity_cost = {}
capcost = {}
dispatch_data={}
dispatch_4days={}
# Green- or brownfield scenario:
for tech in techs:
    greenfield_df[tech] = 1/parameters.at["existing capacity",tech]
for tech in storage_techs:
    greenfield_df[tech] = 1/2


if Greenfield is True:
    for tech in techs:
        parameters.loc["existing age"]      = [0,0,0,0,0,0,0] #years
        parameters.loc["existing capacity"] = [0,0,0,0,0,0,0]  
    print("Greenfield approach")
else:
    for tech in techs:
        greenfield_df[tech] = 1
    for tech in storage:
        greenfield_df[tech] = 1
    print("Brownfield approach")



for tech in techs:
    parameters.at["specific emissions",tech] *= 1e-6 # convert to MtCO2/MWh

#%% Updating learning rates and CO2 budget

#Currently installed capacities in GW is used to assume current demand



if "no_LR" in scenario:
    parameters.loc["learning rate"]     = 0
    store_param.loc["learning rate"]    = 0
    print("No learning")
else:
    if "highLR" in learning_scenario:
        parameters.loc["learning rate"]     = [0.19,0.32,0.47,0.34,0.15,0.12,0.0]#[0.12,0.12,0.23,0.14,0.15,0.05,0.06] 
        store_param.loc["learning rate"]    = [0.18,0.1,0.1,0.26,0.21]
        print("High learning rates")
    else: 
        if "lowLR" in learning_scenario:
            parameters.loc["learning rate"]     = [0.05,0.05,0.1,0.05,0.05,0.0,0.0] # [0.05,0,0.1,-0.01,0.15,0.06,-0.06]
            store_param.loc["learning rate"]    = [0.08,0.1,0.1,0.18,0.15]
            print("Low learning rates")
        else:
            # nom learning
            parameters.loc["learning rate"]     = [0.12,0.12,0.23,0.14,0.15,0.083,0.0] # [0.05,0,0.1,-0.01,0.15,0.06,-0.06]
            store_param.loc["learning rate"]    = [0.08,0.1,0.1,0.18,0.15]
            print("Nominal learning rates")


# Calculating learning parameter gamma
for i in range(len(techs)):
    parameters.loc["learning parameter"][i] = math.log(1/(1-parameters.loc["learning rate"][i])) / math.log(2)
for i in range(len(storage)):
    store_param.loc["learning parameter"][i] = math.log(1/(1-store_param.loc["learning rate"][i])) / math.log(2)


# PV_learning = [0.4]
PV_learning_param = list(np.ones(len(PV_learning)))

for i in range(len(PV_learning)):
    PV_learning_param[i] = math.log(1/(1-PV_learning[i])) / math.log(2)



# carbon budget in average tCO2   
if "no_co2" in scenario:
    # co2_budget = 1
    print("No CO2 budget")
else:
    # co2_budget = co2_until_2050 # [tCO2] 10 Gigatons CO2
    print("CO2 budget of "+ str(co2_until_2050) + " MtCO2")

        
    #%% One node model
for iteration in range(len(PV_learning)):
 
    model = ConcreteModel()
    model.generators            = Var(techs, years, within=NonNegativeReals) # bounds=(0.0,1000)
    model.generators_dispatch   = Var(techs, years, hours, within=NonNegativeReals)
    model.generators_built      = Var(techs, years, within=NonNegativeReals)
    model.fixed_costs           = Var(techs, years, within=NonNegativeReals)


    model.SOC                   = Var(storage_techs,years, hours,initialize=0, within=NonNegativeReals)
    model.storage               = Var(storage_techs, years, initialize=0, within=NonNegativeReals)
    model.storage_built         = Var(storage_techs, years, initialize=0, within=NonNegativeReals)
    model.fixed_costs_storage   = Var(storage_techs, years, initialize=0, within=NonNegativeReals)
    model.storage_charge        = Var(storage_techs, years, hours, initialize=0, within=NonNegativeReals)
    model.storage_discharge     = Var(storage_techs, years, hours, initialize=0, within=NonNegativeReals)

    # Starting point for the interior point optimization
    for tech in techs:
        model.generators[tech,years[0]].value = 100
    model.generators["solar_PV",years[0]].value = 100
    model.generators["onshore_wind",years[0]].value = 100


    #Value of currently installed technology:
    constant = sum(parameters.at['existing capacity',tech] * parameters.at['current annuity', tech]*1000/1e9/(1+r)**(hour-hours[0]) for tech in techs for hour in hours if hour < hours[0] + parameters.at['lifetime',tech] - parameters.at['existing age',tech])
    print("Cost of existing capacity =", "%.2f"%constant)
    
    
    model.objective = Objective(expr=constant +
                                sum(model.generators_built[tech,year] * model.fixed_costs[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                                  for tech in techs for year in years)+
                                sum(model.generators_dispatch[tech,year,hour] * parameters.at['marginal cost',tech]*1000*dty*hour_interval*interval / 1e9 /(1+r)**(year-years[0])
                                    for tech in techs
                                    for year in years
                                    for hour in hours)+
                                sum(model.storage_built[tech,year] * model.fixed_costs_storage[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech])))
                                    for tech in storage_techs for year in years))
    
    
    
    #%% Balance Constraints
    
    def balance_constraint(model,tech,year,hour): # GWh
        return demand[year].at[hour,0]  <= sum(model.generators_dispatch[tech, year, hour] for tech in techs) + model.storage_discharge['battery_store',year,hour] - model.storage_charge['battery_store',year,hour]
    model.balance_constraint = Constraint(techs,years,hours, rule=balance_constraint)
    
      
    #%% Solar capacity constraints
    
    def solar_constraint(model,year,hour):
        return model.generators_dispatch["solar_PV",year,hour] <= model.generators["solar_PV",year]*Cf_solar.at[hour,0]
    model.solar_constraint = Constraint(years, hours, rule=solar_constraint)
    
    
    
    #%% Onshore wind capacity constraints
    
    
    def onshore_constraint(model,year,hour):
        return model.generators_dispatch["onshore_wind",year,hour] <= model.generators["onshore_wind",year]*Cf_onshore.at[hour,0]
    model.onshore_constraint = Constraint(years, hours, rule=onshore_constraint)
    
    
    #%% Offshore wind capacity constraints
    
    
    def offshore_constraint(model,year,hour):
        return model.generators_dispatch["offshore_wind",year,hour] <= model.generators["offshore_wind",year]*Cf_offshore.at[hour,0]
    model.offshore_constraint = Constraint(years, hours, rule=offshore_constraint)
    
    
    #%% Storage
    
    
    def storage__charge_constraint(model,tech,year,hour):
        return model.storage_charge['battery_store',year,hour]*model.storage_discharge['battery_store',year,hour] == 0
    model.storage__charge_constraint = Constraint(techs,years,hours,rule=storage__charge_constraint)
    

    def stored_energy_constraint(model,year,hour):
        if hour == 0: 
            return model.SOC['battery_store',year,hour] == 0
        else:
            return model.SOC['battery_store',year,hour] == model.SOC['battery_store',year,hour-1] + model.storage_charge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter'] - model.storage_discharge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter']
    model.stored_energy_constraint = Constraint(years,hours,rule=stored_energy_constraint)

    
    def storage_constraint(model,year,hour):
        return model.storage_discharge['battery_store',year,hour] <= model.SOC['battery_store',year,hour] # GW
    model.storage_constraint = Constraint(years, hours, rule=storage_constraint)
    
    def storage_capacity_constraint(model,year,hour):
        return model.SOC['battery_store',year,hour] <= model.storage['battery_store',year] # GW
    model.storage_capacity_constraint = Constraint(years,hours,rule=storage_capacity_constraint)
    
    def build_years_storage(model,tech,year):
        if year < years[0] + store_param.at["lifetime",tech] - store_param.at["existing age",tech]:
            constant = store_param.at["existing capacity",tech] 
        else:
            constant = 0.
        
        return model.storage[tech,year] == constant + sum(model.storage_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + store_param.at["lifetime",tech]))) #GW
    model.build_years_storage = Constraint(storage_techs,years, rule=build_years_storage)
    
    def inverter_constraint(model,storage_techs,year):
        return model.storage["battery_store",year] <= model.storage['battery_inverter',year]
    model.inverter_constraint = Constraint(storage_techs,years, rule=inverter_constraint)
    
    def storage_bounding_constraint(model,tech,year):
        return sum(model.storage[tech,year] for tech in storage_techs) <= 10000
    model.storage_bounding_constraint = Constraint(storage_techs, years, rule=storage_bounding_constraint)
    
    
    def fixed_cost_constraint_storage(model,tech,year):
        if store_param.at["learning parameter",tech] == 0:
            return model.fixed_costs_storage[tech,year] == store_param.at["current annuity",tech]*1000 #EUR/GW
        else:
            return model.fixed_costs_storage[tech,year] == store_param.at["current annuity",tech]*1000 * (1+sum(model.storage_built[tech,yeart] for yeart in years if yeart < year)*greenfield_df[tech])**(-store_param.at["learning parameter",tech]) #EUR/GW
            # return model.fixed_costs["battery_store",year] == store_param.at["base cost","battery_store"] + (store_param.at["current annuity","battery_store"]-store_param.at["base cost","battery_store"])*(1+sum(model.generators_built["battery_store",yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter","battery_store"])
    model.fixed_cost_constraint_storage = Constraint(storage_techs,years, rule=fixed_cost_constraint_storage)
    
    #%% Installed capacity constraints
    def generator_constraint(model,tech,year,hour):
        return model.generators_dispatch[tech,year,hour] <= model.generators[tech,year] # GW
    model.generator_constraint = Constraint(techs, years, hours, rule=generator_constraint)
    
    
    
    def capacity_constraint(model,tech,year):
        return sum(model.generators[tech,year] for tech in techs) <= 5000
    model.capacity_constraint = Constraint(techs, years, rule=capacity_constraint)
    
    
    def build_years(model,tech,year):
        if year < years[0] + parameters.at["lifetime",tech] - parameters.at["existing age",tech]:
            constant = parameters.at["existing capacity",tech] 
        else:
            constant = 0.
        
        return model.generators[tech,year] == constant + sum(model.generators_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + parameters.at["lifetime",tech]))) #GW
    model.build_years = Constraint(techs, years, rule=build_years)
    
    
    
    def fixed_cost_constraint(model,tech,year):
        if parameters.at["learning parameter",tech] == 0:
            return model.fixed_costs[tech,year] == parameters.at["current annuity",tech]*1000 #EUR/GW
        else:
            return model.fixed_costs[tech,year] == parameters.at["current annuity",tech]*1000 * (1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year)*greenfield_df[tech])**(-parameters.at["learning parameter",tech]) #EUR/GW
            # return model.fixed_costs[tech,year] == parameters.at["base cost",tech] + (parameters.at["current annuity",tech]-parameters.at["base cost",tech])*(1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
    model.fixed_cost_constraint = Constraint(tech_costs, years, rule=fixed_cost_constraint)
    
    def fixed_cost_constraint_PV(model,year):
        if parameters.at["learning parameter","solar_PV"] == 0:
            return model.fixed_costs["solar_PV",year] == parameters.at["current annuity","solar_PV"]*1000 #EUR/GW
        else:
            return model.fixed_costs["solar_PV",year] == parameters.at["current annuity","solar_PV"]*1000 * (1+sum(model.generators_built["solar_PV",yeart] for yeart in years if yeart < year)*greenfield_df["solar_PV"])**(-PV_learning_param[iteration]) #EUR/GW
            # return model.fixed_costs[tech,year] == parameters.at["base cost",tech] + (parameters.at["current annuity",tech]-parameters.at["base cost",tech])*(1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
    model.fixed_cost_constraint_PV = Constraint(years, rule=fixed_cost_constraint_PV)
    
    #%% CO2 constraint
    # Converting GW to MW (1000), dty to go from 4 days to 365 days, 
    
    # def co2_constraint(model,tech,year,hour): # MtCO2 = MWh*MtCO2/MWh
    #     return co2_budget[year]/interval >= sum((model.generators_dispatch[tech,year,hour]*1000*dty * parameters.at["specific emissions",tech]) for tech in techs for hour in hours) if "co2_constraint" in scenario else Constraint.Skip
    # model.co2_constraint = Constraint(techs,years, hours,rule=co2_constraint)
    
    
    def co2_constraint2050(model,tech,hour): 
            return 0 >= sum((model.generators_dispatch[tech,2050,hour] * 1000 * dty * parameters.at["specific emissions",tech]) for tech in techs for hour in hours) if "co2_constraint" in scenario else Constraint.Skip
    model.co2_constraint2050 = Constraint(techs, hours,rule=co2_constraint2050)
    
    
    
    
    
    #%%
    executionTime = (time.time() - startTime)
    print('Writing time for Pyomo model in seconds: ' + str(executionTime))
    #%% Solving model
    startTime = time.time()
    
    # ipopt #suffixes=['dual'] keepfiles=False mindtpy
    opt = SolverFactory('ipopt')
    opt.options['max_iter'] = 100000
    opt.options['acceptable_constr_viol_tol'] = 0.1
    opt.options['timing_statistics'] = "yes"
    opt.options['max_soft_resto_iters'] = 10000
    opt.options['print_info_string'] = "yes"
    
    ipo.IpoptConfig={'time_limit':100000,'max_iter':100000,'max_cpu_time':100000,'max_soft_resto_iters':10000}
    results = opt.solve(model,suffixes=['dual'], keepfiles=False, tee = True)
    
    # results = opt.solve(model,mip_solver='glpk', nlp_solver='ipopt',tee=True)
    
    
    # results = opt.solve(model,
    #                                     strategy='OA',
    #                                     mip_solver='gurobi',
    #                                     nlp_solver='ipopt',
    #                                     add_regularization='level_L2',
    #                                     nlp_solver_args={'timelimit': 10000},
    #                                     obj_bound = (0,5000)
    #                                     # alternative regularizations
    #                                     # 'level_L1', 'level_L2', 'level_L_infinity',
    #                                     # 'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'
    #                                     )
    
    # results = opt.solve(model,
    #                                     strategy='OA',
    #                                     init_strategy='FP',
    #                                     mip_solver='gurobi',
    #                                     nlp_solver='ipopt',
    #                                     nlp_solver_args={'max_cpu_time':10000},
    #                                     solution_pool=True,
    #                                     num_solution_iteration=300, # default=5
    #                                     tee=True,
    #                                     nlp_solver_tee = True)
    
    # results = opt.solve(model,
    #                                     strategy='GOA',
    #                                     mip_solver='gurobi',
    #                                     nlp_solver='ipopt',
    #                                     nlp_solver_args={'timelimit': 100000},
    #                                     options={"threads": 8},
    #                                     nlp_solver_tee = True,
    #                                     tee=True)
    
    model.objective.display()
    
    
    
    systemcost = model.objective()*1e9/11036900000 # euro/MWh
    print("Avg. cost (in euro/MWh) =","%.2f"% systemcost)
    
    
    CO2_emitted = sum((model.generators_dispatch[tech,year,hour].value*1000*dty*3*interval * parameters.at["specific emissions",tech]) for tech in fossil_techs for hour in hours for year in years)
    print("CO2 emitted =","%.2f"% CO2_emitted) 
    
    if "highLR" in learning_scenario:
        print("High learning rates")
    else: 
        if "lowLR" in learning_scenario:
            print("Low learning rates")
        else:
            # nom learning
            print("Nominal learning rates")
    
    print("Cost for storage =","%.2f"% sum(model.storage_built[tech,year].value * model.fixed_costs_storage[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) 
                                           for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech]))) 
                                           for tech in storage_techs for year in years))
    
    
    print("Cost for renewables =","%.2f"%sum(model.generators_built[tech,year].value * model.fixed_costs[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
      for tech in renewables for year in years))
    
    print("Cost for fossil and nuclear =","%.2f"% sum(model.generators_built[tech,year].value * model.fixed_costs[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
      for tech in fossil_techs for year in years))
    
    print("Marginal costs or fuel costs =","%.2f"%sum(model.generators_dispatch[tech,year,hour].value * parameters.at['marginal cost',tech]*1000*dty*hour_interval*interval / 1e9 /(1+r)**(year-years[0])
        for tech in techs
        for year in years
        for hour in hours))
    print("Total cost (in billion euro) =","%.2f"% model.objective())
    
    
    
    
    results_data[iteration,"marginal_cost"] = sum(model.generators_dispatch[tech,year,hour].value * parameters.at['marginal cost',tech]*1000*dty*hour_interval*interval / 1e9 /(1+r)**(year-years[0])
        for tech in techs
        for year in years
        for hour in hours)
    
    results_data[iteration,"total_cost"] = sum(model.generators_dispatch[tech,year,hour].value * parameters.at['marginal cost',tech]*1000*dty*hour_interval*interval / 1e9 /(1+r)**(year-years[0])
        for tech in techs
        for year in years
        for hour in hours)
    
    results_data[iteration,"fossil_nuclear_cost"] = sum(model.generators_built[tech,year].value * model.fixed_costs[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
      for tech in fossil_techs for year in years)
    
    results_data[iteration,"renewable_cost"] = sum(model.generators_built[tech,year].value * model.fixed_costs[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
      for tech in renewables for year in years)
    
    results_data[iteration,"storage_cost"] = sum(model.storage_built[tech,year].value * model.fixed_costs_storage[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) 
                                           for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech]))) 
                                           for tech in storage_techs for year in years)
    
    


    #%% System costs 
    
    system_costs[iteration,"storage"] = sum(model.storage_built[tech,year].value * model.fixed_costs_storage[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech]))) for tech in storage_techs for year in years)
    system_costs[iteration,"renewables"] = sum(model.generators_built[tech,year].value * model.fixed_costs[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
      for tech in renewables for year in years)
    system_costs[iteration,"fossil"] = sum(model.generators_built[tech,year].value * model.fixed_costs[tech,year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
      for tech in fossil_techs for year in years)
    system_costs[iteration,"nuclear"] = sum(model.generators_built["nuclear",year].value * model.fixed_costs["nuclear",year].value/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',"nuclear"])))
      for year in years)
    system_costs[iteration,"marginal"] = sum(model.generators_dispatch[tech,year,hour].value * parameters.at['marginal cost',tech]*1000*dty*hour_interval*interval / 1e9 /(1+r)**(year-years[0])
        for tech in techs
        for year in years
        for hour in hours)
    #%% Data
    
    # # colormap = "Set2"
    for year in years:
        demand[year] = demand[year].iloc[0:32]
       
    filename = "co2_2050"
    # file name
    if "no_LR" in scenario: 
        filename = filename+scenario
    else:
        filename = filename+scenario+"-"+learning_scenario
    
    if Greenfield == False:
        filename = "Br"+filename
    else:
        filename = "Gr"+filename
        
    
    
    storage_discharge = {}
    storage_charge = {}
    energy_stored = {}
    
    for year in years:
        storage_discharge[year] = pd.DataFrame()
        storage_charge[year] = pd.DataFrame()
        energy_stored[year] = pd.DataFrame()
    
    
        for hour in hours:
            # for tech in storage:
                storage_discharge[year].at[hour,'battery_store'] = model.storage_discharge['battery_store',year,hour].value
                storage_charge[year].at[hour,'battery_store'] = model.storage_charge['battery_store',year,hour].value
                energy_stored[year].at[hour,'battery_store'] = model.SOC['battery_store',year,hour].value
                storage_discharge[year].at[hour,'battery_store'] = -storage_discharge[year].at[hour,'battery_store']
    
    storage_cap = pd.DataFrame(0.,index=years,columns=storage)
    for year in years:
        # for tech in storage:
            storage_cap.at[year,'battery_store'] = model.storage['battery_store',year].value
    
    techsplot = ["nuclear","coal","offshore_wind","onshore_wind","solar_PV", "CCGT","OCGT"]
    
    
    dispatch = {}
    for year in years:
        dispatch[year] = pd.DataFrame(0.,index=hours,columns=techs+storage)
        for hour in hours:
            for tech in techs:
                dispatch[year].at[hour,tech] = model.generators_dispatch[tech,year,hour].value
                dispatch[year].at[hour,"battery_store"] = -storage_discharge[year].at[hour,'battery_store']
                
    for year in years:
        for hour in hours:
            for tech in techs:
                if dispatch[year].at[hour,tech] <= 0:
                    dispatch[year].at[hour,tech] = 0
            if dispatch[year].at[hour,"battery_store"] <= 0:
                dispatch[year].at[hour,"battery_store"] = 0
                
            
    dispatch_allyears = pd.concat([dispatch[year] for year in years])  
    dispatch_allyears = dispatch_allyears.reset_index()
    dispatch_allyears = dispatch_allyears.drop(["index"],axis=1)
    demand_allyears   = pd.concat([demand[year] for year in years]) 
    demand_allyears = demand_allyears.reset_index()
    demand_allyears = demand_allyears.drop(["index"],axis=1)
    
    
    dispatch_4days[iteration] = pd.DataFrame(0.,index=years,columns=techs+storage)
    for year in years:
        for tech in techs:
                dispatch_4days[iteration].at[year,tech] = dispatch[year].loc[:,tech].sum()*3
        for tech in storage:
                dispatch_4days[iteration].at[year,tech] = dispatch[year].loc[:,tech].sum()*3
    
    
    
    capacities_data[iteration] = pd.DataFrame(0.,index=years,columns=techs+storage_techs) 
    for year in years:
        for tech in techs:
            capacities_data[iteration].at[year,tech] = model.generators[tech,year].value
        for tech in storage_techs:
            capacities_data[iteration].at[year,tech] = model.storage[tech,year].value
    
    
    dispatch = {}
    for year in years:
        dispatch[year] = pd.DataFrame(0.,index=hours,columns=techs+storage)
        for hour in hours:
            for tech in techs:
                dispatch[year].at[hour,tech] = model.generators_dispatch[tech,year,hour].value
                dispatch[year].at[hour,"battery_store"] = -storage_discharge[year].at[hour,'battery_store']
                
    for year in years:
        for hour in hours:
            for tech in techs:
                if dispatch[year].at[hour,tech] <= 0:
                    dispatch[year].at[hour,tech] = 0
            if dispatch[year].at[hour,"battery_store"] <= 0:
                dispatch[year].at[hour,"battery_store"] = 0
                
            
    dispatch_data[iteration] = pd.concat([dispatch[year] for year in years])  
    dispatch_data[iteration] = dispatch_data[iteration].reset_index()
    dispatch_data[iteration] = dispatch_data[iteration].drop(["index"],axis=1)
    
        
    build_years_data[iteration] = pd.DataFrame(0.,index=years,columns=techs+storage_techs) 
    for year in years:
        for tech in techs:
            build_years_data[iteration].at[year,tech] = model.generators_built[tech,year].value
        for tech in storage_techs:
            build_years_data[iteration].at[year,tech] = model.storage_built[tech,year].value
    for year in years:
        for tech in techs:
            if build_years_data[iteration].at[year,tech] <= 0:
                build_years_data[iteration].at[year,tech] = 0
                 
                
    dispatch_sum = {}
    lcoe_capfactor = {}
    cap_fac = {}
    for year in years:
        for hour in hours:
            for tech in techs:
                if hour == 0:
                    cap_fac[tech,year] = 0
                if dispatch[year].at[hour,tech] > 0.1:
                    cap_fac[tech,year] = cap_fac[tech,year]+1
            # if dispatch[year].at[hour,"battery_store"] > 0:
            #     dispatch[year].at[hour,"battery_store"] = 0
    
    
    
    for year in years:
            for tech in techs:
                cap_fac[tech,year] = cap_fac[tech,year]*dty*3/8760
                dispatch_sum[year] = dispatch[year].sum(0)*dty*3
                # lcoe_capfactor[tech,year] = dispatch_sum[year].at[tech]/(capacities.at[year,tech]*8760)
    
    
    level_cost = pd.DataFrame(0.,index=years,columns=techs)
    annuity_cost[iteration] = pd.DataFrame(0.,index=years,columns=techs+storage_techs)
    
    for year in years:
            for tech in techs:
                # level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(8760*lcoe_capfactor[tech,year])/1000 + parameters.at["marginal cost",tech]
                level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(1+8760*cap_fac[tech,year])/1000 + parameters.at["marginal cost",tech]
                annuity_cost[iteration].at[year,tech] = model.fixed_costs[tech,year].value/1000 # EUR/MW/a
                # if capacities.at[year,tech]<1: 
                    # level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(8760*1)/1000 + parameters.at["marginal cost",tech]
            for tech in storage_techs:
                annuity_cost[iteration].at[year,tech] = model.fixed_costs_storage[tech,year].value/1000 #EUR/MW/a
    
   
    capcost[iteration]=pd.DataFrame(0.,index=years,columns=techs+storage_techs)
    for year in years:    
        for tech in techs:
            if parameters.at["learning parameter",tech] == 0:
                capcost[iteration].at[year,tech] = parameters.at["current annuity",tech] #EUR/MW
            else:
                capcost[iteration].at[year,tech] = (parameters.at["current annuity",tech] * 
                                         (1+sum(model.generators_built[tech,yeart].value 
                                                for yeart in years if yeart < year)*greenfield_df[tech])**(-parameters.at["learning parameter",tech])) #EUR/MW
        for tech in storage_techs:
            if store_param.at["learning parameter",tech] == 0:
                capcost[iteration].at[year,tech] = store_param.at["current capital cost",tech] #EUR/MW
            else:
                capcost[iteration].at[year,tech] = store_param.at["current capital cost",tech] *(1+sum(model.storage_built[tech,yeart].value 
                                                                                                  for yeart in years if yeart < year)*greenfield_df[tech])**(-store_param.at["learning parameter",tech]) #EUR/MW
    

     
#%% Post processing

for iteration in range(len(PV_learning)):    
    build_years_data[iteration] = build_years_data[iteration].assign(LR=str(PV_learning[iteration]))
    annuity_cost[iteration] = annuity_cost[iteration].assign(LR=str(PV_learning[iteration]))
    capacities_data[iteration] = capacities_data[iteration].assign(LR=str(PV_learning[iteration]))
    capcost[iteration] = capcost[iteration].assign(LR=str(PV_learning[iteration]))
    dispatch_data[iteration] = dispatch_data[iteration].assign(LR=str(PV_learning[iteration]))
    dispatch_4days[iteration] = dispatch_4days[iteration].assign(LR=str(PV_learning[iteration]))
# for iteration in range(28):    
    

build_years_all = pd.concat(build_years_data[iteration] for iteration in range(len(PV_learning)))  
build_years_all = build_years_all.sort_index()
build_years_all["Year"] = build_years_all.index
build_years_all = build_years_all.sort_values(['Year', 'LR'])
build_years_all.to_excel("Results/Exp2/Build_years_allyears_data.xlsx")

dispatch_all = pd.concat(dispatch_data[iteration] for iteration in range(len(PV_learning)))  
dispatch_all = dispatch_all.sort_index()
dispatch_all["Year"] = dispatch_all.index
dispatch_all = dispatch_all.sort_values(['Year', 'LR'])
dispatch_all.to_excel("Results/Exp2/Dispatch_allyears_data.xlsx")

dispatch4d_all = pd.concat(dispatch_4days[iteration] for iteration in range(len(PV_learning)))  
dispatch4d_all = dispatch4d_all.sort_index()
dispatch4d_all["Year"] = dispatch4d_all.index
dispatch4d_all = dispatch4d_all.sort_values(['Year', 'LR'])
dispatch4d_all.to_excel("Results/Exp2/Dispatch4days_allyears_data.xlsx")




fixed_cost_all = pd.concat(capcost[iteration] for iteration in range(len(PV_learning)))  
fixed_cost_all = fixed_cost_all.sort_index()
fixed_cost_all["Year"] = fixed_cost_all.index
fixed_cost_all=fixed_cost_all.sort_values(['Year', 'LR'])
fixed_cost_all.to_excel("Results/Exp2/Fixed_cost_allyears_data.xlsx")

LCOE = fixed_cost_all
for tech in techs:
   LCOE[tech] = (fixed_cost_all[tech]/8760) + parameters.at["marginal cost",tech]
for tech in storage_techs:
   LCOE[tech] = (fixed_cost_all[tech]/8760) + store_param.at["marginal cost",tech]


annuity_cost_all = pd.concat(annuity_cost[iteration] for iteration in range(len(PV_learning)))  
annuity_cost_all = annuity_cost_all.sort_index()
annuity_cost_all["Year"] = annuity_cost_all.index
annuity_cost_all=annuity_cost_all.sort_values(['Year', 'LR'])
annuity_cost_all.to_excel("Results/Exp2/Annuity_cost_allyears_data.xlsx")



capacities_data_all = pd.concat(capacities_data[iteration] for iteration in range(len(PV_learning)))  
capacities_data_all = capacities_data_all.sort_index()
capacities_data_all["Year"] = capacities_data_all.index
capacities_data_all =capacities_data_all.sort_values(['Year', 'LR'])
capacities_data_all.to_excel("Results/Exp2/Capacities_data_all.xlsx")


capex_pv = fixed_cost_all["solar_PV"]
capex_onwind = fixed_cost_all["onshore_wind"]
capex_offwind = fixed_cost_all["offshore_wind"]
capex_battery = fixed_cost_all["battery_store"]
capex_coal = fixed_cost_all["coal"]
capex_nuclear = fixed_cost_all["nuclear"]
capex_ocgt = fixed_cost_all["OCGT"]


build_years_all  = pd.read_excel("Results/Exp2/Build_years_allyears_data.xlsx")
fixed_cost_all = pd.read_excel("Results/Exp2/Fixed_cost_allyears_data.xlsx")
capacities_data_all = pd.read_excel("Results/Exp2/Capacities_data_all.xlsx")
build_years_test  = pd.read_excel("data/Build_years_test.xlsx")




fig = px.scatter(build_years_all, x="Year", y="LR",
	         size="solar_PV", color="solar_PV",
                 hover_name="solar_PV", size_max=30)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        gridwidth=2,
    ),
    yaxis=dict(
        title='PV learning rate',
        gridcolor='white',
        gridwidth=2,
        # range = [0.05,0.48],
    ),)
fig.add_annotation( # add a text callout with arrow
    text="New PV built [GW]", x=2050, y=0.08
)
fig.show()
fig.write_image("Results/Exp2/solarLR_comp_case.png",width=1000, height=500,scale=1.5)


fig = px.scatter(build_years_all, x="Year", y="solar_PV",
	         size="solar_PV", color="LR",
                 hover_name="solar_PV", log_y=True, size_max=30)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        type='log',
        gridwidth=2,
    ),
    yaxis=dict(
        title='New PV built [GW]',
        gridcolor='white',
        gridwidth=2,
        range = [1.5,3.25]
    ),)
fig.show()
fig.write_image("Results/Exp2/solarLR_comp_size_log.png",width=1000, height=500,scale=1.5)

systemcost_df = pd.DataFrame.from_dict(system_costs,orient='index')


fig = px.scatter(build_years_all, x="Year", y="LR",
	         size="onshore_wind", color="onshore_wind",
                 hover_name="battery_store", size_max=30)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        gridwidth=2,
    ),
    yaxis=dict(
        title='PV learning rate',
        gridcolor='white',
        gridwidth=2,
    ),)
fig.add_annotation( # add a text callout with arrow
    text="New Storage built [GW]", x=2050, y=0.4
)
fig.show()
fig.write_image("Results/Exp2/onshore_PVLR_comp_size_log.png",width=1000, height=500,scale=1.5)

fig = px.scatter(build_years_all, x="Year", y="LR",
	         size="offshore_wind", color="offshore_wind",
                 hover_name="battery_store", size_max=30)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        gridwidth=2,
    ),
    yaxis=dict(
        title='PV learning rate',
        gridcolor='white',
        gridwidth=2,
    ),)
fig.add_annotation( # add a text callout with arrow
    text="New Storage built [GW]", x=2050, y=0.4
)
fig.show()
fig.write_image("Results/Exp2/offshore_PVLR_comp_size_log.png",width=1000, height=500,scale=1.5)

fig = px.scatter(build_years_all, x="Year", y="LR",
	         size="nuclear", color="nuclear",
                 hover_name="nuclear", size_max=30)
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        gridwidth=2,
    ),
    yaxis=dict(
        title='PV learning rate',
        gridcolor='white',
        gridwidth=2,
    ),)
fig.add_annotation( # add a text callout with arrow
    text="New Storage built [GW]", x=2050, y=0.4
)
fig.show()
fig.write_image("Results/Exp2/nuclear_PVLR_comp_size_log.png",width=1000, height=500,scale=1.5)

fig = px.scatter(build_years_test, x="Year2", y="LR",
	         size="size", color="type",
                 hover_name="type", size_max=30,
                 color_discrete_sequence=['gold','green','dodgerblue'])
fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
},xaxis=dict(
        title='Year',
        gridcolor='white',
        gridwidth=2,
    ),
    yaxis=dict(
        title='PV learning rate',
        gridcolor='white',
        gridwidth=2,
    ),)
# fig.add_annotation( # add a text callout with arrow
#     text="New Storage built [GW]", x=2050, y=0.4)

fig.show()
fig.write_image("Results/Exp2/triple_PVLR_comp_size_log.png",width=1200, height=400,scale=1.2)




# ine_shape="spline"
# Plotting capital cost of PV
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

# Plotting annuity cost of PV
fig = px.line(annuity_cost_all, x="Year", y="solar_PV",color="LR", line_group="LR", hover_name="LR",
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
        title='Annuity cost of PV [EUR2022/MW/a]',
        gridcolor='grey',
        type='log',
        gridwidth=0.5,
        
    ),)
fig.update_yaxes(tickcolor='crimson', ticklen=10)
fig.show()
fig.write_image("Results/Exp2/solarLR_annuity_log.png",width=800, height=500,scale=1.5)


fig = px.line(LCOE, x="Year", y="solar_PV",color="LR", line_group="LR", hover_name="LR",
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
        title='LCOE of PV [EUR2022/MWh]',
        gridcolor='grey',
        # type='log',
        gridwidth=0.5,
    ),)
fig.show()

def plot_stacked(dfall, labels=years, title="Installed capacities",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe5 = plt.subplot(111)
    for df in dfall : # for each data frame
        axe5 = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe5,
                      legend=False,
                      grid=True,
                      figsize = [14, 6],
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
    axe5.set
    axe5.set_ylabel("GWh")

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe5.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe5.legend(h[:n_col], l[:n_col], loc=[1.0 , 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.0, 0.0]) 
    axe5.add_artist(l1)
    
    left,right = plt.xlim()
    plt.xlim(-0.5,7)
    
    # plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off
    
    plt.xticks(ticks=[0, 6.5/6.5, 2*6.5/6.5, 3*6.5/6.5,4*6.5/6.5,5*6.5/6.5,6*6.5/6.5],labels=['2020','2025','2030','2035','2040','2045','2050'])
    print("Bottom value:",left,"\n","Top Value:",right)
    plt.savefig('Results/Exp2/installedcapacities_PVLR.pdf', 
                dpi=200,
                bbox_inches='tight',
                transparent=True)
    return axe5

from matplotlib.colors import ListedColormap
colors = ["dodgerblue","lightgreen","gold", 'coral',"peru","grey","plum","brown"]
cmap = ListedColormap(colors)


df = capacities_data_all.sort_values(['LR','Year'])
df = df.reset_index()
df = df.drop("Year",axis=1)
# df = df.drop("battery_inverter",axis=1)
df = df.drop("LR",axis=1)
df = df.drop("index",axis=1)
df = df.drop("battery_inverter",axis=1)
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

LR010   = df.iloc[0:7]
LR015   = df.iloc[7:14]
LR020   = df.iloc[14:21]
LR025   = df.iloc[21:28]
LR030   = df.iloc[28:35]
LR035   = df.iloc[35:42]
LR040   = df.iloc[42:49]
LR045   = df.iloc[49:56]



plot_stacked([LR010,LR015,LR020,LR025,LR030,LR035,LR040,LR045],["LR010","LR015","LR020","LR025","LR030","LR035","LR040","LR045"],cmap=cmap)
# plot_stacked([LR010,LR015,LR020],["LR010","LR015","LR020"],cmap=cmap)
# plot_stacked([LR025,LR030,LR035,LR040,LR045],["LR025","LR030","LR035","LR040","LR045"],cmap=cmap)



#%%
#Save data
# # build_years.to_pickle("build_years_one_factor.pkl")
# # capacities.to_pickle("capacities_one_factor.pkl")


# build_years_all.to_pickle("Results/Exp2/{}build_years_2050CO2.pkl".format(filename))
# df.to_pickle("Results/Exp2/{}build_years_2050CO2.pkl".format(filename))
# fixed_cost_all.to_pickle("Results/Exp2/{}fixed_costs_2050CO2.pkl".format(filename))
# capacities.to_pickle("Results/Exp2/{}capacities_2050CO2.pkl".format(filename))
# dispatch_allyears.to_pickle("Results/Exp2/{}dispatch_allyears_2050CO2.pkl".format(filename))
# dispatch_4days.to_pickle("Results/Exp2/{}dispatch_sum_2050CO2.pkl".format(filename))

# capex_pv.to_pickle("Results/Exp2/capex_pv_{}.pkl".format(filename))
# capex_onwind.to_pickle("Results/Exp2/capex_onwind{}.pkl".format(filename))
# capex_offwind.to_pickle("Results/Exp2/capex_offwind{}.pkl".format(filename))
# capex_battery.to_pickle("Results/Exp2/capex_battery{}.pkl".format(filename))
# capex_coal.to_pickle("Results/Exp2/capex_coal{}.pkl".format(filename))
# capex_nuclear.to_pickle("Results/Exp2/capex_nuclear{}.pkl".format(filename))
# capex_ocgt.to_pickle("Results/Exp2/capex_ocgt{}.pkl".format(filename))




executionTime = (time.time() - startTime)
print('Solving model in seconds: ' + str(executionTime))
