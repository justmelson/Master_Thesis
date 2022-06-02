#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:19:15 2022

@author: frederikmelson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:21:35 2022

@author: frederikmelson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:36:01 2022

@author: frederikmelson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:08:12 2022

@author: frederikmelson
"""



#%% Packages
import math 
from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Suffix, exp, value, Boolean
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyomo.contrib.appsi.solvers.ipopt as ipo

import time
startTime = time.time()
plt.style.use("seaborn")


#%% Scenarios and settings

scenario = "no_co2-no_learning"
# scenario = "co2-0p2-no_learning"
# scenario = "co2-0p2-learning"
# scenario = "no_co2-learning"

# learning_scenario = "high_learning"
# learning_scenario = "low_learning"
learning_scenario = "nom_learning"


# CO2 budget for 2050 global warming goals
co2_until_2050 = 10.5e9 # 100 million tCO2 ,10000000000 # 10 gigaton CO2

# Greenfield scenario 
Greenfield = True


# legend on/off when plotting
lgnd = True

r = 0.01 # discount rate

hours = list(range(112))


parameters  = pd.read_pickle("parameters.pkl")
store_param = pd.read_pickle("store_param.pkl")
# demand = pd.DataFrame(columns= ["demand"])

a_file = open("demand_elec.pkl", "rb")
demand = pickle.load(a_file)
a_file.close()

b_file = open("demand_elec5y.pkl", "rb")
demand5y = pickle.load(b_file)
b_file.close()


# CF_solar_one    = pd.read_pickle("CF_solar_one.pkl")
# CF_onwind_one   = pd.read_pickle("CF_onwind_one.pkl")
# CF_offwind_one  = pd.read_pickle("CF_offwind_one.pkl")

# Cf_solar    = pd.read_pickle("Cf_solar.pkl")
# Cf_onshore  = pd.read_pickle("Cf_onshore.pkl")
# Cf_offshore = pd.read_pickle("Cf_offshore.pkl")


Cf_solar      = pd.read_pickle("cf_solar3h.pkl")
Cf_onshore    = pd.read_pickle("cf_onshore3h.pkl")
Cf_offshore   = pd.read_pickle("cf_offshore3h.pkl")
demand2w3h      = pd.read_pickle("demand2w3h.pkl")
demand2w3h.reset_index()



# parameters.at["current capital cost","coal"] = 10000

# demand = 400 #GWh

techs_file          = "techs.pkl"
fossil_techs_file   = "fossil_techs.pkl"
renewables_file     = "renewables.pkl"
wind_file           = "wind.pkl"
colors_file         = "colors.pkl"
storage_file        = "storage.pkl"
color_storage_file  = "color_storage.pkl"

files = [techs_file,fossil_techs_file,renewables_file,wind_file,colors_file,storage_file,color_storage_file]
lists = ["techs","fossil_techs","renewables","wind","colors","storage","color_storage"]


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

storage = ['battery_store']
storage_techs = ['battery_store','battery_inverter']

# Green- or brownfield scenario:
if Greenfield is True:
    for tech in techs:
        parameters.loc["existing age"]      = [0,0,0,0,0,0,0] #years
        parameters.loc["existing capacity"] = [0,0,0,0,0,0,0]
    print("Greenfield approach")
else:
    print("Brownfield approach")
    
# parameters.at["current capital cost","coal"] *= 0.5
# store_param.at["current capital cost","battery_store"] *= 1000

#%% Updating learning rates and CO2 budget

#Currently installed capacities in GW is used to assume current demand

# Hourly demand
years = [2020,2025,2030,2035,2040,2045,2050]

# years = [2020,2023,2026,2029,2032,2035,2038,2041,2044,2047,2050]

interval = years[1]-years[0]


if "no_learning" in scenario:
    parameters.loc["learning rate"]     = 0
    store_param.loc["learning rate"]    = 0
    print("No learning")
else:
    if "high_learning" in learning_scenario:
        parameters.loc["learning rate"]     = [0.12,0.12,0.23,0.14,0.15,0.05,0.06] # [0.19,0.32,0.47,0.34,0.15,0.083]
        store_param.loc["learning rate"]    = [0.18,0.1,0.1,0.26,0.21]
        print("High learning rates")
    else: 
        if "low_learning" in learning_scenario:
            parameters.loc["learning rate"]     = [0.05,0,0.1,0,0.15,0.06,0.0] # [0.05,0,0.1,-0.01,0.15,0.06,-0.06]
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


# carbon budget in average tCO2   
if "no_co2" in scenario:
    co2_budget = 1e30
    print("No CO2 budget")
else:
    co2_budget = co2_until_2050 # [tCO2] 10 Gigatons CO2
    print("CO2 budget of "+ str(co2_until_2050) + " tons CO2")


    
#%% One node model
model = ConcreteModel()
model.generators            = Var(techs, years, within=NonNegativeReals)
model.generators_dispatch   = Var(techs, years, hours, within=NonNegativeReals)
model.generators_built      = Var(techs, years, within=NonNegativeReals)
model.fixed_costs           = Var(techs, years, within=NonNegativeReals)

model.SOC                   = Var(storage_techs,years, hours,initialize=0, within=NonNegativeReals)
# model.posDeltaSOC           = Var(years,hours, initialize=0) #1
# model.negDeltaSOC           = Var(years,hours, initialize=0) #2
# model.Bool_char             = Var(years,hours, within=Boolean) #9
# model.Bool_dis              = Var(years,hours, within=Boolean,initialize=0) #10

# # single value parameters
# m.etaChg = en.Param(initialize = batt.charging_efficiency)
# m.etaDisChg = en.Param(initialize = batt.discharging_efficiency)
# m.ChargingLimit = en.Param(initialize = batt.charging_power_limit*(15/60.))
# m.DischargingLimit = en.Param(initialize = batt.discharging_power_limit*(15/60.))

model.storage               = Var(storage_techs, years, initialize=0, within=NonNegativeReals)
model.storage_built         = Var(storage_techs, years, within=NonNegativeReals)
model.fixed_costs_storage   = Var(storage_techs, years, within=NonNegativeReals)
model.storage_charge        = Var(storage_techs, years, hours, within=NonNegativeReals)
model.storage_discharge     = Var(storage_techs, years, hours, within=NonNegativeReals)



#Value of currently installed technology:
constant = sum(parameters.at['existing capacity',tech] * parameters.at['current capital cost', tech]*1000/1e9/(1+r)**(hour-hours[0]) for tech in techs for hour in hours if hour < hours[0] + parameters.at['lifetime',tech] - parameters.at['existing age',tech])
print("Cost of existing capacity =", "%.2f"%constant)


model.objective = Objective(expr=constant +
                            sum(model.generators_built[tech,year] * model.fixed_costs[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                              for tech in techs for year in years)+
                            sum(model.generators_dispatch[tech,year,hour] * parameters.at['marginal cost',tech]*1000*26*interval / 1e9 /(1+r)**(2020-years[0])
                                for tech in techs
                                for year in years
                                for hour in hours)
                                +
                            sum(model.storage_built[tech,year] * model.fixed_costs_storage[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech])))
                                for tech in storage_techs for year in years))


#%% Balance Constraints
# def balance_constraint(model,tech,year,hour): # GWh
#     return demand[year].at[hour,0]  <= sum(model.generators_dispatch[tech, year, hour] for tech in techs) + model.storage_discharge['battery_store',year,hour]*model.Bool_dis[year,hour] - model.storage_charge['battery_store',year,hour]*model.Bool_char[year,hour]
# model.balance_constraint = Constraint(techs,years,hours, rule=balance_constraint)



def balance_constraint(model,tech,year,hour): # GWh
    return demand5y[year].at[hour,0]  == sum(model.generators_dispatch[tech, year, hour] for tech in techs) + model.storage_discharge['battery_store',year,hour] - model.storage_charge['battery_store',year,hour]
model.balance_constraint = Constraint(techs,years,hours,rule=balance_constraint)


# def ramp_rateup_constraint(model,tech,year,hour):
#     if hour == 0:
#         return Constraint.Skip
#     else:
#         return model.generators_dispatch[tech,year,hour] >= model.generators_dispatch[tech,year,hour-1]-parameters.at['ramp rate',tech]*model.generators[tech,year]
# model.ramp_rateup_constraint = Constraint(techs,years,hours, rule=ramp_rateup_constraint)

# def ramp_ratedown_constraint(model,tech,year,hour):
#     if hour == 0:
#         return Constraint.Skip
#     else:
#         return model.generators_dispatch[tech,year,hour] <= model.generators_dispatch[tech,year,hour-1]+parameters.at['ramp rate',tech]*model.generators[tech,year]
# model.ramp_ratedown_constraint = Constraint(techs,years,hours, rule=ramp_ratedown_constraint)

        
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
        return model.SOC['battery_store',year,hour] == 0# model.storage["battery_store",year] #- model.storage_discharge['battery_store',year,hour]
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

def inverter_constraint(model,year):
    return model.storage["battery_store",year] <= model.storage['battery_inverter',year]
model.inverter_constraint = Constraint(years, rule=inverter_constraint)


def fixed_cost_constraint_storage(model,tech,year):
    if store_param.at["learning parameter",tech] == 0:
        return model.fixed_costs_storage[tech,year] == store_param.at["current capital cost",tech]*1000 #EUR/GW
    else:
        return model.fixed_costs_storage[tech,year] == store_param.at["current capital cost",tech]*1000 * (1+sum(model.storage_built[tech,yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter","battery_store"]) #EUR/GW
        # return model.fixed_costs["battery_store",year] == store_param.at["base cost","battery_store"] + (store_param.at["current capital cost","battery_store"]-store_param.at["base cost","battery_store"])*(1+sum(model.generators_built["battery_store",yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter","battery_store"])
model.fixed_cost_constraint_storage = Constraint(storage_techs,years, rule=fixed_cost_constraint_storage)

#%% Installed capacity constraints
def generator_constraint(model,tech,year,hour):
    return model.generators_dispatch[tech,year,hour] <= model.generators[tech,year] # GW
model.generator_constraint = Constraint(techs, years, hours, rule=generator_constraint)



def capacity_constraint(model,tech,year):
    return sum(model.generators[tech,year] for tech in techs) <= 1000000
model.capacity_constraint = Constraint(techs, years, rule=capacity_constraint)


def build_years(model,tech,year):
    if year < years[0] + parameters.at["lifetime",tech] - parameters.at["existing age",tech]:
        constant = parameters.at["existing capacity",tech] 
    else:
        constant = 0.
    
    return model.generators[tech,year] == constant + sum(model.generators_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + parameters.at["lifetime",tech]))) #GW
model.build_years = Constraint(techs, years, rule=build_years)

# def build_rate_constraint(model,tech,year):
#     model.generators_built[tech,year) >= 


def fixed_cost_constraint(model,tech,year):
    if parameters.at["learning parameter",tech] == 0:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]*1000 #EUR/GW
    else:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]*1000 * (1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech]) #EUR/GW
        # return model.fixed_costs[tech,year] == parameters.at["base cost",tech] + (parameters.at["current capital cost",tech]-parameters.at["base cost",tech])*(1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
model.fixed_cost_constraint = Constraint(techs, years, rule=fixed_cost_constraint)

#%% CO2 constraint
# Converting GW to MW (1000), 26 to go from 2 to 52 weeks, 3 to go from 1 to 3 years as that is the interval
yearsCO2 = [2020,2025,2030,2035,2040,2045]

def co2_constraint(model,tech,year,hour):
    return co2_budget >= sum((model.generators_dispatch[tech,year,hour] *1000*26*interval * parameters.at["specific emissions",tech]) for tech in techs for hour in hours for year in years)
model.co2_constraint = Constraint(techs,yearsCO2, hours,rule=co2_constraint)


def co2_constraint2050(model,tech,hour):
    if "no_co2" in scenario:
        return 1e30 >= sum((model.generators_dispatch[tech,2050,hour] * 1000 * parameters.at["specific emissions",tech]) for tech in techs for hour in hours)
    else:
        return 0 >= sum((model.generators_dispatch[tech,2050,hour] * 1000 * parameters.at["specific emissions",tech]) for tech in techs for hour in hours)
model.co2_constraint2050 = Constraint(techs, hours,rule=co2_constraint2050)


executionTime = (time.time() - startTime)
print('Writing time for Pyomo model in seconds: ' + str(executionTime))
#%% Solving model
# startTime = time.time()

# # opt = SolverFactory('ipopt')
# # ipo.IpoptConfig={'time_limit':100000,'max_iter':10000}
# # results = opt.solve(model,suffixes=['dual'], keepfiles=False, tee = True)


# # opt = SolverFactory('mindtpy')
# # results = opt.solve(model,suffixes=['dual'],keepfiles=False)

# # results = opt.solve(model,mip_solver='glpk', nlp_solver='ipopt',tee=True)

# opt = SolverFactory("gurobi")# solver_io='python')
# opt.options["NonConvex"] = 2
# opt.options['MIPGap'] = 0.01
# opt.options['Threads'] = 8
# results = opt.solve(model, load_solutions=True, tee=True)

# # results = opt.solve(model,
# #                                     strategy='OA',
# #                                     mip_solver='gurobi',
# #                                     nlp_solver='ipopt',
# #                                     add_regularization='level_L1',
# #                                     nlp_solver_args={'timelimit': 10000},
# #                                     # obj_bound = (0,5000)
# #                                     # alternative regularizations
# #                                     # 'level_L1', 'level_L2', 'level_L_infinity',
# #                                     # 'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'
# #                                     )

# # results = opt.solve(model,
# #                                     strategy='OA',
# #                                     init_strategy='FP',
# #                                     mip_solver='gurobi',
# #                                     nlp_solver='ipopt',
# #                                     nlp_solver_args={'timelimit': 100000},
# #                                     solution_pool=True,
# #                                     num_solution_iteration=30, # default=5
# #                                     tee=True
# #                                     )

# # results = opt.solve(model,
# #                                     strategy='GOA',
# #                                     mip_solver='gurobi',
# #                                     nlp_solver='ipopt',
# #                                     nlp_solver_args={'timelimit': 100000},
# #                                     tee=True)

# model.objective.display()

# print("Total cost (in billion euro) =","%.2f"% model.objective())

# systemcost = model.objective()*1e9/MWh_total # euro/MWh
# print("Avg. cost (in euro/MWh) =","%.2f"% systemcost)


#%% Plotting

# # colormap = "Set2"


# file name
if "no_learning" in scenario: 
    filename = scenario+"LR"
else:
    filename = scenario+"_"+learning_scenario+"highres"

if Greenfield == False:
    filename = filename+"brownfield"
    
    
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
    dispatch[year] = pd.DataFrame(0.,index=hours,columns=techs+storage_techs)
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

# yr = 2050
# fig, ax = plt.subplots()
# fig.set_dpi((200))
# dispatch[yr].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
# ax.set_xlabel("day")
# ax.set_ylabel("Gross electricity generation [GWh]")
# plt.title(str(yr))
# # plt.xticks(ticks=[0,111/14,2*111/14],labels=['1','2')
# plt.xticks(ticks=[0,112/14,2*112/14,3*112/14,4*112/14,5*112/14,6*112/14,7*112/14,8*112/14,9*112/14,10*112/14,11*112/14,12*112/14,13*112/14],labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-dispatch2050.png".format(filename),transparent=True)

# fig, ax = plt.subplots(figsize = [12, 4], dpi = 200)
# dispatch_allyears.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
# ax.set_xlabel("year")
# plt.xticks(ticks=[0, 784/ 7,2* 784/ 7,3* 784/ 7,4* 784/ 7,5* 784/ 7,6* 784/ 7],labels=['2020','2025','2030','2035','2040','2045','2050'])
# ax.set_ylabel("Gross electricity generation [GWh]")
# # plt.title('2050')
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-dispatchallyears.png".format(filename),transparent=True)




capacities = pd.DataFrame(0.,index=years,columns=techs+storage_techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = model.generators[tech,year].value
    for tech in storage_techs:
        capacities.at[year,tech] = model.storage[tech,year].value

# fig, ax = plt.subplots()
# fig.set_dpi((200))
# capacities.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
# ax.set_xlabel("year")
# ax.set_ylabel("Installed capacity [GW]")
# # ax.set_ylim([0,10000])
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-capacity.png".format(filename),transparent=True)

build_years = pd.DataFrame(0.,index=years,columns=techs+storage_techs) # +storage
for year in years:
    for tech in techs:
        build_years.at[year,tech] = model.generators_built[tech,year].value
    for tech in storage_techs:
        build_years.at[year,tech] = model.storage_built[tech,year].value

for year in years:
    for tech in techs:
        if build_years.at[year,tech] <= 0:
            build_years.at[year,tech] = 0
    for tech in storage_techs:
        if build_years.at[year,tech] <= 0:
            build_years.at[year,tech] = 0
            
            
# # for year in years:
# #     for tech in storage:
# #         build_years.at[year,tech] = model.storage_built[tech, year].value

# fig, ax = plt.subplots()
# fig.set_dpi((200))
# build_years.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
# ax.set_xlabel("year")
# ax.set_ylabel("new capacity built [GW]")
# # ax.set_ylim([0,250])
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-new_capacity.png".format(filename),transparent=True)

dispatch_sum = {}
lcoe_capfactor = {}
for year in years:
        for tech in techs:
                dispatch_sum[year] = dispatch[year].sum(0)*26*3
                lcoe_capfactor[tech,year] = dispatch_sum[year].at[tech]/(capacities.at[year,tech]*8760)


level_cost = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
        for tech in techs:
            level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(8760*lcoe_capfactor[tech,year])/1000 + parameters.at["marginal cost",tech]
            if capacities.at[year,tech]<1: 
                level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(8760*1)/1000 + parameters.at["marginal cost",tech]
        
                
        # for tech in storage:
        #     LCOE.at[year,tech] = model.fixed_costs_storage[tech, year].value/8760. + store_param.at["marginal cost",tech]
# https://www.nrel.gov/analysis/tech-lcoe-documentation.html

# fig, ax = plt.subplots()
# fig.set_dpi(200)
# level_cost.plot(ax=ax,linewidth=3,color=colors)
# ax.set_xlabel("year")
# ax.set_ylabel("LCOE [EUR/MWh]")
# # ax.set_yscale("log")
# ax.set_ylim([0,1000])
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-lcoe.png".format(filename),transparent=True)


# emissions = pd.DataFrame(0.,index=years,columns=techs)
# for year in years:
#     for tech in techs:
#         emissions.at[year,tech] = model.generators_dispatch[tech,year].value*8760* 1000 * parameters.at["specific emissions",tech]

# fig, ax = plt.subplots()
# fig.set_dpi(2000)
# emissions.plot(ax=ax,linewidth=3,cmap=colormap)
# ax.set_xlabel("year")
# ax.set_ylabel("CO2 [t]")
# # ax.set_yscale("log")
# # ax.set_ylim([0,40])
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-emissions.png".format(filename),transparent=True)

#%% Storage plots
storage_discharge = {}
storage_charge = {}
energy_stored = {}

for year in years:
    storage_discharge[year] = pd.DataFrame()
    storage_charge[year] = pd.DataFrame()
    energy_stored[year] = pd.DataFrame()


    for hour in hours:
        # for tech in storage:
            storage_discharge[year].at[hour,'Discharge'] = model.storage_discharge['battery_store',year,hour].value
            storage_charge[year].at[hour,'Charge'] = model.storage_charge['battery_store',year,hour].value
            energy_stored[year].at[hour,'State of charge (SOC)'] = model.SOC['battery_store',year,hour].value
            storage_discharge[year].at[hour,'Discharge'] = -storage_discharge[year].at[hour,'Discharge']
storage_cap = pd.DataFrame(0.,index=years,columns=storage)
for year in years:
    # for tech in storage:
        storage_cap.at[year,'battery_store'] = model.storage['battery_store',year].value


# fig, ax = plt.subplots(figsize = [12, 4], dpi = 200)
# storage_discharge[2050].plot(kind="line",ax=ax,linewidth=1,color='r',label='Storage discharged')#cmap=colormap)
# storage_charge[2050].plot(kind="line",ax=ax,linewidth=1,color="b",label='Storage charged')#cmap=colormap)
# energy_stored[2050].plot(kind="line",ax=ax,linewidth=1.5,color='g',label='State of charge')#cmap=colormap)
# ax.set_xlabel("days")
# ax.set_ylabel("[GWh]")
# plt.title('2050')
# plt.xticks(ticks=[0,111/14,2*111/14,3*111/14,4*111/14,5*111/14,6*111/14,7*111/14,8*111/14,9*111/14,10*111/14,11*111/14,12*111/14,13*111/14],labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13'])
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# # ax.legend().set_visible(lgnd)
# plt.legend()
# fig.savefig("Figures_2whighres/{}-storageSOC.png".format(filename),transparent=True)


# #%%
# #Save data
# # build_years.to_pickle("build_years_one_factor.pkl")
# # capacities.to_pickle("capacities_one_factor.pkl")


# # build_years.to_pickle("build_years_multi.pkl")
# # capacities.to_pickle("capacities_multi.pkl")


executionTime = (time.time() - startTime)
print('Solving model in seconds: ' + str(executionTime))
