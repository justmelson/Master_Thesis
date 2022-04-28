#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:05:52 2022

@author: frederikmelson
"""


#%% Packages
import math
# from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Suffix, exp, value
# from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import time
startTime = time.time()
plt.style.use("seaborn")


#%% Scenarios and settings

# scenario = "no_co2-no_learning"
# scenario = "co2-0p2-no_learning"
# scenario = "co2-0p2-learning"
scenario = "no_co2-learning"

# learning_scenario = "high_learning"
# learning_scenario = "low_learning"
learning_scenario = "nom_learning"


# CO2 budget for 2050 global warming goals
co2_until_2050 = 1e10 # 100 million tCO2 ,10000000000 # 10 gigaton CO2

# Greenfield scenario 
Greenfield = True


# legend on/off when plotting
lgnd = True

r = 0.01 # discount rate

hours = list(range(111))


parameters  = pd.read_pickle("parameters.pkl")
store_param = pd.read_pickle("store_param.pkl")

a_file = open("demand_elec.pkl", "rb")
demand = pickle.load(a_file)


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
demand2w3h = demand2w3h/1000



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
storage_list         = lists[5]
color_storage   = lists[6]

storage_list = ['battery_store']


# Green- or brownfield scenario:
if Greenfield is True:
    for tech in techs:
        parameters.loc["existing age"]      = [0,0,0,0,0,0,0] #years
        parameters.loc["existing capacity"] = [0,0,0,0,0,0,0]
    print("Greenfield approach")
else:
    print("Brownfield approach")

xx={}
for i in range(len(demand2w3h)):
    xx[i] = getattr(demand2w3h.at[i,0], "tolist", lambda: demand2w3h.at[i,0])()




#%% Updating learning rates and CO2 budget

#Currently installed capacities in GW is used to assume current demand

# Hourly demand
# years = [2020,2023,2026,2029,2032,2035,2038,2041,2044,2047,2050]
years = [2020,2026,2032,2038,2044,2050]
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
    parameters.loc["learning parameter"][i] = (math.log(1/(1-parameters.loc["learning rate"][i])) / math.log(2))
for i in range(len(storage_list)):
    store_param.loc["learning parameter"][i] = (math.log(1/(1-store_param.loc["learning rate"][i])) / math.log(2))



# carbon budget in average tCO2   
if "no_co2" in scenario:
    co2_budget = 1e30
    print("No CO2 budget")
else:
    co2_budget = co2_until_2050 # [tCO2] 10 Gigatons CO2
    print("CO2 budget of "+ str(co2_until_2050) + " tons CO2")



    
#%% One node model

import gurobipy as gp
from gurobipy import GRB

# Create a new model

model = gp.Model('qp')

generators          = model.addVars(techs, years, vtype=GRB.CONTINUOUS, name="generators")
generators_dispatch = model.addVars(techs, years, hours, vtype=GRB.CONTINUOUS, name="generators_dispatch")
generators_built    = model.addVars(techs, years, vtype=GRB.CONTINUOUS, name="generators_built")
fixed_costs         = model.addVars(techs, years, vtype=GRB.CONTINUOUS, name="fixed_costs")

# storage           = model.addVars(storage_list, years, vtype=GRB.CONTINUOUS, name="storage")
# SOC               = model.addVars(storage_list, years, hours, vtype=GRB.CONTINUOUS, name="SOC")
# storage_built     = model.addVars(storage_list, years, vtype=GRB.CONTINUOUS, name="storage_list_built")
# fixed_costs_storage = model.addVars(storage_list, years, vtype=GRB.CONTINUOUS, name="fixed_costs_storage_list")
# storage_charge    = model.addVars(storage_list, years, hours, vtype=GRB.CONTINUOUS, name="storage_list_charge")
# storage_discharge = model.addVars(storage_list, years, hours, vtype=GRB.CONTINUOUS, name="storage_list_discharge")

# Variables for boolean operations
b  = model.addVar(vtype=GRB.BINARY, name="b")
z  = model.addVar(vtype=GRB.CONTINUOUS, name="z")
w1 = model.addVar(vtype=GRB.CONTINUOUS, name="w1")
w2 = model.addVar(vtype=GRB.CONTINUOUS, name="w2")


# # single value parameters
# m.etaChg = en.Param(initialize = batt.charging_efficiency)
# m.etaDisChg = en.Param(initialize = batt.discharging_efficiency)
# m.ChargingLimit = en.Param(initialize = batt.charging_power_limit*(15/60.))
# m.DischargingLimit = en.Param(initialize = batt.discharging_power_limit*(15/60.))


#Value of currently installed technology:
constant = sum(parameters.at['existing capacity',tech] * parameters.at['current capital cost', tech]*1000/1e9/(1+r)**(hour-hours[0]) for tech in techs for hour in hours if hour < hours[0] + parameters.at['lifetime',tech] - parameters.at['existing age',tech])
print("Cost of existing capacity =", "%.2f"%constant)


model.setObjective(constant +
                            sum(generators_built[tech,year] * fixed_costs[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                              for tech in techs for year in years)+
                            sum(generators_dispatch[tech,year,hour] * parameters.at['marginal cost',tech]*1000*26*interval / 1e9 /(1+r)**(2020-years[0])
                                for tech in techs
                                for year in years
                                for hour in hours))#+
                            # sum(storage_built[tech,year] * fixed_costs_storage[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech])))
                            #     for tech in storage_list for year in years))

#%% Balance Constraints


balance_constraint = model.addConstrs(xx[hour] <= gp.quicksum(generators_dispatch[tech, year, hour] for tech in techs) for year in years for hour in hours)
                                      # + storage_discharge['battery_store',year,hour] for hour in hours for year in years)


#%% Solar capacity constraints


solar_constraint = model.addConstrs(generators_dispatch["solar_PV",year,hour] <= generators["solar_PV",year]*Cf_solar.at[hour,0] 
                                    for hour in hours for year in years)


#%% Onshore wind capacity constraints


onshore_constraint = model.addConstrs(generators_dispatch["onshore_wind",year,hour] <= generators["onshore_wind",year]*Cf_onshore.at[hour,0] 
                                    for hour in hours for year in years)


#%% Offshore wind capacity constraints


offshore_constraint = model.addConstrs(generators_dispatch["offshore_wind",year,hour] <= generators["offshore_wind",year]*Cf_offshore.at[hour,0] 
                                    for hour in hours for year in years)


#%% Storage

# storage__charge_constraint = model.addConstrs(storage_charge['battery_store',year,hour]*storage_discharge['battery_store',year,hour] == 0 for hour in hours for year in years)




# # Add indicator constraints
# # model.addConstrs((b == 1) >> (z == w1), name="indicator_constr1")
# # model.addConstrs((b == 0) >> (z == w2), name="indicator_constr2")

# for year in years:
#     for hour in hours: 
#         if year < years[0] + parameters.at["lifetime",tech] - parameters.at["existing age",tech]: 
#             constantbuildstore = parameters.at["existing capacity",tech] 
#         else:
#             constantbuildstore = 0.

# stored_energy_constraint_start = model.addConstrs(SOC['battery_store',year,0] == storage["battery_store",year] for year in years)

# stored_energy_constraint = model.addConstrs(SOC['battery_store',year,hour] == SOC['battery_store',year,hour-1] + storage_charge['battery_store',year,hour] - 
#                                            storage_discharge['battery_store',year,hour] for year in years for hour in hours[1:111])


# storage_constraint = model.addConstrs(storage_discharge['battery_store',year,hour]<= SOC['battery_store',year,hour] for year in years for hour in hours)

# storage_capacity_constraint = model.addConstrs(SOC['battery_store',year,hour] <= storage['battery_store',year] for year in years for hour in hours)


# for year in years:
#     for tech in storage_list: 
#         if year < years[0] + store_param.at["lifetime",tech] - store_param.at["existing age",tech]: 
#             constantbuildstore = store_param.at["existing capacity",tech] 
#         else:
#             constantbuildstore = 0.


# build_years_storage = model.addConstrs(storage["battery_store",year] == constantbuildstore + sum(storage_built["battery_store",yearb] 
#                                                                             for yearb in years if ((year>= yearb) and (year < yearb + 
#                                                                                                                        store_param.at["lifetime","battery_store"]))) for year in years) #GW
# # Constants
# eps = 0.0001
# M   = 2060 + eps  # smallest possible given bounds on x and y

# # Model if x>y then b = 1, otherwise b = 0
# model.addConstrs((year >= hour + eps - M * (1 - b) for hour in hours), name="bigM_constr1")

# model.addConstrs((0 <= hour + M * b for hour in hours), name="bigM_constr2")

# x={}# = model.addVars(storage_list,years,hours, vtype=GRB.CONTINUOUS, name="xvar")


# helper1 = model.addVars(storage_list,years, vtype="C")
# helper2 = model.addVars(storage_list,years, vtype="C")

# # for year in years:
# #     for yeart in years:
# #         if yeart < year:
# #             model.addConstrs(helper1[year] == 1+gp.quicksum(storage_built["battery_store",yeart]))
# #             model.addGenConstrsPow(helper2[year], helper1[year],store_param.at("learning parameter","battery_store"))
# #             # x = model.addGenConstrPow(1+gp.quicksum(storage_built["battery_store",yeart]),x,store_param.at("learning parameter","battery_store"))

# model.addConstrs(helper1["battery_store",year] == 1+(gp.quicksum(storage_built["battery_store",yeart] for yeart in years for year in years if yeart < year)) for year in years)
# for year in years:
#      model.addGenConstrPow(helper2["battery_store",year], helper1["battery_store",year],store_param.at["learning parameter","battery_store"])

# model.Params.FuncPieces = 0
# model.Params.FuncPieceLength = 10


# fixed_cost_constraint_storage = model.addConstrs(fixed_costs_storage["battery_store",year] == store_param.at["current capital cost","battery_store"] * 1000 * helper2["battery_store",year] for year in years) #EUR/GW)


#%% Installed capacity constraints
# def generator_constraint(model,tech,year,hour):
#     return model.generators_dispatch[tech,year,hour] <= model.generators[tech,year] # GW
# model.generator_constraint = Constraint(techs, years, hours, rule=generator_constraint)

generator_constraint = model.addConstrs(generators_dispatch[tech,year,hour] <= generators[tech,year] for tech in techs for year in years for hour in hours)

capacity_constraint = model.addConstrs(sum(generators[tech,year] for tech in techs for year in years) <= 1000000 for tech in techs for year in years)



for year in years:
    for tech in techs: 
        if year < years[0] + parameters.at["lifetime",tech] - parameters.at["existing age",tech]: 
            constantbuild = parameters.at["existing capacity",tech] 
        else:
            constantbuild = 0.


build_years = model.addConstrs(generators[tech,year] == constantbuild + sum(generators_built[tech,yearb] 
                                                                            for yearb in years if ((year>= yearb) and (year < yearb + 
                                                                                                                       parameters.at["lifetime",tech]))) for tech in techs for year in years) #GW
  
# for year in years:
#     gb = gp.quicksum(generators_built[tech,yeart] for tech in techs for yeart in years if yeart < year)

# fixed_cost_constraint = model.addConstrs(fixed_costs[tech,year] == parameters.at["current capital cost",tech]*1000 for tech in techs for year in years) #EUR/GW

# fixed_cost_constraint = model.addGenConstr(fixed_costs[tech,year] == parameters.at["current capital cost",tech] * 1000 * 
#                                          (1+gp.quicksum(generators_built[tech,yeart] for tech in techs for yeart in years if yeart < year))
#                                          **(parameters.at["learning parameter",tech]) for tech in techs for year in years) #EUR/GW)




helper3 = model.addVars(techs,years, vtype="C")
helper4 = model.addVars(techs,years, vtype="C")

# for year in years:
#     for yeart in years:
#         if yeart < year:
#             model.addConstrs(helper1[year] == 1+gp.quicksum(storage_built["battery_store",yeart]))
#             model.addGenConstrsPow(helper2[year], helper1[year],store_param.at("learning parameter","battery_store"))
#             # x = model.addGenConstrPow(1+gp.quicksum(storage_built["battery_store",yeart]),x,store_param.at("learning parameter","battery_store"))

model.addConstrs(helper3[tech,year] == 1+(gp.quicksum(generators_built[tech,yeart] for tech in techs for yeart in years for year in years if yeart < year)) for tech in techs for year in years)

for year in years:
    for tech in techs:
     model.addGenConstrPow(helper4[tech,year], helper3[tech,year],parameters.at["learning parameter",tech])

fixed_cost_constraint = model.addConstrs(fixed_costs[tech,year] == parameters.at["current capital cost",tech] * 1000 * helper4[tech,year] for tech in techs for year in years) #EUR/GW)


#%% CO2 constraint
# Converting GW to MW (1000), 26 to go from 2 to 52 weeks, 3 to go from 1 to 3 years as that is the interval


co2_constraint = model.addConstrs(co2_budget >= sum(generators_dispatch[tech,year,hour] * 1000*26*3 * parameters.at["specific emissions",tech] for tech in techs for year in years for hour in hours)for tech in techs for year in years for hour in hours)

co2_constraint = model.addConstrs(0 >= sum((generators_dispatch[tech,2050,hour] * 1000*26*3 * parameters.at["specific emissions",tech]) for tech in techs for hour in hours) for tech in techs for year in years for hour in hours)


executionTime = (time.time() - startTime)
print('Writing time for Pyomo model in seconds: ' + str(executionTime))
#%% Solving model
startTime = time.time()

model.optimize()

# systemcost = model.objective()*1e9/MWh_total # euro/MWh
# print("Avg. cost (in euro/MWh) =","%.2f"% systemcost)


#%% Plotting

# # colormap = "Set2"


# file name
if "no_learning" in scenario: 
    filename = scenario+"LR"
else:
    filename = scenario+"_"+learning_scenario+"highres"


dispatch = {}
for year in years:
    dispatch[year] = pd.DataFrame(0.,index=hours,columns=techs)
    for hour in hours:
        for tech in techs:
            dispatch[year].at[hour,tech] = generators_dispatch[tech,year,hour].x

for year in years:
    for hour in hours:
        for tech in techs:
            if dispatch[year].at[hour,tech] <= 0:
                dispatch[year].at[hour,tech] = 0
            
dispatch_allyears = pd.concat([dispatch[year] for year in years])  
dispatch_allyears = dispatch_allyears.reset_index()
dispatch_allyears = dispatch_allyears.drop(["index"],axis=1)

fig, ax = plt.subplots()
fig.set_dpi((400))
dispatch[2032].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("day")
ax.set_ylabel("Gross electricity generation [GWh]")
plt.title('2032')
plt.xticks(ticks=[0,111/14,2*111/14,3*111/14,4*111/14,5*111/14,6*111/14,7*111/14,8*111/14,9*111/14,10*111/14,11*111/14,12*111/14,13*111/14],labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Figures_2whighres/{}-dispatch.png".format(filename),transparent=True)

fig, ax = plt.subplots(figsize = [12, 4], dpi = 400)
dispatch_allyears.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("day")
plt.xticks(ticks=[0, 1221/ 11,2* 1221/ 11,3* 1221/ 11,4* 1221/ 11,5* 1221/ 11,6* 1221/ 11,7* 1221/ 11,8* 1221/ 11,9* 1221/ 11,10* 1221/ 11],labels=['2020','2023','2026','2029','2032','2035','2038','2041','2044','2047','2050'])
ax.set_ylabel("Gross electricity generation [GWh]")
# plt.title('2050')
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_2whighres/{}-dispatch.png".format(filename),transparent=True)




capacities = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = generators[tech,year].x
    # capacities.at[year,"battery_store"] = model.storage["battery_store", year].value
    # capacities.at[year,"hydrogen_storage"] = model.storage["battery_store", year].value

fig, ax = plt.subplots()
fig.set_dpi((400))
capacities.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("Installed capacity [GW]")
# ax.set_ylim([0,10000])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_2whighres/{}-capacity.png".format(filename),transparent=True)

build_years = pd.DataFrame(0.,index=years,columns=techs) # +storage
for year in years:
    for tech in techs:
        build_years.at[year,tech] = generators_built[tech,year].x
        
for year in years:
    for tech in techs:
        if build_years.at[year,tech] <= 0:
            build_years.at[year,tech] = 0
            
# # for year in years:
# #     for tech in storage:
# #         build_years.at[year,tech] = model.storage_built[tech, year].value

fig, ax = plt.subplots()
fig.set_dpi((400))
build_years.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("new capacity built [GW]")
# ax.set_ylim([0,250])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_2whighres/{}-new_capacity.png".format(filename),transparent=True)

dispatch_sum = {}
lcoe_capfactor = {}
for year in years:
        for tech in techs:
                dispatch_sum[year] = dispatch[year].sum(0)*26*3
                lcoe_capfactor[tech,year] = dispatch_sum[year].at[tech]/(capacities.at[year,tech]*8760)


level_cost = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
        for tech in techs:
            level_cost.at[year,tech] = fixed_costs[tech,year].x/(8760*lcoe_capfactor[tech,year])/1000 + parameters.at["marginal cost",tech]
            if capacities.at[year,tech]<1: 
                level_cost.at[year,tech] = fixed_costs[tech,year].x/(8760*1)/1000 + parameters.at["marginal cost",tech]
        
                
        # for tech in storage:
        #     LCOE.at[year,tech] = model.fixed_costs_storage[tech, year].value/8760. + store_param.at["marginal cost",tech]
# https://www.nrel.gov/analysis/tech-lcoe-documentation.html

fig, ax = plt.subplots()
fig.set_dpi(400)
level_cost.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("LCOE [EUR/MWh]")
# ax.set_yscale("log")
# ax.set_ylim([0,130])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_2whighres/{}-lcoe.png".format(filename),transparent=True)


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
    storage_discharge[year] = pd.DataFrame(0.,index=hours,columns=storage)
    storage_charge[year] = pd.DataFrame(0.,index=hours,columns=storage)
    energy_stored[year] = pd.DataFrame(0.,index=hours,columns=storage)


    for hour in hours:
        for tech in storage:
            storage_discharge[year].at[hour,tech] = model.storage_discharge[tech,year,hour].value
            storage_charge[year].at[hour,tech] = model.storage_charge[tech,year,hour].value
            energy_stored[year].at[hour,tech] = model.stored_energy[tech,year,hour].value

storage_cap = pd.DataFrame(0.,index=years,columns=storage)
for year in years:
    for tech in storage:
        storage_cap.at[year,tech] = model.storage[tech,year].value

# for year in years:
#     for hour in hours:
#         for tech in storage:
#             if storage_discharge[year].at[hour,tech] <= 0:
#                 storage_discharge[year].at[hour,tech] = 0
            
        
fig, ax = plt.subplots()
fig.set_dpi((400))
energy_stored[2035].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("days")
ax.set_ylabel("Energy stored [GWh]")
plt.title('2032')
plt.xticks(ticks=[0,111/14,2*111/14,3*111/14,4*111/14,5*111/14,6*111/14,7*111/14,8*111/14,9*111/14,10*111/14,11*111/14,12*111/14,13*111/14],labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)

fig, ax = plt.subplots()
fig.set_dpi((400))
storage_cap.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("year")
ax.set_ylabel("Storage capacity [GWh]")
plt.title('Storage capacity')
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)



# #%%
# #Save data
# # build_years.to_pickle("build_years_one_factor.pkl")
# # capacities.to_pickle("capacities_one_factor.pkl")


# # build_years.to_pickle("build_years_multi.pkl")
# # capacities.to_pickle("capacities_multi.pkl")


executionTime = (time.time() - startTime)
print('Solving model in seconds: ' + str(executionTime))
