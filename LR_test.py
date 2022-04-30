#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 11:35:11 2022

@author: frederikmelson
"""

"Simplified scenario to test endogenous learning rate implementation"

#%% Packages

from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Suffix, exp, value
from pyomo.opt import SolverFactory
import pandas as pd
import math

import pickle
import matplotlib.pyplot as plt

plt.style.use("seaborn")


#%% Scenarios and settings

scenario = "no_co2-no_learning"
# scenario = "co2-0p2-no_learning"
# scenario = "co2-0p2-learning"
# scenario = "no_co2-learning"

learning_scenario = "high_learning"
# learning_scenario = "low_learning"
# learning_scenario = "nom_learning"


# Do you want to include capacity factor?
Renewable_balancing = True

# CO2 budget for 2050 global warming goals
co2_until_2050 = 10000000000 # 100 million tCO2 ,10000000000 # 10 gigaton CO2

# Greenfield scenario 
Greenfield = True


# legend on/off when plotting
lgnd = True

r = 0.01 # discount rate


parameters = pd.read_pickle("parameters.pkl")
store_param = pd.read_pickle("store_param.pkl")
demand = pd.DataFrame(columns= ["demand"])


techs_file = "techs.pkl"
fossil_techs_file = "fossil_techs.pkl"
renewables_file = "renewables.pkl"
wind_file = "wind.pkl"
colors_file = "colors.pkl"
storage_file = "storage.pkl"
color_storage_file = "color_storage.pkl"

files = [techs_file,fossil_techs_file,renewables_file,wind_file,colors_file,storage_file,color_storage_file]
lists = ["techs","fossil_techs","renewables","wind","colors","storage","color_storage"]


for i in range(len(files)):
    open_file = open(files[i], "rb")
    lists[i] = pickle.load(open_file)
    open_file.close()

techs = lists[0]
fossil_techs = lists[1]
renewables = lists[2]
wind = lists [3]
colors = lists[4]
storage = lists[5]
color_storage = lists[6]



# Balancing of renewables:
if Renewable_balancing is True:
    for tech in techs:
        parameters.at["current capital cost",tech] = (parameters.at["current capital cost",tech]/parameters.at["capacity factor",tech]) 
        # parameters.at["current LCOE",tech] = (parameters.at["current LCOE",tech]/parameters.at["capacity factor",tech])
        # parameters.at["potential capital cost",tech] = (parameters.at["potential capital cost",tech]/parameters.at["capacity factor",tech]) 

# Green- or brownfield scenario:
if Greenfield is True:
    for tech in techs:
        parameters.loc["existing age"] = [0,0,0,0,0,0,0] #years
        parameters.loc["existing capacity"] = [0,0,0,0,0,0,0]
    print("Greenfield approach")
else:
    print("Brownfield approach")

#%% Updating learning rates and CO2 budget

#Currently installed capacities in GW is used to assume current demand

# considered years
years = list(range(2020,2051))
for year in years:
    if year > 2020:
        for i in demand:
            demand.at[year,i] = 8+demand.at[year-1,i]
    else:
        for i in demand:
            demand.at[year,i] = (600) #from EU Energy Outlook 2050


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


#%%
MWh_total = demand['demand'].sum()*1000*8760/50

    
#%% One node model
model = ConcreteModel("discounted total costs")
model.generators = Var(techs, years, within=NonNegativeReals)
model.generators_dispatch = Var(techs, years, within=NonNegativeReals)
model.generators_built = Var(techs,years,within=NonNegativeReals)
model.fixed_costs = Var(techs, years, within=NonNegativeReals)


model.storage = Var(storage,years,within=NonNegativeReals)
model.storage_dispatch = Var(storage, years, within=NonNegativeReals)
model.storage_built = Var(storage,years,within=NonNegativeReals)
model.fixed_costs_storage = Var(storage, years, within=NonNegativeReals)

constant = sum(parameters.at['existing capacity',tech] * parameters.at['current capital cost', tech]/1e6/(1+r)**(year-years[0]) for tech in techs for year in years if year < years[0] + parameters.at['lifetime',tech] - parameters.at['existing age',tech])
print("Cost of existing capacity =", "%.2f"%constant)

model.objective = Objective(expr=constant +
                           sum(model.generators_built[tech,year] * model.fixed_costs[tech,year]/1e6 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                              for year in years
                              for tech in techs) +
                           sum(model.generators_dispatch[tech,year] * parameters.at['marginal cost',tech] * 8760/1e6/(1+r)**(year-years[0])
                              for year in years
                              for tech in techs))

#%% Constraints
def balance_constraint(model,year):
    return demand.at[year,"demand"] == sum(model.generators_dispatch[tech,year] for tech in techs)
model.balance_constraint = Constraint(years, rule=balance_constraint)


# def storebalancePV_constraint(model,year):
#     return model.storage_dispatch["battery_store",year] == model.generators_dispatch["solar_PV",year]*0.3
# model.storebalancePV_constraint = Constraint(years, rule=storebalancePV_constraint)

# def storebalanceWind_constraint(model,tech,year):
#     return model.storage_dispatch["hydrogen_storage",year] == sum(model.generators_dispatch[tech,year] for tech in wind)*0.3
# model.storebalanceWind_constraint = Constraint(renewables,years, rule=storebalanceWind_constraint)

# def storage_constraint(model,tech,year):
#     return model.storage_dispatch[tech,year] <= model.storage[tech,year]
# model.storage_constraint = Constraint(storage, years, rule=storage_constraint)

# def solar_constraint(model,year):
#     return model.generators["solar_PV",year] <= sum(model.generators_dispatch[tech,year] for tech in techs)*0.5
# model.solar_constraint = Constraint(years, rule=solar_constraint)

# def onshore_constraint(model,year):
#     return model.generators["onshore_wind",year] <= sum(model.generators_dispatch[tech,year] for tech in techs)*0.3
# model.onshore_constraint = Constraint(years, rule=onshore_constraint)

def generator_constraint(model,tech,year):
    return model.generators_dispatch[tech,year] <= model.generators[tech,year] #*parameters.at["capacity factor",tech] # Including capacity factors 
model.generator_constraint = Constraint(techs, years, rule=generator_constraint)

    
def co2_constraint(model,tech,year):
    return co2_budget >= sum((model.generators_dispatch[tech,year] * 8760 * 1000 * parameters.at["specific emissions",tech]) for tech in techs for year in years)
model.co2_constraint = Constraint(techs,years,rule=co2_constraint)

# def inverter_constraint(model,tech,year):
#     return model.storage_dispatch["battery_store",year] == model.storage_dispatch["battery_inverter",year]
# model.inverter_constraint = Constraint(storage, years, rule=inverter_constraint)

# def fuelcell_constraint(model,tech,year):
#     return model.storage_dispatch["hydrogen_storage",year] == model.storage_dispatch["fuel_cell",year]
# model.fuelcell_constraint = Constraint(storage, years, rule=fuelcell_constraint)

# def electrolysis_constraint(model,tech,year):
#     return model.storage_dispatch["hydrogen_storage",year] == model.storage_dispatch["electrolysis",year]
# model.electrolysis_constraint = Constraint(storage, years, rule=electrolysis_constraint)

def build_years(model,tech,year):
    if year < years[0] + parameters.at["lifetime",tech] - parameters.at["existing age",tech]:
        constant = parameters.at["existing capacity",tech]
    else:
        constant = 0.
    
    return model.generators[tech,year] == constant + sum(model.generators_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + parameters.at["lifetime",tech])))
model.build_years = Constraint(techs, years, rule=build_years)

# def build_years_storage(model,tech,year):
#     if year < years[0] + store_param.at["lifetime",tech] - store_param.at["existing age",tech]:
#         constant = store_param.at["existing capacity",tech]
#     else:
#         constant = 0.
    
#     return model.storage[tech,year] == constant + sum(model.storage_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + store_param.at["lifetime",tech])))
# model.build_years_storage = Constraint(storage, years, rule=build_years_storage)

def fixed_cost_constraint(model,tech,year):
    if parameters.at["learning parameter",tech] == 0:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]
    else:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech] * (1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
        # return model.fixed_costs[tech,year] == parameters.at["base cost",tech] + (parameters.at["current capital cost",tech]-parameters.at["base cost",tech])*(1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
model.fixed_cost_constraint = Constraint(techs, years, rule=fixed_cost_constraint)

# def fixed_cost_constraint_storage(model,tech,year):
#     if store_param.at["learning parameter",tech] == 0:
#         return model.fixed_costs_storage[tech,year] == store_param.at["current capital cost",tech]
#     else:
#         return model.fixed_costs_storage[tech,year] == store_param.at["current capital cost",tech] * (1+sum(model.storage_built[tech,yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter",tech])
#         # return model.fixed_costs_storage[tech,year] == store_param.at["potential capital cost",tech] + (store_param.at["current capital cost",tech]-store_param.at["potential capital cost",tech])*(1+sum(model.storage[tech,yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter",tech])
# model.fixed_cost_constraint_storage = Constraint(storage, years, rule=fixed_cost_constraint_storage)

# def renewable_constraint(model,tech,techren,year):
#     if value(sum(model.generators_dispatch[techren,year] for techren in renewables)) > value(sum(model.generators_dispatch[tech,year] for tech in techs)*0.7):
#         return model.storage_dispatch["battery_store",year] == sum(model.generators_dispatch[techren,year] for tech in renewables)*0.3
        
# model.renewable_constraint = Constraint(techs,renewables, years, rule=renewable_constraint)

#%% Solving model

opt = SolverFactory('ipopt')
results = opt.solve(model,suffixes=['dual'],keepfiles=False,tee=True)

print("Total cost (in billion euro) =","%.2f"% model.objective())

systemcost = model.objective()*1e9/MWh_total # euro/MWh
print("Avg. cost (in euro/MWh) =","%.2f"% systemcost)


#%% Plotting

# colormap = "Set2"


# file name
if "no_learning" in scenario: 
    filename = scenario+"LR"
else:
    filename = scenario+"_"+learning_scenario+"LR"


dispatch = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        dispatch.at[year,tech] = model.generators_dispatch[tech,year].value*8760


for year in years:
        for tech in techs:
            if dispatch.at[year,tech] <= 0:
                dispatch.at[year,tech] = 0

            
# for year in years:
# #     for tech in storage:
#     dispatch.at[year,"battery_store"] = model.storage_dispatch["battery_store", year].value*8760
#     dispatch.at[year,"hydrogen_storage"] = model.storage_dispatch["hydrogen_storage", year].value*8760
fig, ax = plt.subplots()
fig.set_dpi((400))
dispatch.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("year")
ax.set_ylabel("Gross electricity generation [GWh]")
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_LR_test/{}-dispatch.png".format(filename),transparent=True)


capacities = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = model.generators[tech,year].value
    # capacities.at[year,"battery_store"] = model.storage["battery_store", year].value
    # capacities.at[year,"hydrogen_storage"] = model.storage["battery_store", year].value
for year in years:
        for tech in techs:
            if capacities.at[year,tech] <= 0:
                capacities.at[year,tech] = 0

        
fig, ax = plt.subplots()
fig.set_dpi((400))
capacities.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("Year")
ax.set_ylabel("Installed capacity [GW]")
ax.set_ylim([0,1500])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_LR_test/{}-capacity.png".format(filename),transparent=True)

build_years = pd.DataFrame(0.,index=years,columns=techs) # +storage
for year in years:
    for tech in techs:
        build_years.at[year,tech] = model.generators_built[tech,year].value
for year in years:
        for tech in techs:
            if build_years.at[year,tech] <= 0:
                build_years.at[year,tech] = 0


fig, ax = plt.subplots()
fig.set_dpi((400))
build_years.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("new capacity built [GW]")
ax.set_ylim([0,250])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_LR_test/{}-new_capacity.png".format(filename),transparent=True)


level_cost = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        level_cost.at[year,tech] = model.fixed_costs[tech,year].value/8760. + parameters.at["marginal cost",tech]
    # for tech in storage:
    #     LCOE.at[year,tech] = model.fixed_costs_storage[tech, year].value/8760. + store_param.at["marginal cost",tech]


fig, ax = plt.subplots()
fig.set_dpi(400)
level_cost.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("LCOE [EUR/MWh]")
# ax.set_yscale("log")
ax.set_ylim([0,130])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures_LR_test/{}-lcoe.png".format(filename),transparent=True)


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
# fig.savefig("Figures_LR_test/{}-emissions.png".format(filename),transparent=True)

#%%
#Save data
build_years.to_pickle("build_years_one_factor.pkl")
capacities.to_pickle("capacities_one_factor.pkl")


# build_years.to_pickle("build_years_multi.pkl")
# capacities.to_pickle("capacities_multi.pkl")
