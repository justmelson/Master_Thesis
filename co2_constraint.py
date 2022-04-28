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
import requests
import io
from annuity_fun import annuity
import pickle
import matplotlib.pyplot as plt

plt.style.use("seaborn")


#%% Scenarios and settings

scenario = "no_co2-no_learning"
# scenario = "co2-0p2-no_learning"
# scenario = "co2-0p2-learning"
# scenario = "no_co2-learning"

# learning_scenario = "high_learning"
# learning_scenario = "low_learning"
learning_scenario = "nom_learning"

# Do you want to include capacity factor?
Renewable_balancing = True

# CO2 budget for 2050 global warming goals
co2_until_2050 = 1e10  # 10 gigaton CO2

# Greenfield scenario 
Greenfield = True


# legend on/off when plotting
lgnd = True

r = 0.0 # discount rate


#%%
# Downloading the csv files from pypsa GitHub account

url="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2020.csv"


# costs = []

# link = url[i] # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

# Reading the downloaded content and turning it into a pandas dataframe

costs = pd.read_csv(io.StringIO(download.decode('utf-8')),index_col=[0,1]).sort_index()
#correct units to MW and EUR
costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
# costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

#min_count=1 is important to generate NaNs which are then filled by fillna
costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
# costs = costs.fillna({"CO2 intensity" : 0,
#                       "FOM" : 0,
#                       "VOM" : 0,
#                       "discount rate" : discount_rate,
#                       "efficiency" : 1,
#                       "fuel" : 0,
#                       "investment" : 0,
#                       "lifetime" : lifetime
# })
# Printing out the first 5 rows of the dataframe

  #print (costs[6].head())

r = 0.07 # discount rate
fuel_cost_gas = 100 # in €/MWh_th from  https://tradingeconomics.com/commodity/eu-natural-gas


#%% Dataframe init

techs = ["offshore_wind","onshore_wind","solar_PV", "CCGT","OCGT","coal","nuclear"]
fossil_techs = ["CCGT","OCGT","coal"]
renewables = ["offshore_wind","onshore_wind","solar_PV"]
wind = ["offshore_wind","onshore_wind"]
colors = ["dodgerblue","lightgreen","gold", 'coral',"peru","grey","plum"]
parameters = pd.DataFrame(columns=techs)
storage = ["battery_store","battery_inverter","hydrogen_storage","electrolysis","fuel_cell"]
color_storage = ["salmon","magenta","aqua","chartreuse","chocolate"]
store_param = pd.DataFrame(columns=storage)
demand = pd.DataFrame(columns= ["demand"])

#%% Technology data
parameters.loc["capacity factor"] = [0.52,0.44,0.21,0.63,0.63,0.83,0.85]
parameters.loc["current capital cost"] = [annuity(costs.at['offwind','lifetime'],r)*costs.at['offwind','investment']*(1+costs.at['offwind','FOM']),
                                     annuity(costs.at['onwind','lifetime'],r)*costs.at['onwind','investment']*(1+costs.at['onwind','FOM']),
                                     annuity(costs.at['solar','lifetime'],r)*costs.at['solar','investment']*(1+costs.at['solar','FOM']),
                                     annuity(costs.at['CCGT','lifetime'],r)*costs.at['CCGT','investment']*(1+costs.at['CCGT','FOM']),
                                     annuity(costs.at['OCGT','lifetime'],r)*costs.at['OCGT','investment']*(1+costs.at['OCGT','FOM']),
                                     (annuity(costs.at['coal','lifetime'],r)*costs.at['coal','investment']*(1+costs.at['coal','FOM']))/10,
                                     annuity(costs.at['nuclear','lifetime'],r)*costs.at['nuclear','investment']*(1+costs.at['nuclear','FOM'])] # EUR/MW/a
# parameters.loc["potential capital cost"] = [annuity(costs[6]['value'][408],r)*costs[6]['value'][407]*1000*(1+costs[6]['value'][405]),
#                                      annuity(costs[6]['value'][425],r)*costs[6]['value'][424]*1000*(1+costs[6]['value'][422]),
#                                      (annuity(costs[6]['value'][437],r)*costs[6]['value'][436]*1000*(1+costs[6]['value'][434])),
#                                      annuity(costs[6]['value'][9],r)*costs[6]['value'][8]*1000*(1+costs[6]['value'][3]),
#                                      annuity(costs[6]['value'][140],r)*costs[6]['value'][139]*1000*(1+costs[6]['value'][136]),
#                                      annuity(costs[6]['value'][274],r)*costs[6]['value'][273]*1000*(1+costs[6]['value'][269]),
#                                      annuity(costs[6]['value'][404],r)*costs[6]['value'][403]*1000*(1+costs[6]['value'][399])] # EUR/MW/a

for tech in techs:
    parameters.at["base cost",tech] = parameters.at['current capital cost',tech]*0.2
    
parameters.loc["learning rate"] = [0.19,0.32,0.47,0.34,0.15,0.083,0] # [0.12,0.12,0.23,0.14,0.15]
parameters.loc["learning parameter"] = [0,0,0,0,0,0,0]
parameters.loc["marginal cost"] = [0,
                                   0,
                                   0,
                                   fuel_cost_gas/costs.at['CCGT','efficiency'],
                                   fuel_cost_gas/costs.at['OCGT','efficiency'],
                                   costs.at['coal','fuel']/costs.at['coal','efficiency'],
                                   26.15] # from lazard #EUR/MWhel
parameters.loc["specific emissions"] = [0.,0.,0.,0.374,0.588,0.76,0] #tcO2/MWhel
parameters.loc["lifetime"] = [27,27,32.5,25,25,40,40]  #years
parameters.loc["existing age"] = [10,10,5,14,14,20,15] # [0,0,0,0,0,0] years
parameters.loc["existing capacity"] = [25,174,100,200,200,128,120.6] #[26,174,123,112,112,128] #[0,0,0,0,0,0] #GW
parameters.loc["current LCOE"] = parameters.loc["current capital cost"]/8760 + parameters.loc["marginal cost"]
# parameters.loc["potential LCOE"] = parameters.loc["potential capital cost"]/8760 + parameters.loc["marginal cost"]
parameters.round(3)

store_param.loc["current capital cost"] = [annuity(costs.at['battery storage','lifetime'],r)*301*1000,
                                      annuity(costs.at['battery inverter','lifetime'],r)*costs.at['battery inverter','investment']*(1+costs.at['battery inverter','FOM']),
                                      annuity(costs.at['H2 (l) storage tank','lifetime'],r)*costs.at['H2 (l) storage tank','investment']*(1+costs.at['H2 (l) storage tank','FOM']),
                                      annuity(costs.at['electrolysis','lifetime'],r)*costs.at['electrolysis','investment']*(1+costs.at['electrolysis','FOM']),
                                      annuity(costs.at['fuel cell','lifetime'],r)*costs.at['fuel cell','investment']*(1+costs.at['fuel cell','FOM'])] # EUR/MW/a
# store_param.loc["potential capital cost"] = [annuity(costs[6]['value'][165],r)*costs[6]['value'][164]*1000,
#                                       annuity(costs[6]['value'][163],r)*costs[6]['value'][162]*1000*(1+costs[6]['value'][160]),
#                                       annuity(costs[6]['value'][365],r)*costs[6]['value'][364]*1000*(1+costs[6]['value'][363]),
#                                       annuity(costs[6]['value'][330],r)*costs[6]['value'][329]*1000*(1+costs[6]['value'][327]),
#                                       annuity(costs[6]['value'][335],r)*costs[6]['value'][334]*1000*(1+costs[6]['value'][331])] # EUR/MW/a]# EUR/MW/a
store_param.loc["learning rate"] = [0.12,0.1,0.1,0.18,0.18] # not sure about inverter learning rate
store_param.loc["learning parameter"] = [0,0,0,0,0]
store_param.loc["marginal cost"] = [0.,0.,0.,0.,0.] #EUR/MWhel
store_param.loc["specific emissions"] = [0.,0.,0.,0.,0.] #tcO2/MWhel
store_param.loc["lifetime"] = [30,10,20,25,10]  #years
store_param.loc["existing age"] = [0,0,0,0,0] #years
store_param.loc["existing capacity"] = [0,0,0,0,0] #[20,20,20,20,20] #[25,195,141,172] #GW

store_param.loc["current LCOE"] = store_param.loc["current capital cost"]/8760 + store_param.loc["marginal cost"]
# store_param.loc["potential LCOE"] = store_param.loc["potential capital cost"]/8760 + store_param.loc["marginal cost"]
# store_param.loc["bLR"] = [0,0,0,0,0]




# Balancing of renewables:
if Renewable_balancing is True:
    for tech in techs:
        parameters.at["current capital cost",tech] = (parameters.at["current capital cost",tech]/parameters.at["capacity factor",tech]) 
        # parameters.at["current LCOE",tech] = (parameters.at["current LCOE",tech]/parameters.at["capacity factor",tech])
        # parameters.at["potential capital cost",tech] = (parameters.at["potential capital cost",tech]/parameters.at["capacity factor",tech]) 

parameters.loc["current LCOE"] = parameters.loc["current capital cost"]/8760 + parameters.loc["marginal cost"]


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
MWh_total = demand['demand'].sum()*1000*8760


if "no_learning" in scenario:
    parameters.loc["learning rate"] = 0
    store_param.loc["learning rate"] = 0
    print("No learning")
else:
    if "high_learning" in learning_scenario:
        parameters.loc["learning rate"] = [0.12,0.12,0.23,0.14,0.15,0.05,0.06] # [0.19,0.32,0.47,0.34,0.15,0.083]
        store_param.loc["learning rate"] = [0.18,0.1,0.1,0.26,0.21]
        print("High learning rates")
    else: 
        if "low_learning" in learning_scenario:
            parameters.loc["learning rate"] = [0.05,0,0.1,0,0.15,0.06,0.0] # [0.05,0,0.1,-0.01,0.15,0.06,-0.06]
            store_param.loc["learning rate"] = [0.08,0.1,0.1,0.18,0.15]
            print("Low learning rates")
        else:
            # nom learning
            parameters.loc["learning rate"] = [0.12,0.12,0.23,0.14,0.15,0.083,0.0] # [0.05,0,0.1,-0.01,0.15,0.06,-0.06]
            store_param.loc["learning rate"] = [0.08,0.1,0.1,0.18,0.15]
            print("Nominal learning rates")


# Calculating learning parameter gamma
for i in range(len(techs)):
    parameters.loc["learning parameter"][i] = math.log(1/(1-parameters.loc["learning rate"][i])) / math.log(2)
for i in range(len(storage)):
    store_param.loc["learning parameter"][i] = math.log(1/(1-store_param.loc["learning rate"][i])) / math.log(2)


# # carbon budget in average tCO2   
# if "no_co2" in scenario:
#     co2_budget = 1e30
#     print("No CO2 budget")
# else:
#     co2_budget = co2_until_2050 # [tCO2] 10 Gigatons CO2
#     print("CO2 budget of "+ str(co2_until_2050) + " tons CO2")



# carbon budget in average tCO2/MWh_el    
if "no_co2" in scenario:
    co2_budget = 1e30
    print("No CO2 budget")

else:
    co2_budget = co2_until_2050#/MWh_total
    print("CO2 budget of "+ str(co2_budget) + " tCO2")


    
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
results = opt.solve(model,suffixes=['dual'],keepfiles=False)

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
fig.savefig("CO2_val_figures/{}-dispatch.png".format(filename),transparent=True)


capacities = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = model.generators[tech,year].value
        if capacities.at[year,tech] <= 0:
            capacities.at[year,tech] = 0
        
    # capacities.at[year,"battery_store"] = model.storage["battery_store", year].value
    # capacities.at[year,"hydrogen_storage"] = model.storage["battery_store", year].value

        
fig, ax = plt.subplots()
fig.set_dpi((400))
capacities.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("Year")
ax.set_ylabel("Installed capacity [GW]")
ax.set_ylim([0,1500])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("CO2_val_figures/{}-capacity.png".format(filename),transparent=True)

build_years = pd.DataFrame(0.,index=years,columns=techs) # +storage
for year in years:
    for tech in techs:
        build_years.at[year,tech] = model.generators_built[tech,year].value
        if build_years.at[year,tech] <= 0:
            build_years.at[year,tech] = 0
        
# for year in years:
#     for tech in storage:
#         build_years.at[year,tech] = model.storage_built[tech, year].value

fig, ax = plt.subplots()
fig.set_dpi((400))
build_years.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("new capacity built [GW]")
ax.set_ylim([0,250])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("CO2_val_figures/{}-new_capacity.png".format(filename),transparent=True)


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
fig.savefig("CO2_val_figures/{}-lcoe.png".format(filename),transparent=True)


emissions = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        emissions.at[year,tech] = model.generators_dispatch[tech,year].value*8760* 1000 * parameters.at["specific emissions",tech]

fig, ax = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
emissions.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("CO2 [t]")

# ax.set_yscale("log")
# ax.set_ylim([0,40])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Figures_LR_test/{}-emissions.png".format(filename),transparent=True)

#%%
#Save data
build_years.to_pickle("build_years_one_factor.pkl")
capacities.to_pickle("capacities_one_factor.pkl")


# build_years.to_pickle("build_years_multi.pkl")
# capacities.to_pickle("capacities_multi.pkl")
