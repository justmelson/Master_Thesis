#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:58:50 2022

@author: frederikmelson
"""



#%% Packages
import math
from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Suffix, exp, value
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

plt.style.use("seaborn")


#%% Scenarios and settings

# scenario = "no_co2-no_learning"
# scenario = "co2-0p2-no_learning"
# scenario = "co2-0p2-learning"
scenario = "no_co2-learning"

# learning_scenario = "high_learning"
# learning_scenario = "low_learning"
learning_scenario = "nom_learning"

# Storage initialization
PowerStore = np.ones(111)
Powerstore = pd.DataFrame(PowerStore)

# CO2 budget for 2050 global warming goals
co2_until_2050 = 1e10 # 100 million tCO2 ,10000000000 # 10 gigaton CO2

# Greenfield scenario 
Greenfield = True


# legend on/off when plotting
lgnd = True

r = 0.00 # discount rate

hours = list(range(111))


parameters  = pd.read_pickle("parameters.pkl")
store_param = pd.read_pickle("store_param.pkl")
demand = pd.read_pickle("df_elec.pkl")
demand.reset_index()
demand.drop(columns=["utc_time"])
demand = demand/1000

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
storage         = lists[5]
color_storage   = lists[6]


# Green- or brownfield scenario:
if Greenfield is True:
    for tech in techs:
        parameters.loc["existing age"]      = [0,0,0,0,0,0,0] #years
        parameters.loc["existing capacity"] = [0,0,0,0,0,0,0]
    print("Greenfield approach")
else:
    print("Brownfield approach")
    
# parameters.at["current capital cost","coal"] *= 0.5

#%% Updating learning rates and CO2 budget

#Currently installed capacities in GW is used to assume current demand

# Hourly demand
years = [2020,2023,2026,2029,2032,2035,2038,2041,2044,2047,2050]

# for year in range(7):
#     if year > 2020:
#         for i in demand:
#             demand.at[year,i] = 8+demand.at[year-1,i]
#     else:
#         for i in demand:
#             demand.at[year,i] = (600) #from EU Energy Outlook 2050


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
# MWh_total = demand['demand'].sum()*1000*8760/50

    
#%% One node model
model = ConcreteModel()
model.generators            = Var(techs, years, within=NonNegativeReals)
model.generators_dispatch   = Var(techs, years, hours, within=NonNegativeReals)
model.generators_built      = Var(techs, years, within=NonNegativeReals)
model.fixed_costs           = Var(techs, years, within=NonNegativeReals)




#Value of currently installed technology:
constant = sum(parameters.at['existing capacity',tech] * parameters.at['current capital cost', tech]*1000/1e9/(1+r)**(hour-hours[0]) for tech in techs for hour in hours if hour < hours[0] + parameters.at['lifetime',tech] - parameters.at['existing age',tech])
print("Cost of existing capacity =", "%.2f"%constant)


model.objective = Objective(expr=constant +
                            sum(model.generators_built[tech,year] * model.fixed_costs[tech,year]/1e9 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                              for tech in techs for year in years)+
                            sum(model.generators_dispatch[tech,2020,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2020-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2023,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2023-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2026,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2026-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2029,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2029-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2032,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2032-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2035,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2035-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2038,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2038-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2041,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2041-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2044,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2044-years[0])
                                for hour in hours
                                for tech in techs)+ 
                            sum(model.generators_dispatch[tech,2047,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2047-years[0])
                                for hour in hours
                                for tech in techs)+
                            sum(model.generators_dispatch[tech,2050,hour] * parameters.at['marginal cost',tech]*1000*26*3 / 1e9 /(1+r)**(2050-years[0])
                                for hour in hours
                                for tech in techs))


#%% Balance Constraints

def balance_constraint2020(model,hour): # GWh
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2020, hour] for tech in techs) 
model.balance_constraint2020 = Constraint(hours, rule=balance_constraint2020)

def balance_constraint2023(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2023, hour] for tech in techs)
model.balance_constraint2023 = Constraint( hours, rule=balance_constraint2023)

def balance_constraint2026(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2026, hour] for tech in techs)
model.balance_constraint2026 = Constraint(hours, rule=balance_constraint2026)

def balance_constraint2029(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2029, hour] for tech in techs)
model.balance_constraint2029 = Constraint(hours, rule=balance_constraint2029)

def balance_constraint2032(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2032, hour] for tech in techs)
model.balance_constraint2032 = Constraint(hours, rule=balance_constraint2032)

def balance_constraint2035(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2035, hour] for tech in techs)
model.balance_constraint2035 = Constraint(hours, rule=balance_constraint2035)

def balance_constraint2038(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2038, hour] for tech in techs)
model.balance_constraint2038 = Constraint(hours, rule=balance_constraint2038)

def balance_constraint2041(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2041, hour] for tech in techs)
model.balance_constraint2041 = Constraint(hours, rule=balance_constraint2041)

def balance_constraint2044(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2044, hour] for tech in techs)
model.balance_constraint2044 = Constraint(hours, rule=balance_constraint2044)

def balance_constraint2047(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2047, hour] for tech in techs)
model.balance_constraint2047 = Constraint(hours, rule=balance_constraint2047)

def balance_constraint2050(model,hour):
    return demand2w3h.at[hour,0] == sum(model.generators_dispatch[tech, 2050, hour] for tech in techs)
model.balance_constraint2050 = Constraint(hours, rule=balance_constraint2050)


#%% Solar capacity constraints

def solar_constraint2020(model,year,hour):
    return model.generators_dispatch["solar_PV",2020,hour] <= model.generators["solar_PV",2020]*Cf_solar.at[hour,0]
model.solar_constraint2020 = Constraint(years, hours, rule=solar_constraint2020)

def solar_constraint2023(model,year,hour):
    return model.generators_dispatch["solar_PV",2023,hour] <= model.generators["solar_PV",2023]*Cf_solar.at[hour,0]
model.solar_constraint2023 = Constraint(years, hours, rule=solar_constraint2023)

def solar_constraint2026(model,year,hour):
    return model.generators_dispatch["solar_PV",2026,hour] <= model.generators["solar_PV",2026]*Cf_solar.at[hour,0]
model.solar_constraint2026 = Constraint(years, hours, rule=solar_constraint2026)

def solar_constraint2029(model,year,hour):
    return model.generators_dispatch["solar_PV",2029,hour] <= model.generators["solar_PV",2029]*Cf_solar.at[hour,0]
model.solar_constraint2029 = Constraint(years, hours, rule=solar_constraint2029)

def solar_constraint2032(model,year,hour):
    return model.generators_dispatch["solar_PV",2032,hour] <= model.generators["solar_PV",2032]*Cf_solar.at[hour,0]
model.solar_constraint2032 = Constraint(years, hours, rule=solar_constraint2032)

def solar_constraint2035(model,year,hour):
    return model.generators_dispatch["solar_PV",2035,hour] <= model.generators["solar_PV",2035]*Cf_solar.at[hour,0]
model.solar_constraint2035 = Constraint(years, hours, rule=solar_constraint2035)

def solar_constraint2038(model,year,hour):
    return model.generators_dispatch["solar_PV",2038,hour] <= model.generators["solar_PV",2038]*Cf_solar.at[hour,0]
model.solar_constraint2038 = Constraint(years, hours, rule=solar_constraint2038)

def solar_constraint2041(model,year,hour):
    return model.generators_dispatch["solar_PV",2041,hour] <= model.generators["solar_PV",2041]*Cf_solar.at[hour,0]
model.solar_constraint2041 = Constraint(years, hours, rule=solar_constraint2041)

def solar_constraint2044(model,year,hour):
    return model.generators_dispatch["solar_PV",2044,hour] <= model.generators["solar_PV",2044]*Cf_solar.at[hour,0]
model.solar_constraint2044 = Constraint(years, hours, rule=solar_constraint2044)

def solar_constraint2047(model,year,hour):
    return model.generators_dispatch["solar_PV",2047,hour] <= model.generators["solar_PV",2047]*Cf_solar.at[hour,0]
model.solar_constraint2047 = Constraint(years, hours, rule=solar_constraint2047)

def solar_constraint2050(model,year,hour):
    return model.generators_dispatch["solar_PV",2050,hour] <= model.generators["solar_PV",2050]*Cf_solar.at[hour,0]
model.solar_constraint2050 = Constraint(years, hours, rule=solar_constraint2050)

#%% Onshore wind capacity constraints


def onshore_constraint2020(model,year,hour):
    return model.generators_dispatch["onshore_wind",2020,hour] <= model.generators["onshore_wind",2020]*Cf_onshore.at[hour,0]
model.onshore_constraint2020 = Constraint(years, hours, rule=onshore_constraint2020)

def onshore_constraint2023(model,year,hour):
    return model.generators_dispatch["onshore_wind",2023,hour] <= model.generators["onshore_wind",2023]*Cf_onshore.at[hour,0]
model.onshore_constraint2023 = Constraint(years, hours, rule=onshore_constraint2023)

def onshore_constraint2026(model,year,hour):
    return model.generators_dispatch["onshore_wind",2026,hour] <= model.generators["onshore_wind",2026]*Cf_onshore.at[hour,0]
model.onshore_constraint2026 = Constraint(years, hours, rule=onshore_constraint2026)

def onshore_constraint2029(model,year,hour):
    return model.generators_dispatch["onshore_wind",2029,hour] <= model.generators["onshore_wind",2029]*Cf_onshore.at[hour,0]
model.onshore_constraint2029 = Constraint(years, hours, rule=onshore_constraint2029)

def onshore_constraint2032(model,year,hour):
    return model.generators_dispatch["onshore_wind",2032,hour] <= model.generators["onshore_wind",2032]*Cf_onshore.at[hour,0]
model.onshore_constraint2032 = Constraint(years, hours, rule=onshore_constraint2032)

def onshore_constraint2035(model,year,hour):
    return model.generators_dispatch["onshore_wind",2035,hour] <= model.generators["onshore_wind",2035]*Cf_onshore.at[hour,0]
model.onshore_constraint2035 = Constraint(years, hours, rule=onshore_constraint2035)

def onshore_constraint2038(model,year,hour):
    return model.generators_dispatch["onshore_wind",2038,hour] <= model.generators["onshore_wind",2038]*Cf_onshore.at[hour,0]
model.onshore_constraint2038 = Constraint(years, hours, rule=onshore_constraint2038)

def onshore_constraint2041(model,year,hour):
    return model.generators_dispatch["onshore_wind",2041,hour] <= model.generators["onshore_wind",2041]*Cf_onshore.at[hour,0]
model.onshore_constraint2041 = Constraint(years, hours, rule=onshore_constraint2041)

def onshore_constraint2044(model,year,hour):
    return model.generators_dispatch["onshore_wind",2044,hour] <= model.generators["onshore_wind",2044]*Cf_onshore.at[hour,0]
model.onshore_constraint2044 = Constraint(years, hours, rule=onshore_constraint2044)

def onshore_constraint2047(model,year,hour):
    return model.generators_dispatch["onshore_wind",2047,hour] <= model.generators["onshore_wind",2047]*Cf_onshore.at[hour,0]
model.onshore_constraint2047 = Constraint(years, hours, rule=onshore_constraint2047)

def onshore_constraint2050(model,year,hour):
    return model.generators_dispatch["onshore_wind",2050,hour] <= model.generators["onshore_wind",2050]*Cf_onshore.at[hour,0]
model.onshore_constraint2050 = Constraint(years, hours, rule=onshore_constraint2050)

#%% Offshore wind capacity constraints


def offshore_constraint2020(model,year,hour):
    return model.generators_dispatch["offshore_wind",2020,hour] <= model.generators["offshore_wind",2020]*Cf_offshore.at[hour,0]
model.offshore_constraint2020 = Constraint(years, hours, rule=offshore_constraint2020)

def offshore_constraint2023(model,year,hour):
    return model.generators_dispatch["offshore_wind",2023,hour] <= model.generators["offshore_wind",2023]*Cf_offshore.at[hour,0]
model.offshore_constraint2023 = Constraint(years, hours, rule=offshore_constraint2023)

def offshore_constraint2026(model,year,hour):
    return model.generators_dispatch["offshore_wind",2026,hour] <= model.generators["offshore_wind",2026]*Cf_offshore.at[hour,0]
model.offshore_constraint2026 = Constraint(years, hours, rule=offshore_constraint2026)

def offshore_constraint2029(model,year,hour):
    return model.generators_dispatch["offshore_wind",2029,hour] <= model.generators["offshore_wind",2029]*Cf_offshore.at[hour,0]
model.offshore_constraint2029 = Constraint(years, hours, rule=offshore_constraint2029)

def offshore_constraint2032(model,year,hour):
    return model.generators_dispatch["offshore_wind",2032,hour] <= model.generators["offshore_wind",2032]*Cf_offshore.at[hour,0]
model.offshore_constraint2032 = Constraint(years, hours, rule=offshore_constraint2032)

def offshore_constraint2035(model,year,hour):
    return model.generators_dispatch["offshore_wind",2035,hour] <= model.generators["offshore_wind",2035]*Cf_offshore.at[hour,0]
model.offshore_constraint2035 = Constraint(years, hours, rule=offshore_constraint2035)

def offshore_constraint2038(model,year,hour):
    return model.generators_dispatch["offshore_wind",2038,hour] <= model.generators["offshore_wind",2038]*Cf_offshore.at[hour,0]
model.offshore_constraint2038 = Constraint(years, hours, rule=offshore_constraint2038)

def offshore_constraint2041(model,year,hour):
    return model.generators_dispatch["offshore_wind",2041,hour] <= model.generators["offshore_wind",2041]*Cf_offshore.at[hour,0]
model.offshore_constraint2041 = Constraint(years, hours, rule=offshore_constraint2041)

def offshore_constraint2044(model,year,hour):
    return model.generators_dispatch["offshore_wind",2044,hour] <= model.generators["offshore_wind",2044]*Cf_offshore.at[hour,0]
model.offshore_constraint2044 = Constraint(years, hours, rule=offshore_constraint2044)

def offshore_constraint2047(model,year,hour):
    return model.generators_dispatch["offshore_wind",2047,hour] <= model.generators["offshore_wind",2047]*Cf_offshore.at[hour,0]
model.offshore_constraint2047 = Constraint(years, hours, rule=offshore_constraint2047)

def offshore_constraint2050(model,year,hour):
    return model.generators_dispatch["offshore_wind",2050,hour] <= model.generators["offshore_wind",2050]*Cf_offshore.at[hour,0]
model.offshore_constraint2050 = Constraint(years, hours, rule=offshore_constraint2050)




#%% Installed capacity constraints
# def generator_constraint(model,tech,year,hour):
#     return model.generators_dispatch[tech,year,hour] <= model.generators[tech,year] # GW
# model.generator_constraint = Constraint(techs, years, hours, rule=generator_constraint)

def generator_constraint2020(model,tech,hour):
    return model.generators_dispatch[tech,2020,hour] <= model.generators[tech,2020] # GW
model.generator_constraint2020 = Constraint(techs, hours, rule=generator_constraint2020)

def generator_constraint2023(model,tech,hour):
    return model.generators_dispatch[tech,2023,hour] <= model.generators[tech,2023] # GW
model.generator_constraint2023 = Constraint(techs, hours, rule=generator_constraint2023)

def generator_constraint2026(model,tech,hour):
    return model.generators_dispatch[tech,2026,hour] <= model.generators[tech,2026] # GW
model.generator_constraint2026 = Constraint(techs, hours, rule=generator_constraint2026)

def generator_constraint2029(model,tech,hour):
    return model.generators_dispatch[tech,2029,hour] <= model.generators[tech,2029] # GW
model.generator_constraint2029 = Constraint(techs, hours, rule=generator_constraint2029)

def generator_constraint2032(model,tech,hour):
    return model.generators_dispatch[tech,2032,hour] <= model.generators[tech,2032] # GW
model.generator_constraint2032 = Constraint(techs, hours, rule=generator_constraint2032)

def generator_constraint2035(model,tech,hour):
    return model.generators_dispatch[tech,2035,hour] <= model.generators[tech,2035] # GW
model.generator_constraint2035 = Constraint(techs, hours, rule=generator_constraint2035)

def generator_constraint2038(model,tech,hour):
    return model.generators_dispatch[tech,2038,hour] <= model.generators[tech,2038] # GW
model.generator_constraint2038 = Constraint(techs, hours, rule=generator_constraint2038)

def generator_constraint2041(model,tech,hour):
    return model.generators_dispatch[tech,2041,hour] <= model.generators[tech,2041] # GW
model.generator_constraint2041 = Constraint(techs, hours, rule=generator_constraint2041)

def generator_constraint2044(model,tech,hour):
    return model.generators_dispatch[tech,2044,hour] <= model.generators[tech,2044] # GW
model.generator_constraint2044 = Constraint(techs, hours, rule=generator_constraint2044)

def generator_constraint2047(model,tech,hour):
    return model.generators_dispatch[tech,2047,hour] <= model.generators[tech,2047] # GW
model.generator_constraint2047 = Constraint(techs, hours, rule=generator_constraint2047)

def generator_constraint2050(model,tech,hour):
    return model.generators_dispatch[tech,2050,hour] <= model.generators[tech,2050] # GW
model.generator_constraint2050 = Constraint(techs, hours, rule=generator_constraint2050)

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


def fixed_cost_constraint(model,tech,year):
    if parameters.at["learning parameter",tech] == 0:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]*1000 #EUR/GW
    else:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]*1000 * (1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech]) #EUR/GW
        # return model.fixed_costs[tech,year] == parameters.at["base cost",tech] + (parameters.at["current capital cost",tech]-parameters.at["base cost",tech])*(1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
model.fixed_cost_constraint = Constraint(techs, years, rule=fixed_cost_constraint)

#%% CO2 constraint


def co2_constraint(model,tech,year,hour):
    return co2_budget >= sum((model.generators_dispatch[tech,year,hour] * 1000*26*3 * parameters.at["specific emissions",tech]) for tech in techs for hour in hours for year in years)
model.co2_constraint = Constraint(techs,years, hours,rule=co2_constraint)


#%% Solving model
# ipopt #suffixes=['dual'] keepfiles=False mindtpy
opt = SolverFactory('mindtpy')
# results = opt.solve(model,suffixes=['dual'],keepfiles=False)

# results = opt.solve(model,mip_solver='glpk', nlp_solver='ipopt',tee=True)


# results = opt.solve(model,
#                                     strategy='OA',
#                                     mip_solver='cplex',
#                                     nlp_solver='ipopt',
#                                     add_regularization='level_L1'
#                                     # alternative regularizations
#                                     # 'level_L1', 'level_L2', 'level_L_infinity',
#                                     # 'grad_lag', 'hess_lag', 'hess_only_lag', 'sqp_lag'
#                                     )

# results = opt.solve(model,
#                                     strategy='OA',
#                                     init_strategy='FP',
#                                     mip_solver='gurobi',
#                                     nlp_solver='ipopt',
#                                     solution_pool=True,
#                                     num_solution_iteration=30, # default=5
#                                     tee=True
#                                     )

results = opt.solve(model,
                                    strategy='GOA',
                                    mip_solver='cplex',
                                    nlp_solver='ipopt',
                                    tee=True)

model.objective.display()

print("Total cost (in billion euro) =","%.2f"% model.objective())

# systemcost = model.objective()*1e9/MWh_total # euro/MWh
# print("Avg. cost (in euro/MWh) =","%.2f"% systemcost)


#%% Plotting

# # colormap = "Set2"


# # file name
# if "no_learning" in scenario: 
#     filename = scenario+"LR"
# else:
#     filename = scenario+"_"+learning_scenario+"highres"


dispatch = {}
for year in years:
    dispatch[year] = pd.DataFrame(0.,index=hours,columns=techs)
    for hour in hours:
        for tech in techs:
            dispatch[year].at[hour,tech] = model.generators_dispatch[tech,year,hour].value

for year in years:
    for hour in hours:
        for tech in techs:
            if dispatch[year].at[hour,tech] <= 0:
                dispatch[year].at[hour,tech] = 0
            
        
# for year in years:
# #     for tech in storage:
#     dispatch.at[year,"battery_store"] = model.storage_dispatch["battery_store", year].value*8760
#     dispatch.at[year,"hydrogen_storage"] = model.storage_dispatch["hydrogen_storage", year].value*8760
fig, ax = plt.subplots()
fig.set_dpi((400))
dispatch[2032].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("hour")
ax.set_ylabel("Gross electricity generation [GWh]")
plt.title('2032')
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Figures_LR_test/{}-dispatch.png".format(filename),transparent=True)

fig, ax = plt.subplots()
fig.set_dpi((400))
dispatch[2023].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("hour")
ax.set_ylabel("Gross electricity generation [GWh]")
plt.title('2023')
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Figures_LR_test/{}-dispatch.png".format(filename),transparent=True)


capacities = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = model.generators[tech,year].value
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
# fig.savefig("Figures_LR_test/{}-capacity.png".format(filename),transparent=True)

build_years = pd.DataFrame(0.,index=years,columns=techs) # +storage
for year in years:
    for tech in techs:
        build_years.at[year,tech] = model.generators_built[tech,year].value
        
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
# fig.savefig("Figures_LR_test/{}-new_capacity.png".format(filename),transparent=True)


level_cost = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
        for tech in techs:
            level_cost.at[year,tech] = model.fixed_costs[tech,year].value/8760/1000 + parameters.at["marginal cost",tech]
        # for tech in storage:
        #     LCOE.at[year,tech] = model.fixed_costs_storage[tech, year].value/8760. + store_param.at["marginal cost",tech]


fig, ax = plt.subplots()
fig.set_dpi(400)
level_cost.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("hour")
ax.set_ylabel("LCOE [EUR/MWh]")
# ax.set_yscale("log")
# ax.set_ylim([0,130])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Figures_LR_test/{}-lcoe.png".format(filename),transparent=True)


# # emissions = pd.DataFrame(0.,index=years,columns=techs)
# # for year in years:
# #     for tech in techs:
# #         emissions.at[year,tech] = model.generators_dispatch[tech,year].value*8760* 1000 * parameters.at["specific emissions",tech]

# # fig, ax = plt.subplots()
# # fig.set_dpi(2000)
# # emissions.plot(ax=ax,linewidth=3,cmap=colormap)
# # ax.set_xlabel("year")
# # ax.set_ylabel("CO2 [t]")
# # # ax.set_yscale("log")
# # # ax.set_ylim([0,40])
# # ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# # ax.legend().set_visible(lgnd)
# # fig.savefig("Figures_LR_test/{}-emissions.png".format(filename),transparent=True)

# #%%
# #Save data
# # build_years.to_pickle("build_years_one_factor.pkl")
# # capacities.to_pickle("capacities_one_factor.pkl")


# # build_years.to_pickle("build_years_multi.pkl")
# # capacities.to_pickle("capacities_multi.pkl")
