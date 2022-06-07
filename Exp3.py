
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 10:36:01 2022

@author: frederikmelson
"""





#%% Packages
import math 
from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
# import pyomo.contrib.preprocessing.plugins.induced_linearity as induced_linearity
import time
import pyomo.contrib.appsi.solvers.ipopt as ipo
startTime = time.time()
plt.style.use("seaborn")


#%% Scenarios and settings


# scenario = "no_co2-no_LR"
# scenario = "co2_constraint-no_LR"
scenario = "co2_constraint-LR" 
# scenario = "no_co2-LR"

# learning_scenarios = ["highLR","nomLR","lowLR"]

# learning rate assumptions
# learning_scenario = "highLR"
# learning_scenario = "lowLR"
learning_scenario = "nomLR"

filename = "exp3_"

# CO2 budget for 2050 global warming goals
co2_until_2050 = 10500 # MtCO2 (10.5 gigaton CO2)
co2_budget = {2020:3000,2025:3000,2030:2000,2035:1500,2040:1000,2045:0,2050:0}


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


parameters  = pd.read_pickle("import_data/parameters.pkl")
store_param = pd.read_pickle("import_data/store_param.pkl")
CC_param    = pd.read_pickle("import_data/CC_param.pkl")

a_file = open("import_data/demand_elec5y.pkl", "rb")
demand = pickle.load(a_file) #GWh



# CF_solar_one    = pd.read_pickle("CF_solar_one.pkl")
# CF_onwind_one   = pd.read_pickle("CF_onwind_one.pkl")
# CF_offwind_one  = pd.read_pickle("CF_offwind_one.pkl")

# Cf_solar    = pd.read_pickle("Cf_solar.pkl")
# Cf_onshore  = pd.read_pickle("Cf_onshore.pkl")
# Cf_offshore = pd.read_pickle("Cf_offshore.pkl")


Cf_solar      = pd.read_pickle("import_data/cf_solar3h4d.pkl")
Cf_onshore    = pd.read_pickle("import_data/cf_onshore3h4d.pkl")
Cf_offshore   = pd.read_pickle("import_data/cf_offshore3h4d.pkl")
# demand6d3h      = pd.read_pickle("demand6d3h.pkl")
# demand6d3h.reset_index()




techs_file          = "import_data/techs.pkl"
fossil_techs_file   = "import_data/fossil_techs.pkl"
renewables_file     = "import_data/renewables.pkl"
wind_file           = "import_data/wind.pkl"
colors_file         = "import_data/colors.pkl"
storage_file        = "import_data/storage.pkl"
color_storage_file  = "import_data/color_storage.pkl"
CC_file             = "import_data/CC.pkl"

files = [techs_file,fossil_techs_file,renewables_file,
         wind_file,colors_file,storage_file,color_storage_file,CC_file]
lists = ["techs","fossil_techs","renewables","wind",
         "colors","storage","color_storage","CC_techs"]


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

# Green- or brownfield scenario:
for tech in techs:
    greenfield_df[tech] = 1/parameters.at["existing capacity",tech]
for tech in storage_techs:
    greenfield_df[tech] = 1/100 


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


# store_param.at["current annuity","battery_store"] += store_param.at["current annuity","battery_inverter"]

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


# carbon budget in average tCO2   
if "no_co2" in scenario:
    # co2_budget = 1
    print("No CO2 budget")
else:
    # co2_budget = co2_until_2050 # [tCO2] 10 Gigatons CO2
    print("CO2 budget of "+ str(co2_until_2050) + " MtCO2")


    
#%% One node model
model = ConcreteModel()
model.generators            = Var(techs, years,bounds=(0,5000), within=NonNegativeReals) # bounds=(0.0,1000)
model.generators_dispatch   = Var(techs, years, hours, within=NonNegativeReals)
model.generators_built      = Var(techs, years, within=NonNegativeReals)
model.fixed_costs           = Var(techs, years, within=NonNegativeReals) #EUR/GW/a
# model.capital_costs         = Var(techs, years, within=NonNegativeReals) #Eur/GW

model.SOC                   = Var(storage_techs,years, hours,initialize=0, within=NonNegativeReals)
model.storage               = Var(storage_techs, years, initialize=0,bounds=(0,1000), within=NonNegativeReals)
model.storage_built         = Var(storage_techs, years, initialize=0, within=NonNegativeReals)
model.fixed_costs_storage   = Var(storage_techs, years, initialize=0, within=NonNegativeReals)
model.storage_charge        = Var(storage_techs, years, hours, initialize=0, within=NonNegativeReals)
model.storage_discharge     = Var(storage_techs, years, hours, initialize=0, within=NonNegativeReals)
# model.capital_costs_storage = Var(storage_techs, years, within=NonNegativeReals) #Eur/GW



# Starting point for the interior point optimization
# for year in years:
#     for tech in techs:
#         model.generators[tech,year] = 500
model.generators["OCGT",years[0]] = 100
model.generators["onshore_wind",years[0]] = 100
model.generators["offshore_wind",years[0]] = 50
model.generators["solar_PV",years[0]] = 200
model.generators["coal",years[0]] = 50
model.generators["nuclear",years[0]] = 0




#Value of currently installed technology:
constant = sum(parameters.at['existing capacity',tech] * 
               parameters.at['current annuity', tech]*1000/1e9/(1+r)**(hour-hours[0]) 
               for tech in techs for hour in hours if hour < hours[0] + 
               parameters.at['lifetime',tech] - parameters.at['existing age',tech])

print("Cost of existing capacity =", "%.2f"%constant)

#1e3 for EUR/MW to EUR/GW, then /1e9 to get billion EUR
model.objective = Objective(expr=constant +
                            sum(model.generators_built[tech,year] * model.fixed_costs[tech,year]*1e3/1e9 * 
                                sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                    if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech]))) 
                                for tech in techs for year in years)+
                            sum(model.generators_dispatch[tech,year,hour] * parameters.at['marginal cost',tech]*
                                1000*dty*hour_interval*interval/1e9 /(1+r)**(year-years[0])
                                for tech in techs
                                for year in years
                                for hour in hours)+
                            sum(model.storage_built[tech,year] * model.fixed_costs_storage[tech,year]*1e3/1e9 * 
                                sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                    if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech])))
                                for tech in storage_techs for year in years))



#%% Balance Constraints

def balance_constraint(model,tech,year,hour): # GWh
    return (demand[year].at[hour,0]  <= sum(model.generators_dispatch[tech, year, hour] for tech in techs) + 
            model.storage_discharge['battery_store',year,hour] - model.storage_charge['battery_store',year,hour])
model.balance_constraint = Constraint(techs,years,hours, rule=balance_constraint)

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


# def stored_energy_constraint(model,year,hour):
#     if year == 2020:
#         if hour == 0: 
#             return model.SOC['battery_store',year,hour] == 0# model.storage["battery_store",year] #- model.storage_discharge['battery_store',year,hour]
#         else:
#             return model.SOC['battery_store',year,hour] == model.SOC['battery_store',year,hour-1] + model.storage_charge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter'] - model.storage_discharge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter']
#     else:
#         if hour == 0: 
#             return model.SOC['battery_store',year,hour] == model.SOC['battery_store',year-interval,31] + model.storage_charge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter'] - model.storage_discharge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter']
#         else:
#             return model.SOC['battery_store',year,hour] == model.SOC['battery_store',year,hour-1] + model.storage_charge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter'] - model.storage_discharge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter']
# model.stored_energy_constraint = Constraint(years,hours,rule=stored_energy_constraint)

def stored_energy_constraint(model,year,hour):
    if hour == 0: 
        return model.SOC['battery_store',year,hour] == 0
    else:
        return (model.SOC['battery_store',year,hour] == model.SOC['battery_store',year,hour-1] + 
                model.storage_charge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter'] - 
                model.storage_discharge['battery_store',year,hour]*store_param.at['efficiency','battery_inverter'])
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
    
    return (model.storage[tech,year] == constant + sum(model.storage_built[tech,yearb] for yearb in years 
                                                       if ((year>= yearb) and (year < yearb + store_param.at["lifetime",tech])))) #GW
model.build_years_storage = Constraint(storage_techs,years, rule=build_years_storage)

def inverter_constraint(model,storage_techs,year):
    return model.storage["battery_store",year] <=  model.storage['battery_inverter',year]
model.inverter_constraint = Constraint(storage_techs,years, rule=inverter_constraint)

# def storage_bounding_constraint(model,tech,year):
#     return sum(model.storage[tech,year] for tech in storage_techs) <= 1000
# model.storage_bounding_constraint = Constraint(storage_techs, years, rule=storage_bounding_constraint)


def fixed_cost_constraint_storage(model,tech,year):
    if store_param.at["learning parameter",tech] == 0:
        return model.fixed_costs_storage[tech,year] == store_param.at["current annuity",tech] #EUR/MW/a
    else:
        return (model.fixed_costs_storage[tech,year] == (store_param.at["2050 annuity",tech] + 
                                                              (1+store_param.at["current annuity",tech]-store_param.at["2050 annuity",tech]) * 
                                                              (1+sum(model.storage_built[tech,yeart] for yeart in years 
                                                                     if yeart < year)*greenfield_df[tech])**(-store_param.at["learning parameter",tech]))) #EUR/MW/a
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
    
    return (model.generators[tech,year] == constant + sum(model.generators_built[tech,yearb] 
                                                          for yearb in years 
                                                          if ((year>= yearb) and (year < yearb + parameters.at["lifetime",tech])))) #GW
model.build_years = Constraint(techs, years, rule=build_years)



def fixed_cost_constraint(model,tech,year):
    if parameters.at["learning parameter",tech] == 0:
        return model.fixed_costs[tech,year] == parameters.at["current annuity",tech] 
    else:
        return (model.fixed_costs[tech,year] == (parameters.at["2050 annuity",tech] + 
                                                      (1+parameters.at["current annuity",tech]-parameters.at["2050 annuity",tech]) * 
                                                      (1+sum(model.generators_built[tech,yeart] for yeart in years 
                                                             if yeart < year)*greenfield_df[tech])**(-parameters.at["learning parameter",tech]))) 
model.fixed_cost_constraint = Constraint(techs, years, rule=fixed_cost_constraint)


#%% CO2 constraint
# Converting GW to MW (1000), dty to go from 4 days to 365 days, 

# def co2_constraint(model,tech,year,hour): # MtCO2 = MWh*MtCO2/MWh
#     return co2_budget[year]/interval >= sum((model.generators_dispatch[tech,year,hour]*1000*dty * parameters.at["specific emissions",tech]) for tech in techs for hour in hours) if "co2_constraint" in scenario else Constraint.Skip
# model.co2_constraint = Constraint(techs,years, hours,rule=co2_constraint)


def co2_constraint2050(model,tech,hour): 
        return (0 >= sum((model.generators_dispatch[tech,2050,hour] * 1000 * dty * 
                          parameters.at["specific emissions",tech]) 
                          for tech in techs 
                          for hour in hours) 
                if "co2_constraint" in scenario else Constraint.Skip)
model.co2_constraint2050 = Constraint(techs, hours,rule=co2_constraint2050)

# def co2_constraint2050(model,tech,year,hour): 
        # return (co2_until_2050 >= sum((model.generators_dispatch[tech,year,hour] * 1000 * 
        #                                dty * parameters.at["specific emissions",tech]) 
        #                               for tech in techs 
        #                               for hour in hours 
        #                               for year in years) 
        #         if "co2_constraint" in scenario else Constraint.Skip)
# model.co2_constraint2050 = Constraint(techs, years, hours,rule=co2_constraint2050)




#%%
executionTime = (time.time() - startTime)
print('Writing time for Pyomo model in seconds: ' + str(executionTime))
#%% Solving model
startTime = time.time()

# ipopt #suffixes=['dual'] keepfiles=False mindtpy
opt = SolverFactory('ipopt')
opt.options['max_iter'] = 100000
# opt.options['acceptable_constr_viol_tol'] = 0.1
opt.options['timing_statistics'] = "yes"
opt.options['max_soft_resto_iters'] = 10000
# opt.options['print_info_string'] = "yes"
# opt.options['acceptable_obj_change_tol'] = 0.001
# opt.options['acceptable_tol'] = 1e-4
# opt.options['derivative_test'] ='first-order'
# opt.options['compl_inf_tol'] =0.01


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


CO2_emitted = sum((model.generators_dispatch[tech,year,hour].value*1000*dty*3*interval * 
                   parameters.at["specific emissions",tech]) 
                  for tech in fossil_techs for hour in hours for year in years)
print("CO2 emitted =","%.2f"% CO2_emitted) 

if "no_LR" in scenario:
    print("No learning")
else:
    if "highLR" in learning_scenario:
        print("High learning rates")
    else: 
        if "lowLR" in learning_scenario:
            print("Low learning rates")
        else:
            # nom learning
            print("Nominal learning rates")

print("Cost for storage =","%.2f"% sum(model.storage_built[tech,year].value * 
                                       model.fixed_costs_storage[tech,year].value/1e6 * 
                                       sum(1/(1+r)**(yearb-years[0]) 
                                           for yearb in years 
                                           if ((yearb>=year) and (yearb < year + 
                                                                  store_param.at['lifetime',tech]))) 
                                       for tech in storage_techs for year in years))


print("Cost for renewables =","%.2f"%sum(model.generators_built[tech,year].value * 
                                         model.fixed_costs[tech,year].value/1e6 * 
                                         sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                             if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                                         for tech in renewables for year in years))


fossil_techs = ["CCGT","OCGT","coal"]
print("Cost for fossil =","%.2f"% sum(model.generators_built[tech,year].value * 
                                                  model.fixed_costs[tech,year].value/1e6 * 
                                                  sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                                      if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech])))
                                                  for tech in fossil_techs for year in years))

print("Cost for nuclear =","%.2f"% sum(model.generators_built['nuclear',year].value * 
                                                  model.fixed_costs['nuclear',year].value/1e6 * 
                                                  sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                                      if ((yearb>=year) and (yearb < year + parameters.at['lifetime','nuclear'])))
                                                  for year in years))


print("Marginal costs or fuel costs =","%.2f"%sum(model.generators_dispatch[tech,year,hour].value * 
                                                  parameters.at['marginal cost',tech]*1000*dty*
                                                  hour_interval*interval / 1e9 /(1+r)**(year-years[0]) 
                                                  for tech in techs 
                                                  for year in years  
                                                  for hour in hours))

print("Total cost (in billion euro) =","%.2f"% model.objective())
#%% System costs 

system_costs={}

system_costs["storage"] = sum(model.storage_built[tech,year].value * 
                              model.fixed_costs_storage[tech,year].value/1e9 * 
                              sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                  if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech]))) 
                              for tech in storage_techs 
                              for year in years)
system_costs["renewables"] = sum(model.generators_built[tech,year].value * 
                                 model.fixed_costs[tech,year].value/1e9 * 
                                 sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                     if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech]))) 
                                 for tech in renewables 
                                 for year in years)

system_costs["fossil"] = sum(model.generators_built[tech,year].value * 
                             model.fixed_costs[tech,year].value/1e9 * 
                             sum(1/(1+r)**(yearb-years[0]) for yearb in years 
                                 if ((yearb>=year) and (yearb < year + parameters.at['lifetime',tech]))) 
                             for tech in fossil_techs 
                             for year in years)

system_costs["marginal"] = sum(model.generators_dispatch[tech,year,hour].value * 
                               parameters.at['marginal cost',tech]*1000*dty*
                               hour_interval*interval / 1e9 /(1+r)**(year-years[0])
                            for tech in techs
                            for year in years
                            for hour in hours)

for year in years:
    demand[year] = demand[year].iloc[0:32]
   

capcost=pd.DataFrame(0.,index=years,columns=techs+storage_techs)
for year in years:    
    for tech in techs:
        if parameters.at["learning parameter",tech] == 0:
            capcost.at[year,tech] = parameters.at["current capital cost",tech] #EUR/MW
        else:
            capcost.at[year,tech] = (parameters.at["2050 capital cost",tech] + 
                                                          (1+parameters.at["current capital cost",tech]-parameters.at["2050 capital cost",tech]) * 
                                                          (1+sum(model.generators_built[tech,yeart].value for yeart in years 
                                                                 if yeart < year)*greenfield_df[tech])**(-parameters.at["learning parameter",tech])) #EUR/MW
    for tech in storage_techs:
        if store_param.at["learning parameter",tech] == 0:
            capcost.at[year,tech] = store_param.at["current capital cost",tech] #EUR/MW
        else:
            capcost.at[year,tech] = (store_param.at["2050 capital cost",tech] + 
                                                                  (1+store_param.at["current capital cost",tech]-store_param.at["2050 capital cost",tech]) * 
                                                                  (1+sum(model.storage_built[tech,yeart].value for yeart in years 
                                                                         if yeart < year)*greenfield_df[tech])**(-store_param.at["learning parameter",tech])) #EUR/MW


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


dispatch_4days = pd.DataFrame(0.,index=years,columns=techs+storage)
for year in years:
    for tech in techs:
            dispatch_4days.at[year,tech] = dispatch[year].loc[:,tech].sum()*3
    for tech in storage:
            dispatch_4days.at[year,tech] = dispatch[year].loc[:,tech].sum()*3


capacities = pd.DataFrame(0.,index=years,columns=techs+storage_techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = model.generators[tech,year].value
    for tech in storage_techs:
        capacities.at[year,tech] = model.storage[tech,year].value
for year in years:
    for tech in techs:
        if capacities.at[year,tech] <= 0:
            capacities.at[year,tech] = 0
    for tech in storage_techs:
        if capacities.at[year,tech] <= 0:
            capacities.at[year,tech] = 0

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
        for tech in storage_techs:
            if hour == 0:
                cap_fac[tech,year] = 0
            if dispatch[year].at[hour,"battery_store"] > 0.1:
                cap_fac[tech,year] = cap_fac[tech,year]+1



for year in years:
    for tech in techs:
        cap_fac[tech,year] = cap_fac[tech,year]*dty*3/8760
        dispatch_sum[year] = dispatch[year].sum(0)*dty*3
        lcoe_capfactor[tech,year] = dispatch_sum[year].at[tech]/(capacities.at[year,tech]*8760)
    for tech in storage_techs:
        cap_fac[tech,year] = cap_fac[tech,year]*dty*3/8760
        dispatch_sum[year] = dispatch[year].sum(0)*dty*3
        lcoe_capfactor[tech,year] = dispatch_sum[year].at['battery_store']/(capacities.at[year,tech]*8760)

level_cost = pd.DataFrame(0.,index=years,columns=techs+storage_techs)
fixed_cost = pd.DataFrame(0.,index=years,columns=techs+storage_techs)

for year in years:
        for tech in techs:
            level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(8760*lcoe_capfactor[tech,year]) + parameters.at["marginal cost",tech]
            fixed_cost.at[year,tech] = model.fixed_costs[tech,year].value # EUR/MW/a
            # level_cost.at[year,tech] = (model.fixed_costs[tech,year].value/8760 +parameters.at["marginal cost",tech])#/parameters.at["capacity factor",tech]
            # if capacities.at[year,tech]<1: 
                # level_cost.at[year,tech] = model.fixed_costs[tech,year].value/(8760*1)/1000 + parameters.at["marginal cost",tech]
        for tech in storage_techs:
            fixed_cost.at[year,tech] = model.fixed_costs_storage[tech,year].value #EUR/MW/a
            level_cost.at[year,tech] = model.fixed_costs_storage[tech,year].value/(8760*lcoe_capfactor[tech,year])


capex_pv = capcost["solar_PV"]
capex_onwind = capcost["onshore_wind"]
capex_offwind = capcost["offshore_wind"]
capex_battery = capcost["battery_store"]
capex_coal = capcost["coal"]
capex_nuclear = capcost["nuclear"]
capex_ocgt = capcost["OCGT"]


build_years = pd.DataFrame(0.,index=years,columns=techs+storage_techs) 
for year in years:
    for tech in techs:
        build_years.at[year,tech] = model.generators_built[tech,year].value
    for tech in storage_techs:
        build_years.at[year,tech] = model.storage_built[tech,year].value
for year in years:
    for tech in techs:
        if build_years.at[year,tech] <= 0:
            build_years.at[year,tech] = 0
for year in years:
    for tech in storage_techs:
        if build_years.at[year,tech] <= 0:
            build_years.at[year,tech] = 0
#%% Plotting

# # colormap = "Set2"
for year in years:
    demand[year] = demand[year].iloc[0:32]
   

# file name
if "no_LR" in scenario: 
    filename = filename+scenario
else:
    filename = filename+scenario+"-"+learning_scenario

if Greenfield == False:
    filename = "Br"+filename
else:
    filename = "Gr"+filename
    




plotyear= 2050
fig, ax = plt.subplots()
fig.set_dpi((200))
dispatch[plotyear].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
demand[plotyear].plot(kind="line",ax=ax)
ax.set_xlabel("day")
ax.set_ylabel("Gross electricity generation [GWh]")
ax.set_ylim([0,1000])
plt.title("Hourly dispatch in "+ str(plotyear))
# plt.xticks(ticks=[0,16/14,2*16/14,3*16/14,4*16/14,5*16/14,6*16/14,7*16/14,8*16/14,9*16/14,10*16/14,11*16/14,12*16/14,13*16/14],labels=['1','2','3','4','5','6','7','8','9','10','11','12','13','14'])
plt.xticks(ticks=[0, 32/4, 2*32/4, 3*32/4],labels=['1','2','3','4'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-dispatch.png".format(filename),transparent=True)


fig, ax = plt.subplots(figsize = [12, 4], dpi = 200)
dispatch_allyears.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
demand_allyears.plot(kind="line",ax=ax)
ax.set_xlabel("year")
plt.xticks(ticks=[0, 224/7, 2*224/7, 3*224/7, 4*224/7, 5*224/7, 6*224/7],labels=['2020','2025','2030','2035','2040','2045','2050'])
ax.set_ylabel("Gross electricity generation [GWh]")
ax.set_ylim([0,600])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-dispatchallyears.png".format(filename),transparent=True)



fig, ax = plt.subplots(figsize = [12, 4], dpi = 200)
dispatch_4days.plot(kind="bar",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
ax.set_xlabel("year")
plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6],labels=['2020','2025','2030','2035','2040','2045','2050'])
# plt.xticks(ticks=[0, 1221/ 11,2* 1221/ 11,3* 1221/ 11,4* 1221/ 11,5* 1221/ 11,6* 1221/ 11,7* 1221/ 11,8* 1221/ 11,9* 1221/ 11,10* 1221/ 11],labels=['2020','2023','2026','2029','2032','2035','2038','2041','2044','2047','2050'])
ax.set_ylabel("Electricity generation in model period [GWh]")
# plt.title('2050')
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-dispatch4days.png".format(filename),transparent=True)



fig, ax = plt.subplots()
fig.set_dpi((200))
capacities.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("Installed capacity [GW]")
# ax.set_ylim([0,10000])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-capacity.png".format(filename),transparent=True)

fig, ax = plt.subplots()
fig.set_dpi((200))
build_years.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("new capacity built [GW]")
# ax.set_ylim([0,250])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-new_capacity.png".format(filename),transparent=True)





        # for tech in storage:
        #     LCOE.at[year,tech] = model.fixed_costs_storage[tech, year].value/8760. + store_param.at["marginal cost",tech]
# https://www.nrel.gov/analysis/tech-lcoe-documentation.html

fig, ax = plt.subplots()
fig.set_dpi(200)
level_cost.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("LCOE [EUR/MWh]")
ax.set_yscale("log")
ax.set_ylim([10,200])
ax.legend().set_visible(lgnd)
ax.legend(bbox_to_anchor=(1.01, 0.8), ncol=1, fancybox=False, shadow=False)
ax.tick_params(axis='y', which='minor')
ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
# fig.savefig("Results/Exp3/figures/{}-LCOE.png".format(filename),transparent=True)

fig, ax = plt.subplots()
fig.set_dpi(200)
fixed_cost.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("Annualized costs [EUR/MW/a]")
# ax.set_yscale("log")
# ax.set_ylim([0,150])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)


fig, ax = plt.subplots()
fig.set_dpi(200)
capcost.plot(ax=ax,linewidth=3,color=colors)
ax.set_xlabel("year")
ax.set_ylabel("Capital costs [EUR/MW]")
ax.ticklabel_format(axis='both', style='plain')
# ax.set_yscale("log")
ax.set_ylim([0,2000000])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-capex.png".format(filename),transparent=True)



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
            storage_discharge[year].at[hour,'Battery discharge'] = model.storage_discharge['battery_store',year,hour].value
            storage_charge[year].at[hour,'Battery charge'] = model.storage_charge['battery_store',year,hour].value
            energy_stored[year].at[hour,'Battery SOC'] = model.SOC['battery_store',year,hour].value
            storage_discharge[year].at[hour,'Battery discharge'] = -storage_discharge[year].at[hour,'Battery discharge']
storage_cap = pd.DataFrame(0.,index=years,columns=storage)
for year in years:
    # for tech in storage:
        storage_cap.at[year,'battery_store'] = model.storage['battery_store',year].value

  
storage_discharge_allyears = pd.concat([storage_discharge[year] for year in years])  
storage_discharge_allyears = storage_discharge_allyears.reset_index()
storage_discharge_allyears = storage_discharge_allyears.drop(["index"],axis=1)

storage_charge_allyears = pd.concat([storage_charge[year] for year in years])  
storage_charge_allyears = storage_charge_allyears.reset_index()
storage_charge_allyears = storage_charge_allyears.drop(["index"],axis=1)

energy_stored_allyears = pd.concat([energy_stored[year] for year in years])  
energy_stored_allyears = energy_stored_allyears.reset_index()
energy_stored_allyears = energy_stored_allyears.drop(["index"],axis=1)


# for year in years:
#     for hour in hours:
#         # for tech in storage:
#             if storage_discharge[year].at[hour,'battery_store'] <= 0:
#                 storage_discharge[year].at[hour,'battery_store'] = 0
            
        
# fig, ax = plt.subplots()
# fig.set_dpi((400))
# energy_stored[2032].plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
# ax.set_xlabel("days")
# ax.set_ylabel("Energy stored [GWh]")
# plt.title('2032')
# plt.xticks(ticks=[0,111/14,2*111/14,3*111/14,4*111/14,5*111/14,6*111/14,7*111/14,8*111/14,9*111/14,10*111/14,11*111/14,12*111/14,13*111/14],labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13'])
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)

# fig, ax = plt.subplots()
# fig.set_dpi((400))
# storage_cap.plot(kind="area",stacked=True,ax=ax,linewidth=0,color=colors)#cmap=colormap)
# ax.set_xlabel("year")
# ax.set_ylabel("Storage capacity [GWh]")
# plt.title('Storage capacity')
# fig.tight_layout()
# ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
# ax.legend().set_visible(lgnd)


fig, ax = plt.subplots(figsize = [12, 4], dpi = 200)
storage_discharge[plotyear].plot(kind="line",ax=ax,linewidth=1,color='r',label='Storage discharged')#cmap=colormap)
storage_charge[plotyear].plot(kind="line",ax=ax,linewidth=1,color="b",label='Storage charged')#cmap=colormap)
energy_stored[plotyear].plot(kind="line",ax=ax,linewidth=1.5,color='g',label='Storage capacity')#cmap=colormap)
ax.set_xlabel("days")
ax.set_ylabel("Storage [GWh]")
plt.title(str(plotyear))
plt.xticks(ticks=[0, 32/4, 2*32/4, 3*32/4],labels=['1','2','3','4'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-SOC.png".format(filename),transparent=True)

fig, ax = plt.subplots(figsize = [12, 4], dpi = 200)
storage_discharge_allyears.plot(kind="line",ax=ax,linewidth=1,color='r',label='Storage discharged')#cmap=colormap)
storage_charge_allyears.plot(kind="line",ax=ax,linewidth=1,color="b",label='Storage charged')#cmap=colormap)
energy_stored_allyears.plot(kind="line",ax=ax,linewidth=1.5,color='g',label='Storage capacity')#cmap=colormap)
ax.set_xlabel("year")
ax.set_ylabel("Storage [GWh]")
plt.xticks(ticks=[0, 224/7, 2*224/7, 3*224/7, 4*224/7, 5*224/7, 6*224/7],labels=['2020','2025','2030','2035','2040','2045','2050'])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
# fig.savefig("Results/Exp3/figures/{}-SOCallyears.png".format(filename),transparent=True)



#%%
#Save data
# # build_years.to_pickle("build_years_one_factor.pkl")
# # capacities.to_pickle("capacities_one_factor.pkl")

level_cost.to_excel("Results/Exp3/LCOE_{}.xlsx".format(filename))
fixed_cost.to_excel("Results/Exp3/annuity_costs_{}.xlsx".format(filename))
capcost.to_excel("Results/Exp3/capital_costs_{}.xlsx".format(filename))


# build_years.to_pickle("Results/Exp3/{}build_years_2050CO2.pkl".format(filename))
# capacities.to_pickle("Results/Exp3/{}capacities_2050CO2.pkl".format(filename))
# dispatch_allyears.to_pickle("Results/Exp3/{}dispatch_allyears_2050CO2.pkl".format(filename))
# dispatch_4days.to_pickle("Results/Exp3/{}dispatch_sum_2050CO2.pkl".format(filename))

# capex_pv.to_pickle("Results/Exp3/capex_pv_{}.pkl".format(filename))
# capex_onwind.to_pickle("Results/Exp3/capex_onwind{}.pkl".format(filename))
# capex_offwind.to_pickle("Results/Exp3/capex_offwind{}.pkl".format(filename))
# capex_battery.to_pickle("Results/Exp3/capex_battery{}.pkl".format(filename))
# capex_coal.to_pickle("Results/Exp3/capex_coal{}.pkl".format(filename))
# capex_nuclear.to_pickle("Results/Exp3/capex_nuclear{}.pkl".format(filename))
# capex_ocgt.to_pickle("Results/Exp3/capex_ocgt{}.pkl".format(filename))


# build_years.to_pickle("build_years_lowLR_2050CO2.pkl")
# capacities.to_pickle("capacities_lowLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("dispatch_allyears_lowLR_2050CO2.pkl")
# dispatch_4days.to_pickle("dispatch_sum_lowLR_2050CO2.pkl")

# build_years.to_pickle("build_years_nomLR_2050CO2.pkl")
# capacities.to_pickle("capacities_nomLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("dispatch_allyears_nomLR_2050CO2.pkl")
# dispatch_4days.to_pickle("dispatch_sum_nomLR_2050CO2.pkl")

# build_years.to_pickle("build_years_noLR_2050CO2.pkl")
# capacities.to_pickle("capacities_noLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("dispatch_allyears_noLR_2050CO2.pkl")
# dispatch_4days.to_pickle("dispatch_sum_noLR_2050CO2.pkl")

# build_years.to_pickle("build_years_highLR_2050CO2.pkl")
# capacities.to_pickle("capacities_highLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("dispatch_allyears_highLR_2050CO2.pkl")
# dispatch_4days.to_pickle("dispatch_sum_highLR_2050CO2.pkl")


# build_years.to_pickle("Br_build_years_noLR_2050CO2.pkl")
# capacities.to_pickle("Br_capacities_noLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("Br_dispatch_allyears_noLR_2050CO2.pkl")
# dispatch_4days.to_pickle("Br_dispatch_4days_noLR_2050CO2.pkl")

# build_years.to_pickle("Br_build_years_lowLR_2050CO2.pkl")
# capacities.to_pickle("Br_capacities_lowLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("Br_dispatch_allyears_lowLR_2050CO2.pkl")
# dispatch_4days.to_pickle("Br_dispatch_4days_lowLR_2050CO2.pkl")

# build_years.to_pickle("Br_build_years_nomLR_2050CO2.pkl")
# capacities.to_pickle("Br_capacities_nomLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("Br_dispatch_allyears_nomLR_2050CO2.pkl")
# dispatch_4days.to_pickle("Br_dispatch_4days_nomLR_2050CO2.pkl")

# build_years.to_pickle("Br_build_years_highLR_2050CO2.pkl")
# capacities.to_pickle("Br_capacities_highLR_2050CO2.pkl")
# dispatch_allyears.to_pickle("Br_dispatch_allyears_highLR_2050CO2.pkl")
# dispatch_4days.to_pickle("Br_dispatch_4days_highLR_2050CO2.pkl")


executionTime = (time.time() - startTime)
print('Solving model in seconds: ' + str(executionTime))
