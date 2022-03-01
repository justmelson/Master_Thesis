# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:58:42 2021

@author: justm
"""
#%% Packages

from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, Constraint, Suffix, exp, value
from pyomo.opt import SolverFactory
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')

import requests
import io

from annuity_fun import annuity


#%% Scenarios and settings

# scenario = "no_co2-no_learning"
# scenario = "co2-0p2-no_learning"
scenario = "co2-0p2-learning"
# scenario = "no_co2-learning"

# learning_scenario = "high_learning"
learning_scenario = "low_learning"

# Do you want to include capacity factor?
Capacityfactor = True

co2_until_2050 = 10000000000 # 100 million tCO2 ,10000000000 # 10 gigaton CO2


# legend on/off when plotting
lgnd = False

#%% Importing data

# Downloading the csv files from pypsa GitHub account

url={}
url[0]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2020.csv"
url[1]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2025.csv"
url[2]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2030.csv"
url[3]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2035.csv"
url[4]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2040.csv"
url[5]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2045.csv"
url[6]="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2050.csv"

costs = []
for i in range(7):
    link = url[i] # Make sure the url is the raw version of the file on GitHub
    download = requests.get(link).content

    # Reading the downloaded content and turning it into a pandas dataframe

    df = pd.read_csv(io.StringIO(download.decode('utf-8')))
    costs.append(df)
  # Printing out the first 5 rows of the dataframe

    #print (costs[6].head())

r = 0.07 # discount rate
fuel_cost_gas = 21.6 # in €/MWh_th from  https://doi.org/10.1016/j.enconman.2019.111977
#%% Dataframe init

techs = ["offshore_wind","onshore_wind","solar_PV", "CCGT","OCGT","coal"]
fossil_techs = ["CCGT","OCGT","coal"]
renewables = ["offshore_wind","onshore_wind","solar_PV"]
wind = ["offshore_wind","onshore_wind"]
colors = ["#707070","#ff9000","#f9d002", '#00FF00',"g","r","b","black"]
parameters = pd.DataFrame(columns=techs)
storage = ["battery_store","battery_inverter","hydrogen_storage","electrolysis","fuel_cell"]
color_storage = ["salmon","magenta","aqua","chartreuse","chocolate"]
store_param = pd.DataFrame(columns=storage)
demand = pd.DataFrame(columns= ["demand"])

#%% Technology data
parameters.loc["capacity factor"] = [0.52,0.44,0.21,0.63,0.63,0.83]
parameters.loc["current capital cost"] = [annuity(costs[0]['value'][408],r)*costs[0]['value'][407]*1000*(1+costs[0]['value'][405]),
                                     annuity(costs[0]['value'][425],r)*costs[0]['value'][424]*1000*(1+costs[0]['value'][422]),
                                     (annuity(costs[0]['value'][437],r)*costs[0]['value'][436]*1000*(1+costs[0]['value'][434])),
                                     annuity(costs[0]['value'][9],r)*costs[0]['value'][8]*1000*(1+costs[0]['value'][3]),
                                     annuity(costs[0]['value'][140],r)*costs[0]['value'][139]*1000*(1+costs[0]['value'][136]),
                                     annuity(costs[0]['value'][274],r)*costs[0]['value'][273]*1000*(1+costs[0]['value'][269])] # EUR/MW/a
parameters.loc["potential capital cost"] = [annuity(costs[6]['value'][408],r)*costs[6]['value'][407]*1000*(1+costs[6]['value'][405]),
                                     annuity(costs[6]['value'][425],r)*costs[6]['value'][424]*1000*(1+costs[6]['value'][422]),
                                     (annuity(costs[6]['value'][437],r)*costs[6]['value'][436]*1000*(1+costs[6]['value'][434])),
                                     annuity(costs[6]['value'][9],r)*costs[6]['value'][8]*1000*(1+costs[6]['value'][3]),
                                     annuity(costs[6]['value'][140],r)*costs[6]['value'][139]*1000*(1+costs[6]['value'][136]),
                                     annuity(costs[6]['value'][274],r)*costs[6]['value'][273]*1000*(1+costs[6]['value'][269])]# EUR/MW/a
parameters.loc["learning parameter"] = [0.19,0.32,0.47,0.34,0.15,0.083] # [0.12,0.12,0.23,0.14,0.15]
parameters.loc["marginal cost"] = [0,
                                   0,
                                   0,
                                   fuel_cost_gas/costs[0]['value'][7],
                                   fuel_cost_gas/costs[0]['value'][138],
                                   costs[0]['value'][272]/costs[0]['value'][271]] #EUR/MWhel
parameters.loc["specific emissions"] = [0.,0.,0.,0.374,0.588,0.76] #tcO2/MWhel
parameters.loc["lifetime"] = [27,27,32.5,25,25,40]  #years
parameters.loc["existing age"] = [10,10,5,14,14,20] # [0,0,0,0,0,0] years
parameters.loc["existing capacity"] = [25,174,100,200,200,128] #[26,174,123,112,112,128] #[0,0,0,0,0,0] #GW

parameters.loc["current LCOE"] = parameters.loc["current capital cost"]/8760 + parameters.loc["marginal cost"]
parameters.loc["potential LCOE"] = parameters.loc["potential capital cost"]/8760 + parameters.loc["marginal cost"]


parameters.round(3)

store_param.loc["current capital cost"] = [annuity(costs[0]['value'][165],r)*301*1000,
                                      annuity(costs[0]['value'][163],r)*costs[0]['value'][162]*1000*(1+costs[0]['value'][160]),
                                      annuity(costs[0]['value'][365],r)*costs[0]['value'][364]*1000*(1+costs[0]['value'][363]),
                                      annuity(costs[0]['value'][330],r)*costs[0]['value'][329]*1000*(1+costs[0]['value'][327]),
                                      annuity(costs[0]['value'][335],r)*costs[0]['value'][334]*1000*(1+costs[0]['value'][331])] # EUR/MW/a
store_param.loc["potential capital cost"] = [annuity(costs[6]['value'][165],r)*costs[6]['value'][164]*1000,
                                      annuity(costs[6]['value'][163],r)*costs[6]['value'][162]*1000*(1+costs[6]['value'][160]),
                                      annuity(costs[6]['value'][365],r)*costs[6]['value'][364]*1000*(1+costs[6]['value'][363]),
                                      annuity(costs[6]['value'][330],r)*costs[6]['value'][329]*1000*(1+costs[6]['value'][327]),
                                      annuity(costs[6]['value'][335],r)*costs[6]['value'][334]*1000*(1+costs[6]['value'][331])] # EUR/MW/a]# EUR/MW/a
store_param.loc["learning parameter"] = [0.12,0.1,0.1,0.18,0.18] # 0.24not sure about inverter learning rate
store_param.loc["marginal cost"] = [0.,0.,0.,0.,0.] #EUR/MWhel
store_param.loc["specific emissions"] = [0.,0.,0.,0.,0.] #tcO2/MWhel
store_param.loc["lifetime"] = [30,10,20,25,10]  #years
store_param.loc["existing age"] = [0,0,0,0,0] #years
store_param.loc["existing capacity"] = [0,0,0,0,0] #[20,20,20,20,20] #[25,195,141,172] #GW

store_param.loc["current LCOE"] = store_param.loc["current capital cost"]/8760 + store_param.loc["marginal cost"]
store_param.loc["potential LCOE"] = store_param.loc["potential capital cost"]/8760 + store_param.loc["marginal cost"]
# store_param.loc["bLR"] = [0,0,0,0,0]

# Capacity factor is included:
if Capacityfactor is True:
    for tech in techs:
        parameters.at["current capital cost",tech] = (parameters.at["current capital cost",tech]/parameters.at["capacity factor",tech]) 
        # parameters.at["current LCOE",tech] = (parameters.at["current LCOE",tech]/parameters.at["capacity factor",tech])
        parameters.at["potential capital cost",tech] = (parameters.at["potential capital cost",tech]/parameters.at["capacity factor",tech]) 


#capital_cost = annuity(lifetime,discount rate)*Investment*(1+FOM) # in €/MW

store_param.round(3)

#%%

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
    parameters.loc["learning parameter"] = 0
    store_param.loc["learning parameter"] = 0
    print("No learning")
else:
    if "high_learning" in learning_scenario:
        parameters.loc["learning parameter"] = [0.19,0.32,0.47,0.34,0.15,0.083] # [0.12,0.12,0.23,0.14,0.15]
        store_param.loc["learning parameter"] = [0.18,0.1,0.1,0.26,0.21]
        print("High learning rates")
    else: #low learning
        parameters.loc["learning parameter"] = [0.05,0,0.1,-0.01,0.15,0.06] # [0.12,0.12,0.23,0.14,0.15]
        store_param.loc["learning parameter"] = [0.08,0.1,0.1,0.18,0.15]
        print("Low learning rates")

# Calculating rate of cost reduction bLR
# for i in range(len(techs)):
#     parameters.loc["bLR"][i] = math.log(1-parameters.loc["learning parameter"][i]) / math.log(2)
# for i in range(len(storage)):
#     store_param.loc["bLR"][i] = math.log(1-store_param.loc["learning parameter"][i]) / math.log(2)


# carbon budget in average tCO2   
if "no_co2" in scenario:
    co2_budget = 1e30
    print("No CO2 budget")
else:
    co2_budget = co2_until_2050 # [tCO2] 10 Gigatons CO2
    print("CO2 budget of "+ str(co2_until_2050) + " tons CO2")




    
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
                              for tech in techs) +
                           sum(model.storage_built[tech,year] * model.fixed_costs_storage[tech,year]/1e6 * sum(1/(1+r)**(yearb-years[0]) for yearb in years if ((yearb>=year) and (yearb < year + store_param.at['lifetime',tech])))
                              for year in years
                              for tech in storage) +
                           sum(model.storage_dispatch[tech,year] * store_param.at['marginal cost',tech] * 8760/1e6/(1+r)**(year-years[0])
                              for year in years
                              for tech in storage))

#%% Constraints
def balance_constraint(model,year):
    return demand.at[year,"demand"] == sum(model.generators_dispatch[tech,year] for tech in techs)
model.balance_constraint = Constraint(years, rule=balance_constraint)


def storebalancePV_constraint(model,year):
    return model.storage_dispatch["battery_store",year] == model.generators_dispatch["solar_PV",year]*0.3
model.storebalancePV_constraint = Constraint(years, rule=storebalancePV_constraint)

def storebalanceWind_constraint(model,tech,year):
    return model.storage_dispatch["hydrogen_storage",year] == sum(model.generators_dispatch[tech,year] for tech in wind)*0.3
model.storebalanceWind_constraint = Constraint(renewables,years, rule=storebalanceWind_constraint)


def storage_constraint(model,tech,year):
    return model.storage_dispatch[tech,year] <= model.storage[tech,year]
model.storage_constraint = Constraint(storage, years, rule=storage_constraint)

def solar_constraint(model,year):
    return model.generators["solar_PV",year] <= sum(model.generators_dispatch[tech,year] for tech in techs)*0.5
model.solar_constraint = Constraint(years, rule=solar_constraint)

def onshore_constraint(model,year):
    return model.generators["onshore_wind",year] <= sum(model.generators_dispatch[tech,year] for tech in techs)*0.3
model.onshore_constraint = Constraint(years, rule=onshore_constraint)

def generator_constraint(model,tech,year):
    return model.generators_dispatch[tech,year] <= model.generators[tech,year] #*parameters.at["capacity factor",tech] # Including capacity factors 
model.generator_constraint = Constraint(techs, years, rule=generator_constraint)
        
    
def co2_constraint(model,tech,year):
    return co2_budget >= sum((model.generators_dispatch[tech,year] * 8760 * 1000 * parameters.at["specific emissions",tech]) for tech in techs for year in years)
model.co2_constraint = Constraint(techs,years,rule=co2_constraint)


def inverter_constraint(model,tech,year):
    return model.storage_dispatch["battery_store",year] == model.storage_dispatch["battery_inverter",year]
model.inverter_constraint = Constraint(storage, years, rule=inverter_constraint)

def fuelcell_constraint(model,tech,year):
    return model.storage_dispatch["hydrogen_storage",year] == model.storage_dispatch["fuel_cell",year]
model.fuelcell_constraint = Constraint(storage, years, rule=fuelcell_constraint)


def electrolysis_constraint(model,tech,year):
    return model.storage_dispatch["hydrogen_storage",year] == model.storage_dispatch["electrolysis",year]
model.electrolysis_constraint = Constraint(storage, years, rule=electrolysis_constraint)


def build_years(model,tech,year):
    if year < years[0] + parameters.at["lifetime",tech] - parameters.at["existing age",tech]:
        constant = parameters.at["existing capacity",tech]
    else:
        constant = 0.
    
    return model.generators[tech,year] == constant + sum(model.generators_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + parameters.at["lifetime",tech])))
model.build_years = Constraint(techs, years, rule=build_years)

def build_years_storage(model,tech,year):
    if year < years[0] + store_param.at["lifetime",tech] - store_param.at["existing age",tech]:
        constant = store_param.at["existing capacity",tech]
    else:
        constant = 0.
    
    return model.storage[tech,year] == constant + sum(model.storage_built[tech,yearb] for yearb in years if ((year>= yearb) and (year < yearb + store_param.at["lifetime",tech])))
model.build_years_storage = Constraint(storage, years, rule=build_years_storage)

def fixed_cost_constraint(model,tech,year):
    if parameters.at["learning parameter",tech] == 0:
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]
    else:
#         return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech]*(sum(model.generators[tech]))**-(parameters.at["bLR",tech])
        return model.fixed_costs[tech,year] == parameters.at["current capital cost",tech] * (1+sum(model.generators_built[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
        # return model.fixed_costs[tech,year] == parameters.at["potential capital cost",tech] + (parameters.at["current capital cost",tech]-parameters.at["potential capital cost",tech])*(1+sum(model.generators[tech,yeart] for yeart in years if yeart < year))**(-parameters.at["learning parameter",tech])
model.fixed_cost_constraint = Constraint(techs, years, rule=fixed_cost_constraint)

def fixed_cost_constraint_storage(model,tech,year):
    if store_param.at["learning parameter",tech] == 0:
        return model.fixed_costs_storage[tech,year] == store_param.at["current capital cost",tech]
    else:
        return model.fixed_costs_storage[tech,year] == store_param.at["current capital cost",tech] * (1+sum(model.storage_built[tech,yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter",tech])
        # return model.fixed_costs_storage[tech,year] == store_param.at["potential capital cost",tech] + (store_param.at["current capital cost",tech]-store_param.at["potential capital cost",tech])*(1+sum(model.storage[tech,yeart] for yeart in years if yeart < year))**(-store_param.at["learning parameter",tech])
model.fixed_cost_constraint_storage = Constraint(storage, years, rule=fixed_cost_constraint_storage)

# def renewable_constraint(model,tech,techren,year):
#     if value(sum(model.generators_dispatch[techren,year] for techren in renewables)) > value(sum(model.generators_dispatch[tech,year] for tech in techs)*0.7):
#         return model.storage_dispatch["battery_store",year] == sum(model.generators_dispatch[techren,year] for tech in renewables)*0.3
        
# model.renewable_constraint = Constraint(techs,renewables, years, rule=renewable_constraint)

#%% Solving model

opt = SolverFactory('ipopt')
results = opt.solve(model,suffixes=['dual'],keepfiles=False)

print("Total cost (in billion euro) =","%.2f"% model.objective())

#%% Plotting

# file name
if "no_learning" in scenario: 
    filename = scenario
else:
    filename = scenario+"_"+learning_scenario


dispatch = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        dispatch.at[year,tech] = model.generators_dispatch[tech,year].value*8760

# for year in years:
# #     for tech in storage:
#     dispatch.at[year,"battery_store"] = model.storage_dispatch["battery_store", year].value*8760
#     dispatch.at[year,"hydrogen_storage"] = model.storage_dispatch["hydrogen_storage", year].value*8760
fig, ax = plt.subplots()
fig.set_dpi((400))
dispatch.plot(kind="area",stacked=True,color=colors,ax=ax,linewidth=0)
ax.set_xlabel("year")
ax.set_ylabel("Gross electricity generation [GWh]")
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures/{}-dispatch.png".format(filename),transparent=True)


capacities = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        capacities.at[year,tech] = model.generators[tech,year].value
    capacities.at[year,"battery_store"] = model.storage["battery_store", year].value
    capacities.at[year,"hydrogen_storage"] = model.storage["battery_store", year].value

        
fig, ax = plt.subplots()
fig.set_dpi((400))
capacities.plot(kind="area",stacked=True,color=colors,ax=ax,linewidth=0)
ax.set_xlabel("Year")
ax.set_ylabel("Installed capacity [GW]")
ax.set_ylim([0,1500])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures/{}-capacity.png".format(filename),transparent=True)

build_years = pd.DataFrame(0.,index=years,columns=techs+storage)
for year in years:
    for tech in techs:
        build_years.at[year,tech] = model.generators_built[tech,year].value

for year in years:
    for tech in storage:
        build_years.at[year,tech] = model.storage_built[tech, year].value

fig, ax = plt.subplots()
fig.set_dpi((200))
build_years.plot(kind="area",stacked=True,color=colors,ax=ax,linewidth=0)
ax.set_xlabel("year")
ax.set_ylabel("new capacity built [GW]")
ax.set_ylim([0,250])
fig.tight_layout()
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures/{}-new_capacity.png".format(filename),transparent=True)


level_cost = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        level_cost.at[year,tech] = model.fixed_costs[tech,year].value/8760. + parameters.at["marginal cost",tech]
    # for tech in storage:
    #     LCOE.at[year,tech] = model.fixed_costs_storage[tech, year].value/8760. + store_param.at["marginal cost",tech]

fig, ax = plt.subplots()
fig.set_dpi(400)
level_cost.plot(color=colors+color_storage,ax=ax,linewidth=3)
ax.set_xlabel("year")
ax.set_ylabel("LCOE [EUR/MWh]")
# ax.set_yscale("log")
ax.set_ylim([0,130])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures/{}-lcoe.png".format(filename),transparent=True)


emissions = pd.DataFrame(0.,index=years,columns=techs)
for year in years:
    for tech in techs:
        emissions.at[year,tech] = model.generators_dispatch[tech,year].value*8760* 1000 * parameters.at["specific emissions",tech]

fig, ax = plt.subplots()
fig.set_dpi(2000)
emissions.plot(color=colors+color_storage,ax=ax,linewidth=3)
ax.set_xlabel("year")
ax.set_ylabel("CO2 [t]")
# ax.set_yscale("log")
# ax.set_ylim([0,40])
ax.legend(bbox_to_anchor=(1, 1.05), ncol=1, fancybox=False, shadow=False)
ax.legend().set_visible(lgnd)
fig.savefig("Figures/{}-emissions.png".format(filename),transparent=True)
