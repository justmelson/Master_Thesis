#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:54:53 2022

@author: frederikmelson
"""
import pandas as pd
import requests
import io
from annuity_fun import annuity
import pickle
import numpy as np

#%% Importing data

ct = "DEU"
# Downloading the csv files from pypsa GitHub account

url="https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2020.csv"



# link = url[i] # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

# Reading the downloaded content and turning it into a pandas dataframe

costs = pd.read_csv(io.StringIO(download.decode('utf-8')),index_col=[0,1]).sort_index()
#correct units to MW and EUR
costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
# costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

#min_count=1 is important to generate NaNs which are then filled by fillna
costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)


url2050 = "https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_2050.csv"
download = requests.get(url2050).content
costs2050 = pd.read_csv(io.StringIO(download.decode('utf-8')),index_col=[0,1]).sort_index()
#correct units to MW and EUR
costs2050.loc[costs2050.unit.str.contains("/kW"), "value"] *= 1e3
costs2050 = costs2050.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)


r = 0.01 # discount rate
fuel_cost_gas = costs.at["gas","fuel"] # in €/MWh_th from  https://tradingeconomics.com/commodity/eu-natural-gas
fuel_cost_coal = costs.at['coal','fuel'] #3.61 # costs.at['coal','fuel']


years = [2020,2026,2032,2038,2044,2050]
years5 = [2020,2025,2030,2035,2040,2045,2050]
years3 = [2020,2023,2026,2029,2032,2035,2038,2041,2044,2047,2050]

#%% Dataframe init

techs = ["offshore_wind","onshore_wind","solar_PV", "CCGT","OCGT","coal","nuclear"]
CC_techs = ["direct_air","CC"]
fossil_techs = ["CCGT","OCGT","coal","nuclear"]
renewables = ["offshore_wind","onshore_wind","solar_PV"]
wind = ["offshore_wind","onshore_wind"]
colors = ["dodgerblue","lightgreen","gold", 'coral',"peru","grey","plum","brown","darkkhaki"]
parameters = pd.DataFrame(columns=techs)
storage = ["battery_store","battery_inverter","hydrogen_storage","electrolysis","fuel_cell"]
color_storage = ["salmon","magenta","aqua","chartreuse","chocolate"]
store_param = pd.DataFrame(columns=storage)
demand = pd.DataFrame(columns= ["demand"])
CC_param = pd.DataFrame(columns=CC_techs)

#%% Technology data
parameters.loc["current capital cost"] = [costs.at['offwind','investment'],
                                     costs.at['onwind','investment'],
                                     costs.at['solar','investment'],
                                     costs.at['CCGT','investment'],
                                     costs.at['OCGT','investment'],
                                     costs.at['coal','investment'],
                                     costs.at['nuclear','investment']] # EUR/MW
parameters.loc["2050 capital cost"] = [costs2050.at['offwind','investment'],
                                     costs2050.at['onwind','investment'],
                                     costs2050.at['solar','investment'],
                                     costs2050.at['CCGT','investment'],
                                     costs2050.at['OCGT','investment'],
                                     costs2050.at['coal','investment'],
                                     costs2050.at['nuclear','investment']] # EUR/MW
parameters.loc["current annuity"] = [annuity(costs.at['offwind','lifetime'],r)*costs.at['offwind','investment']*(1+costs.at['offwind','FOM']*0.01),
                                     annuity(costs.at['onwind','lifetime'],r)*costs.at['onwind','investment']*(1+costs.at['onwind','FOM']*0.01),
                                     annuity(costs.at['solar','lifetime'],r)*costs.at['solar','investment']*(1+costs.at['solar','FOM']*0.01),
                                     annuity(costs.at['CCGT','lifetime'],r)*costs.at['CCGT','investment']*(1+costs.at['CCGT','FOM']*0.01),
                                     annuity(costs.at['OCGT','lifetime'],r)*costs.at['OCGT','investment']*(1+costs.at['OCGT','FOM']*0.01),
                                     annuity(costs.at['coal','lifetime'],r)*costs.at['coal','investment']*(1+costs.at['coal','FOM']*0.01),
                                     annuity(costs.at['nuclear','lifetime'],r)*costs.at['nuclear','investment']*(1+costs.at['nuclear','FOM']*0.01)] # EUR/MW/a
parameters.loc["2050 annuity"] = [annuity(costs2050.at['offwind','lifetime'],r)*costs2050.at['offwind','investment']*(1+costs2050.at['offwind','FOM']*0.01),
                                     annuity(costs2050.at['onwind','lifetime'],r)*costs2050.at['onwind','investment']*(1+costs2050.at['onwind','FOM']*0.01),
                                     annuity(costs2050.at['solar','lifetime'],r)*costs2050.at['solar','investment']*(1+costs2050.at['solar','FOM']*0.01),
                                     annuity(costs2050.at['CCGT','lifetime'],r)*costs2050.at['CCGT','investment']*(1+costs2050.at['CCGT','FOM']*0.01),
                                     annuity(costs2050.at['OCGT','lifetime'],r)*costs2050.at['OCGT','investment']*(1+costs2050.at['OCGT','FOM']*0.01),
                                     annuity(costs2050.at['coal','lifetime'],r)*costs2050.at['coal','investment']*(1+costs2050.at['coal','FOM']*0.01),
                                     annuity(costs2050.at['nuclear','lifetime'],r)*costs2050.at['nuclear','investment']*(1+costs2050.at['nuclear','FOM']*0.01)] # EUR/MW/a


parameters.loc["learning rate"] = [0.19,0.32,0.47,0.34,0.15,0.083,0] # [0.12,0.12,0.23,0.14,0.15]
parameters.loc["learning parameter"] = [0,0,0,0,0,0,0]
parameters.loc["marginal cost"] = [0,
                                   0,
                                   0,
                                   fuel_cost_gas/costs.at['CCGT','efficiency'],
                                   fuel_cost_gas/costs.at['OCGT','efficiency'],
                                   fuel_cost_coal/costs.at['coal','efficiency'],
                                   costs.at['nuclear','fuel']/costs.at['nuclear','efficiency']] # from lazard #EUR/MWhel
parameters.loc["specific emissions"] = [0.,0.,0.,
                                        costs.at['gas','CO2 intensity']/costs.at['CCGT','efficiency'],
                                        costs.at['gas','CO2 intensity']/costs.at['OCGT','efficiency'],
                                        costs.at['coal','CO2 intensity']/costs.at['coal','efficiency'],
                                        0] #tcO2/MWhel
parameters.loc["lifetime"] = [27,27,32.5,25,25,40,40]  #years
parameters.loc["ramp rate"] = [1,1,1,0.6,0.5,0.5,0.25] # ramp rate for capacities for hourly balancing
parameters.loc["existing age"] = [10,10,5,14,14,20,15] # [0,0,0,0,0,0] years
parameters.loc["existing capacity"] = [25,174,141,96,96,94,117] #[26,174,123,112,112,128] #[0,0,0,0,0,0] #GW
parameters.loc["current LCOE"] = parameters.loc["current annuity"]/8760 + parameters.loc["marginal cost"]
parameters.loc["efficiency"] = [1,
                                1,
                                1,
                                costs.at['CCGT','efficiency'],
                                costs.at['OCGT','efficiency'],
                                costs.at['coal','efficiency'],
                                costs.at['nuclear','efficiency']]
parameters.loc["capacity factor"] = [0.52,0.44,0.21,0.63,0.63,0.83,0.85] # Avg yearly capacity factor

parameters.round(3)

# annuity(20,r)*1042000*(1+540) for Li-ion battery
# [annuity(costs.at['battery storage','lifetime'],r)*costs.at['battery storage','investment']*(1+costs.at['battery inverter','FOM']),


store_param.loc["current capital cost"] = [1042000, #Li-ion battery cost from Danish energy agency
                                      costs.at['battery inverter','investment'],
                                      costs.at['H2 (l) storage tank','investment'],
                                      costs.at['electrolysis','investment'],
                                      costs.at['fuel cell','investment']] # EUR/MW
store_param.loc["2050 capital cost"] = [975000, #Li-ion battery cost from Danish energy agency
                                      costs2050.at['battery inverter','investment'],
                                      costs2050.at['H2 (l) storage tank','investment'],
                                      costs2050.at['electrolysis','investment'],
                                      costs2050.at['fuel cell','investment']] # EUR/MW
store_param.loc["current annuity"] = [annuity(20,r)*1042000*(1+costs.at['battery inverter','FOM']*0.01), #Li-ion battery cost from Danish energy agency
                                      annuity(costs.at['battery inverter','lifetime'],r)*costs.at['battery inverter','investment']*(1+costs.at['battery inverter','FOM']*0.01),
                                      annuity(costs.at['H2 (l) storage tank','lifetime'],r)*costs.at['H2 (l) storage tank','investment']*(1+costs.at['H2 (l) storage tank','FOM']*0.01),
                                      annuity(costs.at['electrolysis','lifetime'],r)*costs.at['electrolysis','investment']*(1+costs.at['electrolysis','FOM']*0.01),
                                      annuity(costs.at['fuel cell','lifetime'],r)*costs.at['fuel cell','investment']*(1+costs.at['fuel cell','FOM']*0.01)] # EUR/MW/a
store_param.loc["2050 annuity"] = [annuity(20,r)*975000*(1+costs2050.at['battery inverter','FOM']*0.01), #Li-ion battery cost from Danish energy agency
                                      annuity(costs2050.at['battery inverter','lifetime'],r)*costs2050.at['battery inverter','investment']*(1+costs2050.at['battery inverter','FOM']*0.01),
                                      annuity(costs2050.at['H2 (l) storage tank','lifetime'],r)*costs2050.at['H2 (l) storage tank','investment']*(1+costs2050.at['H2 (l) storage tank','FOM']*0.01),
                                      annuity(costs2050.at['electrolysis','lifetime'],r)*costs2050.at['electrolysis','investment']*(1+costs2050.at['electrolysis','FOM']*0.01),
                                      annuity(costs2050.at['fuel cell','lifetime'],r)*costs2050.at['fuel cell','investment']*(1+costs2050.at['fuel cell','FOM']*0.01)] # EUR/MW/a
store_param.loc["learning rate"] = [0.12,0.1,0.1,0.18,0.18] # not sure about inverter learning rate
store_param.loc["learning parameter"] = [0,0,0,0,0]
store_param.loc["marginal cost"] = [2.,0.,0.,0.,0.] #EUR/MWhel
store_param.loc["specific emissions"] = [0.,0.,0.,0.,0.] #tcO2/MWhel
store_param.loc["lifetime"] = [20,10,20,25,10]  #years
store_param.loc["existing age"] = [0,0,0,0,0] #years
store_param.loc["existing capacity"] = [0,0,0,0,0] #[20,20,20,20,20] #[25,195,141,172] #GW
store_param.loc["efficiency"] = [1,
                                 0.92,
                                 costs.at['H2 (l) storage tank','efficiency'],
                                 costs.at['electrolysis','efficiency'],
                                 costs.at['fuel cell','efficiency']] #[20,20,20,20,20] #[25,195,141,172] #GW

store_param.loc["current LCOE"] = store_param.loc["current annuity"]/8760 + store_param.loc["marginal cost"]

store_param.round(3)
CC_param.loc["current capital cost"] = [costs.at['direct air capture','investment'], #EUR/(tCO2/h)
                                        costs.at['cement capture','investment']]
CC_param.loc["2050 capital cost"] = [costs2050.at['direct air capture','investment'], #EUR/(tCO2/h)
                                        costs2050.at['cement capture','investment']]
CC_param.loc["current annuity"] = [annuity(costs.at['direct air capture','lifetime'],r)*costs.at['direct air capture','investment']*(1+costs.at['direct air capture','FOM']*0.01), #EUR/(tCO2/h)
                                        annuity(costs.at['cement capture','lifetime'],r)*costs.at['cement capture','investment']*(1+costs.at['cement capture','FOM']*0.01)]
CC_param.loc["2050 annuity"] = [annuity(costs2050.at['direct air capture','lifetime'],r)*costs2050.at['direct air capture','investment']*(1+costs2050.at['direct air capture','FOM']*0.01), #EUR/(tCO2/h)
                                        annuity(costs2050.at['cement capture','lifetime'],r)*costs2050.at['cement capture','investment']*(1+costs2050.at['cement capture','FOM']*0.01)]
CC_param.loc["learning rate"] = [0.15,0.15] 
CC_param.loc["learning parameter"] = [0,0]
CC_param.loc["power input"] = [0.5,0.12] #MWhel/tCO2
CC_param.loc["specific emissions"] = [-2,-8.33] #tcO2/MWhel
CC_param.loc["lifetime"] = [costs.at['direct air capture','lifetime'],costs.at['cement capture','lifetime']]  #years
CC_param.loc["existing age"] = [0,0] #years
CC_param.loc["existing capacity"] = [0,0] # #GW

CC_param.round(3)


#%% Capacity factors 

# ct = "DEU"
# df_solar = pd.read_csv('data/pv_optimal.csv',sep=';',index_col=0)
# df_onwind = pd.read_csv('data/onshore_wind_1979-2017.csv',sep=';',index_col=0)
# df_offwind = pd.read_csv('data/offshore_wind_1979-2017.csv',sep=';',index_col=0)

# year = pd.date_range('1979-01-01T00:00Z','1979-01-14T23:00Z',freq='H')
# one_year = pd.date_range('2025-01-01T00:00Z','2025-01-14T23:00Z',freq='H')



# CF_solar_one = df_solar[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]
# CF_solar_one = CF_solar_one.reset_index()
# # CF_solar_one = CF_solar_one.set_index(one_year)
# CF_solar_one = CF_solar_one.drop(columns=["utc_time"])

# CF_onwind_one = df_onwind[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]
# CF_onwind_one = CF_onwind_one.reset_index()
# # CF_onwind_one = CF_onwind_one.set_index(one_year)
# CF_onwind_one = CF_onwind_one.drop(columns=["utc_time"])

# CF_offwind_one = df_offwind[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]
# CF_offwind_one = CF_offwind_one.reset_index()
# # CF_offwind_one = CF_offwind_one.set_index(one_year)
# CF_offwind_one = CF_offwind_one.drop(columns=["utc_time"])

# Cf_solar = pd.read_excel("Cf_cluster.xlsx","Solar")
# Cf_onshore = pd.read_excel("Cf_cluster.xlsx","Onshore")
# Cf_offshore = pd.read_excel("Cf_cluster.xlsx","Offshore")

ct = "DEU" 
res = 3


cf_solar_raw = pd.read_excel('data/capacityfactor_twoweeks.xlsx','pv',index_col=0)
cf_solar_raw = cf_solar_raw[ct]
cf_solar = cf_solar_raw.to_numpy()
cf_solar3h = np.mean(cf_solar.reshape(-1,res),axis=1)
cf_solar3h = pd.DataFrame(cf_solar3h)

cf_onshore_raw = pd.read_excel('data/capacityfactor_twoweeks.xlsx','onshore',index_col=0)
cf_onshore_raw = cf_onshore_raw[ct]
cf_onshore = cf_onshore_raw.to_numpy()
cf_onshore3h = np.mean(cf_onshore.reshape(-1,res),axis=1)
cf_onshore3h = pd.DataFrame(cf_onshore3h)


cf_offshore_raw = pd.read_excel('data/capacityfactor_twoweeks.xlsx','offshore',index_col=0)
cf_offshore_raw = cf_offshore_raw[ct]
cf_offshore = cf_offshore_raw.to_numpy()
cf_offshore3h = np.mean(cf_offshore.reshape(-1,res),axis=1)
cf_offshore3h = pd.DataFrame(cf_offshore3h)


cf_solar3h.to_pickle("cf_solar3h.pkl")
cf_onshore3h.to_pickle("cf_onshore3h.pkl")
cf_offshore3h.to_pickle("cf_offshore3h.pkl")


cf_solar_raw = pd.read_excel('data/capacityfactor_4days.xlsx','pv',index_col=0)
cf_solar_raw = cf_solar_raw[ct]
cf_solar = cf_solar_raw.to_numpy()
cf_solar3h = np.mean(cf_solar.reshape(-1,res),axis=1)
cf_solar3h = pd.DataFrame(cf_solar3h)

cf_onshore_raw = pd.read_excel('data/capacityfactor_4days.xlsx','onshore',index_col=0)
cf_onshore_raw = cf_onshore_raw[ct]
cf_onshore = cf_onshore_raw.to_numpy()
cf_onshore3h = np.mean(cf_onshore.reshape(-1,res),axis=1)
cf_onshore3h = pd.DataFrame(cf_onshore3h)


cf_offshore_raw = pd.read_excel('data/capacityfactor_4days.xlsx','offshore',index_col=0)
cf_offshore_raw = cf_offshore_raw[ct]
cf_offshore = cf_offshore_raw.to_numpy()
cf_offshore3h = np.mean(cf_offshore.reshape(-1,res),axis=1)
cf_offshore3h = pd.DataFrame(cf_offshore3h)

# cf_solar_raw = pd.read_excel('data/solar_agg_4d3h.xlsx',index_col=1)
# cf_solar_raw = cf_solar_raw[ct]
# cf_solar = cf_solar_raw.to_numpy()
# cf_solar3h = np.mean(cf_solar.reshape(-1,res),axis=1)
# cf_solar3h = pd.DataFrame(cf_solar3h)

# cf_onshore_raw = pd.read_excel('data/onshore_agg_4d3h.xlsx',index_col=1)
# cf_onshore_raw = cf_onshore_raw[ct]
# cf_onshore = cf_onshore_raw.to_numpy()
# cf_onshore3h = np.mean(cf_onshore.reshape(-1,res),axis=1)
# cf_onshore3h = pd.DataFrame(cf_onshore3h)


# cf_offshore_raw = pd.read_excel('data/offshore_agg_4d3h.xlsx',index_col=1)
# cf_offshore_raw = cf_offshore_raw[ct]
# cf_offshore = cf_offshore_raw.to_numpy()
# cf_offshore3h = np.mean(cf_offshore.reshape(-1,res),axis=1)
# cf_offshore3h = pd.DataFrame(cf_offshore3h)


cf_solar3h.to_pickle("cf_solar3h4d.pkl")
cf_onshore3h.to_pickle("cf_onshore3h4d.pkl")
cf_offshore3h.to_pickle("cf_offshore3h4d.pkl")


cf_solar_raw = pd.read_excel('data/capacityfactor_2days.xlsx','pv',index_col=0)
cf_solar_raw = cf_solar_raw[ct]
cf_solar = cf_solar_raw.to_numpy()
cf_solar3h = np.mean(cf_solar.reshape(-1,res),axis=1)
cf_solar3h = pd.DataFrame(cf_solar3h)

cf_onshore_raw = pd.read_excel('data/capacityfactor_2days.xlsx','onshore',index_col=0)
cf_onshore_raw = cf_onshore_raw[ct]
cf_onshore = cf_onshore_raw.to_numpy()
cf_onshore3h = np.mean(cf_onshore.reshape(-1,res),axis=1)
cf_onshore3h = pd.DataFrame(cf_onshore3h)


cf_offshore_raw = pd.read_excel('data/capacityfactor_2days.xlsx','offshore',index_col=0)
cf_offshore_raw = cf_offshore_raw[ct]
cf_offshore = cf_offshore_raw.to_numpy()
cf_offshore3h = np.mean(cf_offshore.reshape(-1,res),axis=1)
cf_offshore3h = pd.DataFrame(cf_offshore3h)



cf_solar3h.to_pickle("cf_solar3h2d.pkl")
cf_onshore3h.to_pickle("cf_onshore3h2d.pkl")
cf_offshore3h.to_pickle("cf_offshore3h2d.pkl")

#%% Demand
week_summer = pd.date_range('2015-06-19T00:00:00Z','2015-06-21T23:00:00Z',freq='H')
week_winter = pd.date_range('2015-12-23T00:00:00Z','2015-12-25T23:00:00Z',freq='H')

# d3_summer   = pd.date_range('2015-06-19T00:00:00Z','2015-06-21T23:00:00Z',freq='H')
# d3_winter   = pd.date_range('2015-12-23T00:00:00Z','2015-12-25T23:00:00Z',freq='H')
# weekdemand = pd.date_range('2025-01-01T00:00:00Z','2025-01-14T23:00:00Z',freq='H')

df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0) # in MWh
df_elec = df_elec.sum(axis=1)
df_elec = df_elec/1000
df_elec_summer = df_elec[[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in week_summer]]
df_elec_winter = df_elec[[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in week_winter]]


demand2w_raw = pd.concat([df_elec_summer,df_elec_winter])
demand2w = demand2w_raw.to_numpy()
demand2w3h = np.mean(demand2w.reshape(-1,res),axis=1)
demand2w3h = pd.DataFrame(demand2w3h)

demand_elec = {}

for year in years:
    demand_elec[year] = pd.DataFrame()
    for j in range(len(demand2w3h)):
        if year == 2020:
            demand_elec[year].at[j,0] = demand2w3h.at[j,0]
        else:
            demand_elec[year].at[j,0] = demand_elec[year-(years[1]-years[0])].at[j,0]*1.042

demand_elec5y = {}

for year in years5:
    demand_elec5y[year] = pd.DataFrame()
    for j in range(len(demand2w3h)):
        if year == 2020:
            demand_elec5y[year].at[j,0] = demand2w3h.at[j,0]
        else:
            demand_elec5y[year].at[j,0] = demand_elec5y[year-(years5[1]-years5[0])].at[j,0]*1.035895

demand_elec3y = {}

for year in years3:
    demand_elec3y[year] = pd.DataFrame()
    for j in range(len(demand2w3h)):
        if year == 2020:
            demand_elec3y[year].at[j,0] = demand2w3h.at[j,0]
        else:
            demand_elec3y[year].at[j,0] = demand_elec3y[year-(years3[1]-years3[0])].at[j,0]*1.022696



# df_elec.index = pd.to_datetime(df_elec.index) #change index to datetime
# df_elec = df_elec.reset_index()
# df_elec = df_elec.set_index(weekdemand)
# df_elec = df_elec.drop(columns=["utc_time"])

#%% Saving dataframes and lists

parameters.to_pickle("parameters.pkl")
store_param.to_pickle("store_param.pkl")
CC_param.to_pickle("CC_param.pkl")
# CF_solar_one.to_pickle("CF_solar_one.pkl")
# CF_onwind_one.to_pickle("CF_onwind_one.pkl")
# CF_offwind_one.to_pickle("CF_offwind_one.pkl")
df_elec.to_pickle("df_elec.pkl")
# demand2w3h.to_pickle("demand2w3h.pkl")
demand2w3h.to_pickle("demand6d3h.pkl")


a_file = open("demand_elec.pkl", "wb")
pickle.dump(demand_elec, a_file)
a_file.close()

a_file = open("demand_elec5y.pkl", "wb")
pickle.dump(demand_elec5y, a_file)
a_file.close()

a_file = open("demand_elec3y.pkl", "wb")
pickle.dump(demand_elec3y, a_file)
a_file.close()



CC_file             = "CC.pkl"
techs_file          = "techs.pkl"
fossil_techs_file   = "fossil_techs.pkl"
renewables_file     = "renewables.pkl"
wind_file           = "wind.pkl"
colors_file         = "colors.pkl"
storage_file        = "storage.pkl"
color_storage_file  = "color_storage.pkl"

files = [techs_file,fossil_techs_file,renewables_file,wind_file,colors_file,storage_file,color_storage_file,CC_file]
lists = [techs,fossil_techs,renewables,wind,colors,storage,color_storage,CC_techs]

for i in range(len(files)):
    open_file = open(files[i], "wb")
    pickle.dump(lists[i], open_file)
    open_file.close()


# Cf_solar.to_pickle("Cf_solar.pkl")
# Cf_onshore.to_pickle("Cf_onshore.pkl")
# Cf_offshore.to_pickle("Cf_offshore.pkl")

