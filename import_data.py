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

#%% Importing data

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
                                     annuity(costs.at['coal','lifetime'],r)*costs.at['coal','investment']*(1+costs.at['coal','FOM']),
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
                                   costs.at['nuclear','fuel']/costs.at['nuclear','efficiency']] # from lazard #EUR/MWhel
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


#capital_cost = annuity(lifetime,discount rate)*Investment*(1+FOM) # in €/MW

store_param.round(3)

#%% Capacity factors 

ct = "DNK"
df_solar = pd.read_csv('data/pv_optimal.csv',sep=';',index_col=0)
df_onwind = pd.read_csv('data/onshore_wind_1979-2017.csv',sep=';',index_col=0)
df_offwind = pd.read_csv('data/offshore_wind_1979-2017.csv',sep=';',index_col=0)

year = pd.date_range('1979-01-01T00:00Z','1979-01-14T23:00Z',freq='H')
one_year = pd.date_range('2025-01-01T00:00Z','2025-01-14T23:00Z',freq='H')



CF_solar_one = df_solar[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]
CF_solar_one = CF_solar_one.reset_index()
# CF_solar_one = CF_solar_one.set_index(one_year)
CF_solar_one = CF_solar_one.drop(columns=["utc_time"])

CF_onwind_one = df_onwind[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]
CF_onwind_one = CF_onwind_one.reset_index()
# CF_onwind_one = CF_onwind_one.set_index(one_year)
CF_onwind_one = CF_onwind_one.drop(columns=["utc_time"])

CF_offwind_one = df_offwind[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]
CF_offwind_one = CF_offwind_one.reset_index()
# CF_offwind_one = CF_offwind_one.set_index(one_year)
CF_offwind_one = CF_offwind_one.drop(columns=["utc_time"])

#%% Demand
date_demand = pd.date_range('2015-01-01T00:00Z','2015-01-14T23:00Z',freq='H')
weekdemand = pd.date_range('2025-01-01T00:00:00Z','2025-01-14T23:00:00Z',freq='H')

df_elec = pd.read_csv('data/electricity_demand.csv', sep=';', index_col=0) # in MWh
df_elec = df_elec.sum(axis=1)
df_elec = df_elec[[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in date_demand]]


df_elec.index = pd.to_datetime(df_elec.index) #change index to datatime
df_elec = df_elec.reset_index()
df_elec = df_elec.set_index(weekdemand)
df_elec = df_elec.drop(columns=["utc_time"])

#%% Saving dataframes and lists

parameters.to_pickle("parameters.pkl")
store_param.to_pickle("store_param.pkl")
CF_solar_one.to_pickle("CF_solar_one.pkl")
CF_onwind_one.to_pickle("CF_onwind_one.pkl")
CF_offwind_one.to_pickle("CF_offwind_one.pkl")
df_elec.to_pickle("df_elec.pkl")

techs_file = "techs.pkl"
fossil_techs_file = "fossil_techs.pkl"
renewables_file = "renewables.pkl"
wind_file = "wind.pkl"
colors_file = "colors.pkl"
storage_file = "storage.pkl"
color_storage_file = "color_storage.pkl"

files = [techs_file,fossil_techs_file,renewables_file,wind_file,colors_file,storage_file,color_storage_file]
lists = [techs,fossil_techs,renewables,wind,colors,storage,color_storage]

for i in range(len(files)):
    open_file = open(files[i], "wb")
    pickle.dump(lists[i], open_file)
    open_file.close()



