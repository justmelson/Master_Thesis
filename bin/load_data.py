#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 06:15:41 2022

@author: frederikmelson
"""

def load_data(yes):
    #%% Importing data
    import requests
    import io
    import pandas as pd
    from annuity_fun import annuity
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
    
    return parameters, store_param