# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:40:12 2021

@author: justm
"""

import pandas as pd
import numpy as np
import matplotlib as plt 


ct = "DNK"
hours_in_x=pd.date_range('1979-01-01T00:00Z','2014-12-31T23:00Z',freq='H')

year = pd.date_range('1979-01-01T00:00Z','1979-12-31T23:00Z',freq='H')

# df_elec = pd.read_csv('electricity_demand.csv',sep=';',index_col=0) # in MWh
# #print(df_elec[ct].head())
# df_elec.index = np.arange(0,8760)

# df_elec = df_elec["DNK"]

# hours = 8760*30
# hour_range = range(hours)
# count = -1
# demand = np.ones(hours)

# for i in range(0,hours):
#     count = count+1
#     if count < 8759:
#         demand[i] = df_elec[count]
#     if count == 8759:
#         demand[i] = df_elec[count]
#         count = 0
        
# demand3h = np.mean(demand.reshape(-1,3), axis=1)



df_solar = pd.read_csv('data/pv_optimal.csv',sep=';',index_col=0)
# CF_solar=df_solar[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in hours_in_x]]

# hours_new=pd.date_range('2025-01-01T00:00Z','2060-12-31T23:00Z',freq='H')



# CF_new = CF_solar.reset_index()


# h = hours_new.strftime("%Y-%m-%dT%H:%M:%SZ")

# CF_new = CF_new.set_index(h)

# CF_solar = CF_new.drop(columns=["utc_time"])


ct = "DNK"
df_solar = pd.read_csv('data/pv_optimal.csv',sep=';',index_col=0)
year = pd.date_range('1979-01-01T00:00Z','1979-12-31T23:00Z',freq='H')
one_year = pd.date_range('2025-01-01T00:00Z','2025-12-31T23:00Z',freq='H')
CF_solar_one =df_solar[ct][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in year]]

CF_solar_one = CF_solar_one.reset_index()
CF_solar_one = CF_solar_one.set_index(one_year)
CF_solar_one = CF_solar_one.drop(columns=["utc_time"])
