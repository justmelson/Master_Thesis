#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:49:48 2022

@author: frederikmelson
"""

# load_ext autoreload
# autoreload 2
import copy
import os
import pandas as pd
from sewar.full_ref import uqi,rmse
import matplotlib.pyplot as plt
from matplotlib import image
import tsam.timeseriesaggregation as tsam
# matplotlib inline
import numpy as np
import matplotlib.cm as cm
ct = "DEU"

def plotTS(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.hot, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel('Hour')
    plt.xlabel('Day [0-6: summer] [7-13: winter]')
    plt.title('Solar power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
    
def plotTS3h(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.hot, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel(str(res)+' hour steps')
    plt.xlabel('Day [0-6: summer] [7-13: winter]')
    plt.title('Solar power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
    plt.yticks(ticks=[-0.5,(24/res)/2-0.5,24/res-0.5],labels=['0','12','24'])


def plotTSonwind(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.summer, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel('Hour')
    plt.xlabel('Day [0-6: summer] [7-13: winter]')
    plt.title('Onshore wind power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
    
    
def plotTSonwind3h(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.summer, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel(str(res)+' hour steps')
    plt.xlabel('Day [0-6: summer] [7-13: winter]')
    plt.title('Onshore wind power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
    plt.yticks(ticks=[-0.5,(24/res)/2-0.5,24/res-0.5],labels=['0','12','24'])

    
def plotTSoffwind(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.winter, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel('Hour')
    plt.xlabel('Day [0-6: summer] [7-13: winter]')
    plt.title('Offshore wind power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')

    
def plotTSoffwind3h(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.winter, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel(str(res)+' hour steps')
    plt.xlabel('Day [0-6: summer] [7-13: winter]')
    plt.title('Offshore wind power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
    plt.yticks(ticks=[-0.5,(24/res)/2-0.5,24/res-0.5],labels=['0','12','24'])
    
#%% Manual picking of time series
ct = "DEU" 
res = 1


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

# plotting
plotTS(cf_solar_raw, 24, vmin = cf_solar_raw.min(), vmax = cf_solar_raw.max())
plotTS3h(cf_solar3h, 24/res, vmin = cf_solar3h.min(), vmax = cf_solar3h.max())

plotTSonwind(cf_onshore_raw, 24, vmin = cf_onshore_raw.min(), vmax = cf_onshore_raw.max())
plotTSonwind3h(cf_onshore3h, 24/res, vmin = cf_onshore3h.min(), vmax = cf_onshore3h.max())

plotTSoffwind(cf_offshore_raw, 24, vmin = cf_offshore_raw.min(), vmax = cf_offshore_raw.max())
plotTSoffwind3h(cf_offshore3h, 24/res, vmin = cf_offshore3h.min(), vmax = cf_offshore3h.max())

#%% Time series aggregation

n_days = 14
hpp = 24 # hours per period

cf_solar_raw = pd.read_csv('data/pv_optimal.csv',sep=";",index_col=0)
cf_solar_raw_ct = pd.DataFrame(cf_solar_raw[ct])
cf_solar_rawagg = tsam.TimeSeriesAggregation(cf_solar_raw_ct, noTypicalPeriods = n_days, hoursPerPeriod = hpp,
                                        clusterMethod = 'k_means', 
                                        extremePeriodMethod = 'new_cluster_center',
                                        addPeakMin = [ct], addPeakMax = [ct] )
typPeriods_solar = cf_solar_rawagg.createTypicalPeriods()

print(cf_solar_rawagg.accuracyIndicators())
predictedPeriodsWithEx = cf_solar_rawagg.predictOriginalData()


# fig, axes = plt.subplots(figsize = [6, 2], dpi = 100, nrows = 1, ncols = 1)
# cf_solar_raw_ct[ct].sort_values(ascending=False).reset_index(drop=True).plot(label = 'Original')
# predictedPeriodsWithEx[ct].sort_values(
#     ascending=False).reset_index(drop=True).plot(label = '14 typ days \n + peak period')
# plt.legend()
# plt.xlabel('Hours [h]')
# plt.ylabel('Capacity factor')


cf_onshore_raw = pd.read_csv('data/onshore_wind_1979-2017.csv',sep=";",index_col=0)
cf_onshore_raw_ct = pd.DataFrame(cf_onshore_raw[ct])
cf_onshore_rawagg = tsam.TimeSeriesAggregation(cf_onshore_raw_ct, noTypicalPeriods = n_days, hoursPerPeriod = hpp,
                                        clusterMethod = 'k_means', 
                                        extremePeriodMethod = 'new_cluster_center',
                                        addPeakMin = [ct], addPeakMax = [ct] )
typPeriods_onshore = cf_onshore_rawagg.createTypicalPeriods()
print(cf_onshore_rawagg.accuracyIndicators())



cf_offshore_raw = pd.read_csv('data/offshore_wind_1979-2017.csv',sep=";",index_col=0)
cf_offshore_raw_ct = pd.DataFrame(cf_onshore_raw[ct])
cf_offshore_rawagg = tsam.TimeSeriesAggregation(cf_offshore_raw_ct, noTypicalPeriods = n_days, hoursPerPeriod = hpp,
                                        clusterMethod = 'k_means', 
                                        extremePeriodMethod = 'new_cluster_center',
                                        addPeakMin = [ct], addPeakMax = [ct] )
typPeriods_offshore = cf_offshore_rawagg.createTypicalPeriods()
print(cf_offshore_rawagg.accuracyIndicators())



def aggplotTS(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.hot, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel('Hour')
    plt.xlabel('Aggregated days')
    plt.title('Solar power')
    fig.subplots_adjust(right = 1.2)
    plt.grid(True)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
def aggplotTSonwind(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.summer, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel('Hour')
    plt.xlabel('Aggregated days')
    plt.title('Onshore wind power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')
def aggplotTSoffwind(data, periodlength, vmin, vmax):
    fig, axes = plt.subplots(figsize = [12, 4], dpi = 400, nrows = 1, ncols = 1)
    stacked, timeindex = tsam.unstackToPeriods(copy.deepcopy(data), periodlength)
    cax = axes.imshow(stacked.values.T,cmap=cm.winter, interpolation = 'nearest', vmin = vmin, vmax = vmax)
    axes.set_aspect('auto')  
    axes.set_ylabel('Hour')
    plt.xlabel('Aggregated days')
    plt.title('Offshore wind power')
    fig.subplots_adjust(right = 1.2)
    cbar=plt.colorbar(cax)    
    cbar.set_label('capacity factor')

     
# plotting
aggplotTS(typPeriods_solar,    hpp, vmin = typPeriods_solar.min(),     vmax = typPeriods_solar.max())
aggplotTSonwind(typPeriods_onshore,  hpp, vmin = typPeriods_onshore.min(),   vmax = typPeriods_onshore.max())
aggplotTSoffwind(typPeriods_offshore, hpp, vmin = typPeriods_offshore.min(),  vmax = typPeriods_offshore.max())

#%% Image correlation 
im1 = image.imread('Capacity_factors_plots/solar1h.png')
im2 = image.imread('Capacity_factors_plots/solar3h.png')

cor = uqi(im1, im2)
print(cor)
