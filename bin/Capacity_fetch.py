#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:40:46 2022

@author: frederikmelson
"""

import powerplantmatching as pm
import pandas as pd

geo = pm.data.GEO()

geo.head()

entsoe = pm.data.ENTSOE()

entsoe.head()

geo.powerplant.plot_map()

geo.powerplant.lookup().head(20).to_frame()

geo.powerplant.fill_missing_commyears().head()

print('Total capacity of entsoe is: \n  {} MW \n'.format(entsoe.Capacity.sum()));
print('The technology types are: \n {} '.format(geo.Technology.unique()))


pm.plot.fueltype_totals_bar([geo], keys=["ENTSOE", ])

