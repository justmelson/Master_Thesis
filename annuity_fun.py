# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:00:28 2021

@author: justm
"""
#capital_cost = annuity(lifetime,discount rate)*Investment*(1+FOM) # in €/MW

def annuity(n,r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r/(1. - 1./(1.+r)**n)
    else:
        return 1/n