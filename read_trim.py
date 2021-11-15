# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:33:44 2021

@author: ivewi
"""

import numpy as np

filename = input('Specify file to be read: ')
file = np.loadtxt(filename, dtype=str,
                  skiprows=12,max_rows=1000,usecols=(3))

data =[]
for line in file:
    newline = line.replace(',','')
    print(newline)
    data.append(newline)

    
