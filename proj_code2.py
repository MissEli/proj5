# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 14:45:49 2021

@author: ivewi
"""

# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt

file = np.loadtxt('protons_and_alphas.txt', dtype=str,delimiter='*',
                  skiprows=3, usecols=(1,2,3))

E = []
T = []
Ec = [0]*200*100 # channels


i = 0 # count events
n = 0 # count added events to channel
events = int(input('How many events should be read?: '))
for row in file:
    if i == events:
        break

    try:
        e = float(row[1])
        E.append(e)
        for j in range(len(Ec)):
            if j/100 == round(e,2):
                Ec[j] += 1  #add to channel
                n += 1

    except IndexError as ie:    #Only needed to find last index
        print("*** Index Error: ")
        print(i)          
    i += 1
    
print(i)
print(n)


# Plot
x1 = np.arange(0, len(E))
plt.plot(x1,E,'.')
plt.xlabel('events')
plt.ylabel('energy channel')
plt.show()

x2 = np.arange(0, 200, 0.01)
plt.plot(x2,Ec,'-')
plt.xlabel('energy channel')
plt.ylabel('events')
plt.show()

# Look at each event 
# print('(PH, ToF)')
# for n in ev:
#     print(ev[n])