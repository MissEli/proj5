# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 11:17:36 2021

Script for error estimation in ToF
"""

import numpy as np
import matplotlib.pyplot as plt
l = 38.35 #mm
dl = 1.95 #mm

E = [570.8090032, 1811.897686 , 1367.90409  , 1432.463957 ,
       2022.533437 , 4733.415934 , 2725.314222 ] #Particle energies after foil
dE = [4.457,5.317,15.585,6.942,5.406,6.683,4.951] #Std from fit 
rel_err = []
rel_err_l = []
dEE = [] #Experimental points
for i in range(len(E)):
    #dEE.append(dE[i]/E[i])
    dEE.append(0.04)
    rel_err.append(((dl/l)**2 + (dEE[i]/2)**2)**(1/2))
    

#Graph time
dls = [0.5,1,1.5,1.95,2.5] #Test these dl [mm]
for i in range(len(dls)):
    rel_err_l.append(round(dls[i]/l,3))
                     
dEEs = np.linspace(0,0.2,1000) #x vector

rel_errors = np.zeros((len(dls),len(dEEs)))

for i, dl in enumerate(dls):
    for j, E in enumerate(dEEs):
        rel_errors[i,j] = ((dl/l)**2+(E/2)**2)**(1/2)

colors = ['r','b','y','k','c']

plt.figure()
for i in range(len(dls)):
    plt.plot(dEEs,rel_errors[i,:],colors[i], label=f'\u0394l/l = {rel_err_l[i]}')

plt.plot(dEE,rel_err,'*',color='r', label='Experimental points')
plt.title('Relative error in tof over different errors in length and energy')
plt.legend(fontsize=12)
plt.ylabel('\u0394t / t')
plt.xlabel('\u0394E / E')
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)

plt.savefig('Rel_err_2.svg', bbox_inches='tight',format='svg')
ToF=[3.741226671982719,2.0998708941461015, 4.817580046997456, 4.707766474890197, 3.961949251685958, 2.589819098875753, 2.962748483671883]
err_abs = [rel_err[i]*t for t in ToF]

plt.savefig('Rel_err_3.svg', bbox_inches='tight',format='svg')


 

