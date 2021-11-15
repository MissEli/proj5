# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

filenames = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']
energies = [577000,1815000,1400000,1464000,2050000,4750000,2730000] #eV
loss_avg = []
loss_std = []
for i, name in enumerate(filenames):
    file = np.loadtxt(name, dtype=str,
                      skiprows=12,max_rows=1000,usecols=(3))

    data =[]
    for line in file:
        newline = line.replace(',','0.')
        data.append(float(newline))
        
    
    energy = np.full(len(data),energies[i])
    loss = np.subtract(energy,data)
    loss_avg.append(-1*np.mean(loss))
    loss_std.append(np.std(loss))
    
    x = range(len(data))
    plt.figure()
    plt.plot(x, data ,linestyle='None',marker='.')
    bottom, top = plt.ylim()
    plt.ylim(0,1.2*top)
    plt.xlabel('Event number')
    plt.ylabel('Energy loss [eV]')

print(f'Average losses {loss_avg}')
print(f'Standard deviations {loss_std}')
    
