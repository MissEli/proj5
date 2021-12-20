# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

filenames = ['RANGE_p577.txt','RANGE_p1815.txt','RANGE_a1400.txt','RANGE_a1464.txt','RANGE_a2050.txt','RANGE_a4750.txt','RANGE_t2730.txt']
energies = [577000,1815000,1400000,1464000,2050000,4750000,2730000] #eV
loss_avg = []
loss_std = []
range_avg = []
for i, name in enumerate(filenames):
    file = np.loadtxt(name, dtype=str,
                      skiprows=17,max_rows=10000,usecols=(1,2,3))

    data =[]
    i = 0
    for line in file:
        # i += 1
        # if i == 10:
        #     break
        newline = np.char.replace(line, ',','.')
        x = float(newline[0])*10**(-8)
        y = float(newline[1])*10**(-8)
        z = float(newline[2])*10**(-8)
        data.append(math.sqrt(x**2+y**2+z**2))
 
    range_avg.append(np.mean(data))

#     energy = np.full(len(data),energies[i])
#     loss = np.subtract(energy,data)
#     loss_avg.append(-1*np.mean(loss))
#     loss_std.append(np.std(loss))
    
#     x = range(len(data))
#     plt.figure()
#     plt.plot(x, data ,linestyle='None',marker='.')
#     bottom, top = plt.ylim()
#     plt.ylim(0,1.2*top)
#     plt.xlabel('Event number')
#     plt.ylabel('Energy loss [eV]')

# print(f'Average losses {loss_avg}')
# print(f'Standard deviations {loss_std}')
print(f'Ranges of ions in Si {range_avg}')

