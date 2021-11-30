import numpy as np
import matplotlib.pyplot as plt

def load(fname):
    file = np.loadtxt(fname, dtype=float,delimiter=',', skiprows=1)
    E = file[:,0]
    return E

Ec = []
file = np.loadtxt('ec.txt', dtype=float, skiprows=1)

for line in file:
    Ec.append(line)

energies = []
fname = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']
for name in fname:
        energy = load(name)
        for x in energy:
            energies.append(x)

# plot events per channel
plt.figure()
x2 = np.arange(0, 200, 0.01)
plt.plot(x2,Ec,'r-')
plt.xlabel('energy channel')
plt.ylabel('events')

# plt.figure()
plt.hist(energies,bins=20000,range=(0,200))
plt.xlabel('energy channel')
plt.ylabel('events')


plt.show()