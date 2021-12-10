import numpy as np
import matplotlib.pyplot as plt

# def load(fname):
#     file = np.loadtxt(fname, dtype=float,delimiter=',', skiprows=1)
#     E = file[:,0]
#     return E

# Ec = []
# file = np.loadtxt('ec.txt', dtype=float, skiprows=1)

# for line in file:
#     Ec.append(line)

# energies = []
# fname = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']
# for name in fname:
#         energy = load(name)
#         for x in energy:
#             energies.append(x)

# # plot events per channel
# plt.figure()
# x2 = np.arange(0, 200, 0.01)
# plt.plot(x2,Ec,'r-')
# plt.xlabel('energy channel')
# plt.ylabel('events')

# # plt.figure()
# plt.hist(energies,bins=20000,range=(0,200))
# plt.xlabel('energy channel')
# plt.ylabel('events')




# x = [16.87235773026404+0.2629641664630598, 53.22593825320579-0.2881365468471792, 40.298930370938926, 42.008345063338794, 59.41721439976735, 139.3187252301411, 79.67688501006664]
x = [16.87235773026404, 53.22593825320579, 40.298930370938926, 42.008345063338794, 59.41721439976735, 139.3187252301411, 79.67688501006664]
y = [570.8090032, 1811.897686, 1370.577657701188, 1428.9360146616714, 2023.2640121515062, 4751.052015526256, 2714.916596298767]
Etrue = [0.577,1.815,1.4, 1.464, 2.05, 4.75,2.73]
Eloss = [-0.0061909968, -0.0031023139999999997, -0.03209591, -0.031536043, -0.027466562999999996, -0.016584065999999998, -0.004685778]
E = []

for i in range(len(Etrue)):
    E.append(1000*(Etrue[i]+Eloss[i]))



k = (E[1]-E[0])/(x[1]-x[0])
m = E[0]-k*x[0]
xvec = np.linspace(0,140)
xx = []

print(k)
print(m)
line = [k*xx+m for xx in xvec]
y = [k*peak+m for peak in x]

PHD = []
for i in range(len(E)):
    PHD.append(E[i]-y[i])
print(PHD)

print(y)
print(E)

# Plot
# plt.plot(y,'r*')
plt.scatter(y,E, c='g',marker='.', linewidths=2)
plt.plot(y,y, linestyle ='dotted')
plt.xlabel('E measured')
plt.ylabel('E true')
plt.legend(['calibrated peaks','real particle energies', 'calibration line'])
plt.show()


