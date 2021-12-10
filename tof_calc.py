# A program that calculates velocity and ToF from mass and energy
import math
import numpy as np
import matplotlib.pyplot as plt
# E = mv^2/2
# Run	p 0.577 MeV	p 1.815 MeV	α 1.4 MeV	α 1.464 MeV	α 2.05 MeV	α 4.75 MeV	3H 2.73 MeV

V = []
T = []
Tns = []
E = [0.577, 1.815, 1.4, 1.464, 2.05, 4.75, 2.73] #MeV
Eloss = [-0.0061909968, -0.0031023139999999997, -0.03209591, -0.031536043, -0.027466562999999996, -0.016584065999999998, -0.004685778]
Eloss_std = [0.001133035593161027, 0.0008626657309781118, 0.0023466318249567826, 0.0024820656609266, 0.002387973360829429, 0.0023835786195642887, 0.001032048523430948]
m = [938.27, 938.27, 3728.4, 3728.4, 3728.4, 3728.4, 2809.41] # MeV
c = 3*10**8

s = 39.15 #mm
E_det = [(E+Eloss[i])*1000 for i, E in enumerate(E)]

s = 38.35 #mm


for i in range(len(E)):
    v = math.sqrt(2*(E[i]+Eloss[i])/m[i])*c
    V.append(round(v,2))

    t = (s*10**-3)/v
    T.append(t)
    Tns.append(t*10**9)

# print(V)
print(T)


E1 = [570.8090032, 8.977433482068347]
E2 = [1811.8976859999998, 9.836802929713219]
E3 = [1370.5776578369482, 20.10503836315407]
E4 = [1428.936015304666, 11.461834631719038]
E5 = [2023.2640121302848, 9.926343101597883]
E6 = [4751.052015174993, 11.203482616650504]
E7 = [2714.916596558814, 9.470540698659216]

E_measured = np.vstack((E1,E2,E3,E4,E5,E6,E7))
means = E_measured[:,0]
stdss = E_measured[:,1]
Velocities = []
Times = []

Means = []
for i in range(len(means)):
    spread = [means[i]-stdss[i],means[i],means[i]+stdss[i]]
    Means.append(spread)


for i in range(len(Means)):
    Vm = []
    Tm = []
    for j in range(len(Means[i])):
        vm = math.sqrt(2*10**(-3)*Means[i][j]/m[i])*c
        Vm.append(round(vm,2))

        tm = (s*10**-3)/vm
        Tm.append(tm)

    Velocities.append(Vm)
    Times.append(Tm)
labels = ['Proton577','Proton1815','\u03b11400','\u03b11464','\u03b12050','\u03b14750','Triton2730']   
plt.figure()
for i in range(len(Means)):
    plt.plot(Means[i],Times[i],'o',label=labels[i],markersize=2)
plt.plot(E_det,T,'*',color='k',markersize=2)
plt.xlabel('Mean energy')
plt.ylabel('Tof')
plt.legend()  
plt.savefig('e_to_t.svg',format='svg')

print(Tns)

