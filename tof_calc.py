# A program that calculates velocity and ToF from mass and energy
import math
import numpy as np
import matplotlib.pyplot as plt
# E = mv^2/2
# Run	p 0.577 MeV	p 1.815 MeV	α 1.4 MeV	α 1.464 MeV	α 2.05 MeV	α 4.75 MeV	3H 2.73 MeV

#%% Calculate ToF (theoretical)
V = []
T = []
Tns = []
E = [0.577, 1.815, 1.4, 1.464, 2.05, 4.75, 2.73] #MeV
Eloss = [-0.0061909968, -0.0031023139999999997, -0.03209591, -0.031536043, -0.027466562999999996, -0.016584065999999998, -0.004685778]
Eloss_std = [0.001133035593161027, 0.0008626657309781118, 0.0023466318249567826, 0.0024820656609266, 0.002387973360829429, 0.0023835786195642887, 0.001032048523430948]
m = [938.27, 938.27, 3728.4, 3728.4, 3728.4, 3728.4, 2809.41] # MeV
c = 3*10**8

l = 38.35 #mm
E_det = [(E+Eloss[i])*1000 for i, E in enumerate(E)]

for i in range(len(E)):
    v = math.sqrt(2*(E[i]+Eloss[i])/m[i])*c
    V.append(round(v,2))

    t = (l*10**-3)/v
    T.append(t)
    Tns.append(t*10**9)

# print(V)
print(T)
#%% Calculate error in ToF for different errors in E, l
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
                     
dEEs = np.linspace(0,0.1,1000) #x vector

rel_errors = np.zeros((len(dls),len(dEEs)))

for i, dl in enumerate(dls):
    for j, e in enumerate(dEEs):
        rel_errors[i,j] = ((dl/l)**2+(e/2)**2)**(1/2)

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
#ToF=[3.741226671982719,2.0998708941461015, 4.817580046997456, 4.707766474890197, 3.961949251685958, 2.589819098875753, 2.962748483671883]
err_abs = [rel_err[i]*t for t in Tns]

plt.savefig('Rel_err_3.svg', bbox_inches='tight',format='svg')

#%% Translate energy data to time

# E1 = [570.8090032, 8.977433482068347]
# E2 = [1811.8976859999998, 9.836802929713219]
# E3 = [1370.5776578369482, 20.10503836315407]
# E4 = [1428.936015304666, 11.461834631719038]
# E5 = [2023.2640121302848, 9.926343101597883]
# E6 = [4751.052015174993, 11.203482616650504]
# E7 = [2714.916596558814, 9.470540698659216]
Energies = [575.5388107650124,
1811.9514279111986,
1372.2941668449973,
1430.4326467852263,
2022.5213864396444,
4740.03186784654,
2711.5680226073932,
    ]
stds = [5.695130399391655,
6.907118295139223,
6.373703237930785,
6.427570384406133,
7.2038057605473345,
12.132465023784567,
8.295230333120175]

E_measured = np.vstack((Energies,stds)).T
means = E_measured[:,0]
stdss = E_measured[:,1]
Velocities = []
Times = []
std_T = []
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

        tm = (l*10**-3)/vm
        Tm.append(tm)

    Velocities.append(Vm)
    Times.append(Tm)
    std_T.append(Times[i][0]-Times[i][1])
labels = ['Proton577','Proton1815','\u03b11400','\u03b11464','\u03b12050','\u03b14750','Triton2730']   
plt.figure()
for i in range(len(Means)):
    plt.plot(Means[i],Times[i],'o',label=labels[i],markersize=2)
plt.plot(E_det,T,'*',color='k',markersize=2)
plt.xlabel('Mean energy')
plt.ylabel('Tof')
plt.legend()  
plt.savefig('e_to_t.svg',format='svg')

#%% Errorbar plots for tof over energy for theoretical and measured values
#Convert everything to ns
Times_m = [Times[i][1]*10**9 for i in range(len(Times))]
std_T = [s*10**9 for s in std_T]
calib_time = [9.304,2.057,6.112,5.809,3.918,2.227,2.724]
colors = ['r','b','y','c','m','g','k']
colors2 = ['tab:purple','tab:blue','tab:orange','tab:pink','tab:brown','tab:olive','tab:gray']
plt.figure()
plt.title('Plot of calibrated TOFs (lines), TOFs from energy data (*) and theoretical TOFs',fontsize=10)
for i in range(len(Times)):
    # if i==0:
    #     plt.errorbar(Means[i][1],Times_m[i],std_T[i],marker='.',markersize=3,capsize=5,color='b',label='Experimental data')
    #     plt.errorbar(E_det[i],Tns[i],err_abs[i],marker='.',markersize=3,capsize=3,color='r',elinewidth=0.5,label='Theoretical prediction')
    # else:
    plt.errorbar(Means[i][1],Times_m[i],std_T[i],marker='.',markersize=3,capsize=5,
                 color=colors[i],label=f'{labels[i]}*')
    plt.errorbar(E_det[i],Tns[i],err_abs[i],marker='.',markersize=3,capsize=3,
                 color=colors2[i],label=labels[i])
    plt.axhline(calib_time[i],color=colors2[i],linewidth=1)
plt.xlabel('Energy [keV]')
plt.ylabel('Time of flight [ns]')
#plt.title('Comparing TOF data to theoretical predictions within errors')
plt.legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
plt.savefig('toferrorbars.svg',format='svg')
print(Tns)

