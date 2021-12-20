import math

M = [1,1,4,4,4,4,3] # mass amu
Z = [1,1,4,4,4,4,1] # atomic number (charge)

rho = 9500 # (range 4000-15000) Ohm cm
V = 140 # bias voltage V
C = 190 # pF
d = 0.03 # cm
mu = 1481
tau = rho*10**(-12)
dd = d # math.sqrt(2*tau*mu*V) # depletion depth


Etrue = [0.577,1.815,1.4, 1.464, 2.05, 4.75,2.73]
Eloss = [-0.0061909968, -0.0031023139999999997, -0.03209591, -0.031536043, -0.027466562999999996, -0.016584065999999998, -0.004685778]
E = [E+Eloss[i] for i, E in enumerate(Etrue)]

S = [2.411E2, 1.195E2, 1.185E3, 1.166E3, 1.019E3, 6.404E2, 8.968E1] # stopping power MeV cm2/mg
Sp = [2.411E2,1.195E2, 3.069E2, 3.005E2, 2.547E2, 1.572E2, 1.859E2] # stopping power protons
Z2 = [s/Sp[i] for i,s in enumerate(S)] # effective charge

R = [0.000752516462312825, 0.004163276429979496, 0.00049132672233583, 0.0005152769574172682, 
        0.0007521621771829435, 0.002278116220403816, 0.00433567685956621] # ranges cm
# rhoSi = 2.33 # g/cm3
# x = [r/(3*rhoSi) for r in R] # cm
F = [(dd-xx)/(mu*tau) for xx in R]


td = [Z2[i]**2/(2*M[i])*math.exp(-E[i]/(3.75*M[i]))*C*(rho/d)**(2/5)*1/F[i] for i in range(len(R))]

PHD = [2.33e-4*(M[i]*Z[i])**(6/5)*(E[i]/M[i])**(1/2)*(1+(1.32e3*Z[i]*(S[i]/E[i]**2)**(1/3))/(rho**(1/4)*F[i])) for i in range(len(R))]

print(td)
print(PHD)