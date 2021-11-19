# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt

f = open('protons_and_alphas_with_mcp.txt', 'r')

# create vectors
ev = []
p577 = []
p1815 = []
a1400 = []
a1464 = []
a2050 = []
a4750 = []
t2730 = []

E = []
T = []

peaks = [16.83597726854288, 53.18815906929055, 40.330077483222645, 42.00658965579711, 59.38278971039864, 139.2813243716698, 79.63379111094523]
sigma = [0.2567847993549146, 0.27959312391297864, 0.6725911179773683, 0.2934586147656381, 0.28921837117948007, 0.32791849633710735, 0.2785334352492435]

i = 0 # count events

for line in f:
    # if i == 10000:
    #     break
    
    line = line.strip()
    # print(repr(line))
    columns = line.split()
    
    
    if (i >= 3) and (i < 19589):
        # save each event into a dictionary
        #   event nr             PH                  ToF            MCP PH
        #ev[columns[1]] = float(columns[3]), float(columns[5]) , float(columns[7])
        
        if float(columns[5]) > 0:
            try:
                e = float(columns[3])
                t = float(columns[5])
                T.append(t)
                E.append(e)

                if e > peaks[0]-3*sigma[0] and e < peaks[0]+3*sigma[0]:
                    p577.append([e, t , float(columns[7])])
                if e > peaks[1]-3*sigma[1] and e < peaks[1]+3*sigma[1]:
                    p1815.append([e, t , float(columns[7])])
                if e > peaks[2]-3*sigma[2] and e < 41.32:
                    a1400.append([e, t , float(columns[7])])
                if e> 41.32 and e < peaks[3]+3*sigma[3]:
                    a1464.append([e, t , float(columns[7])])
                if e > peaks[4]-3*sigma[4] and e < peaks[4]+3*sigma[4]:
                    a2050.append([e, t , float(columns[7])])
                if e > peaks[5]-3*sigma[5] and e < peaks[5]+3*sigma[5]:
                    a4750.append([e, t , float(columns[7])])
                if e > peaks[6]-3*sigma[6] and e < peaks[6]+3*sigma[6]:
                    t2730.append([e, t , float(columns[7])])
            except IndexError as ie:    #Only needed to find last index
                print("*** Index Error: ")
                print(i)          
    else:
        print(repr(line)) 
    
    i += 1
f.close()
print(i)


print('done')

# *** Plot *** #
# x1 = np.arange(0, len(E))
# plt.plot(x1,E,'.')
# plt.xlabel('events')
# plt.ylabel('energy channel')
# plt.show()

# np.histogram(T, bins=10, range=None, normed=None, weights=None, density=None)
# plt.hist(E,bins = 2000, range=(0,200))
# plt.xlabel('energy channel')
# plt.ylabel('events')
# plt.show()

# plt.plot(E,T,'.')
# plt.xlabel('pulse height')
# plt.ylabel('delta T [ns]')
# plt.show()


# Look at each event separately
# print('(PH, ToF)')
ToF = []
PH = []
ev = [p577,p1815,a1400,a1464,a2050,a4750,t2730]
i=1
for list in ev:
    for x in list:
        ToF.append(x[1])
        PH.append(x[0])
    plt.figure(i)
    plt.hist(ToF,bins=100, range=(60,100))
    plt.xlabel('tof')
    
    i+=1

plt.show()
# plt.plot(PH,ToF,'r.')
# plt.show()

# Save to text-file
fname = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt','allp.txt']

for i,list in enumerate(ev):
    print(i)
    with open(fname[i], 'w+') as file:
        file.write('#'+fname[i]+', PH, ToF, MCP PH')
        for x in list:
            file.write('\n')
            file.write(str(x[0]))
            file.write(', '+ str(x[1]))
            file.write(', '+ str(x[2]))




