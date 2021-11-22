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
MCP = []

peaks = [16.83597726854288, 53.18815906929055, 40.330077483222645, 42.00658965579711, 59.38278971039864, 139.2813243716698, 79.63379111094523]
sigma = [0.2567847993549146, 0.27959312391297864, 0.6725911179773683, 0.2934586147656381, 0.28921837117948007, 0.32791849633710735, 0.2785334352492435]

i = 0 # count events

for line in f:
    # if i == 10000:
    #     break
    
    line = line.strip()
    columns = line.split()
    
    
    if (i >= 3) and (i < 19589):
        # save each event into an array
        #   event nr             PH                  ToF            MCP PH
        #ev[columns[1]], float(columns[3]), float(columns[5]) , float(columns[7])
        
        n = float(columns[1])
        e = float(columns[3])
        t = float(columns[5])
        mcp = float(columns[7])
        T.append(t)
        E.append(e)
        MCP.append(mcp)

        if float(columns[7]) > 10:
            try:
                
                if i<1740:
                    p577.append([e, t , mcp])
                if i>1800 and i<4340:
                    p1815.append([e, t , mcp])
                if i>4660 and i<7730:
                    a1400.append([e, t , mcp])
                if i>7760 and i<10720:
                    a1464.append([e, t , mcp])
                if i>10760 and i<13630:
                    a2050.append([e, t , mcp])
                if i>13670 and i<17150:
                    a4750.append([e, t , mcp])
                if i>17170:
                    t2730.append([e, t , mcp])
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

# histogram
# plt.hist(E,bins = 2000, range=(0,200))
# plt.xlabel('energy channel')
# plt.ylabel('events')
# plt.show()

# plt.plot(T,MCP,'.')
# plt.xlabel('delta T [ns]')
# plt.ylabel('MCP pulse height')

# plt.show()


# Look at each event separately
ToF = []
PH = []
ev = [p577,p1815,a1400,a1464,a2050,a4750,t2730]
# m=0
# for list in ev:
#     for x in list:
#         ToF.append(x[1])
#         PH.append(x[0])
#         m+=1
#     # plt.figure(i)
#     # plt.hist(ToF,bins=100)
#     # plt.xlabel('tof')


# print(m)

# plt.show()
# plt.plot(E,T,'b.')
# plt.xlabel('PH')
# plt.ylabel('t')
# plt.plot(PH,ToF,'r.')
# plt.legend(['all data','peak data'])
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




