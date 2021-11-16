# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt

f = open('protons_and_alphas_and_mcp.txt', 'r')

# create vectors
ev = {}
E = []
T = []


i = 0 # count events

for line in f:
    # if i == 100:
    #     break
    
    line = line.strip()
    # print(repr(line))
    columns = line.split()
    
    
    if (i >= 3) and (i < 19589):
        # save each event into a dictionary
        #   event nr             PH                  ToF            MCP PH
        #ev[columns[1]] = float(columns[3]), float(columns[5]) , float(columns[7])
        
        if float(columns[7]) > 10:
            try:
                e = float(columns[3])
                t = float(columns[5])
                T.append(t)
                E.append(e)
                ev[columns[1]] = float(columns[3]), float(columns[5]) , float(columns[7])
            except IndexError as ie:    #Only needed to find last index
                print("*** Index Error: ")
                print(i)          
    else:
        print(repr(line)) 
    
    i += 1
f.close()
print(i)


# with open('ec.txt', 'w') as f2:
#     f2.write('# Events per channel')
#     for x in Ec:
#         f2.write('\n')
#         f2.write(str(x)) 
# f2.close()

print('done')

# *** Plot *** #
# x1 = np.arange(0, len(E))
# plt.plot(x1,E,'.')
# plt.xlabel('events')
# plt.ylabel('energy channel')
# plt.show()


# np.histogram(T, bins=10, range=None, normed=None, weights=None, density=None)
plt.hist(E,bins = 2000, range=(0,200))
plt.xlabel('energy channel')
plt.ylabel('events')
plt.show()

# plt.plot(E,T,'.')
# plt.xlabel('pulse height')
# plt.ylabel('delta T [ns]')
# plt.show()

# Look at each event 
# print('(PH, ToF)')
# for n in ev:
#     print(ev[n])