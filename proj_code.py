# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt

f = open('protons_and_alphas.txt', 'r')

# create vectors
ev = {}
E = []
T = []
Ec = [0]*200*100 # channels


i = 0 # count events
n = 0 # count added events to channel
for line in f:
    # if i == 100:
    #     break
    
    line = line.strip()
    # print(repr(line))
    columns = line.split()
    
    
    if (i >= 3) and (i < 19589):
        # save each event into a dictionary
        #   event nr             PH                  ToF
        # ev[columns[1]] = float(columns[3]), float(columns[5]) 
        try:
            e = float(columns[3])
            E.append(e)
            for j in range(len(Ec)):
                if j/100 == round(e,2):
                    Ec[j] += 1  #add to channel
                    n += 1

        except IndexError as ie:    #Only needed to find last index
            print("*** Index Error: ")
            print(i)          
    else:
        print(repr(line)) 
    
    i += 1
f.close()
print(i)
print(n)


with open('ec.txt', 'w') as f2:
# Plot
x1 = np.arange(0, len(E))
plt.plot(x1,E,'.')
plt.xlabel('events')
plt.ylabel('energy channel')
plt.show()

x2 = np.arange(0, 200, 0.01)
plt.plot(x2,Ec,'-')
plt.xlabel('energy channel')
plt.ylabel('events')
plt.show()

# Look at each event 
# print('(PH, ToF)')
# for n in ev:
#     print(ev[n])