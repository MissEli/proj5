# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt

f = open('protons_and_alphas_new.txt', 'r')

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
TSi = []
TMCP = []
MCP = []

i = 0  # count events

for line in f:
    # if i == 10000:
    #     break

    line = line.strip()
    columns = line.split()

    if (i >= 3) and (i < 18239):  # old i=19589
        # save each event into an array
        #   event nr             PH                      ToF                            MCP PH
        #ev[columns[1]], float(columns[3]), float(columns[5]) - float(columns[7]), float(columns[9])

        n = float(columns[1])
        e = float(columns[3])
        tSi = float(columns[5])
        tMCP = float(columns[7])
        mcp = float(columns[9])
        TSi.append(tSi)
        TMCP.append(tMCP)
        E.append(e)
        MCP.append(mcp)

        # if t < 0:
        #     print('t<0')
        #     print(t)
        #     print(mcp)

        # if float(columns[7]) > 10:
        # try:

        #     if i < 1535:
        #         p577.append([e, t, mcp])
        #     if i > 1560 and i < 3715:
        #         p1815.append([e, t, mcp])
        #     if i > 3725 and i < 6955:
        #         a1400.append([e, t, mcp])
        #     if i > 6970 and i < 9825:
        #         a1464.append([e, t, mcp])
        #     if i > 9835 and i < 12610:
        #         a2050.append([e, t, mcp])
        #     if i > 12620 and i < 15980:
        #         a4750.append([e, t, mcp])
        #     if i > 15995:
        #         t2730.append([e, t, mcp])

        # except IndexError as ie:  # Only needed to find last index
        #     print("*** Index Error: ")
        #     print(i)
    else:
        print(repr(line))

    i += 1

f.close()
print(i)
print('done')

# *** Plot *** #
# plt.figure()
# x1 = np.arange(0, len(E))
# plt.plot(x1, E, '.')
# plt.xlabel('events')
# plt.ylabel('energy channel')


# # histogram
# plt.figure()
# plt.hist(E, bins=2000, range=(0, 200))
# plt.xlabel('energy channel')
# plt.ylabel('events')


plt.figure()
plt.hist2d(MCP, TMCP, bins=1000, range=([0,400],[400,600]), cmap='gist_heat_r')
plt.ylabel('T [ns]')
plt.xlabel('MCP pulse height')
plt.colorbar()

plt.figure()
plt.hist2d(E, TSi, bins=1000, range=([0,200],[550,650]), cmap='gist_heat_r')
plt.ylabel('T [ns]')
plt.xlabel('Detector pulse height')
plt.colorbar()




# Look at each event separately
# ToF = []
# PH = []
# ev = [p577, p1815, a1400, a1464, a2050, a4750, t2730]
# m=0
# j = 0
# for list in ev:
#     for x in list:
#         ToF.append(x[1])
#         PH.append(x[0])
#         m+=1
    # plt.figure(j)
    # plt.hist(ToF,bins=100)
    # plt.xlabel('tof')
    # j+=1
# print(m)

# plt.figure()
# plt.plot(E, T, 'b.')
# plt.xlabel('PH')
# plt.ylabel('t')
# plt.plot(PH,ToF,'r.')
# plt.legend(['all data','peak data'])


plt.show()

# Save to text-file
# fname = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt','allp.txt']

# for i,list in enumerate(ev):
#     print(i)
#     with open(fname[i], 'w+') as file:
#         file.write('#'+fname[i]+', PH, ToF, MCP PH')
#         for x in list:
#             file.write('\n')
#             file.write(str(x[0]))
#             file.write(', '+ str(x[1]))
#             file.write(', '+ str(x[2]))
