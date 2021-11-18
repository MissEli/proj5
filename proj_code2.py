# -*- coding: utf-8 -*-

# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm as norm
from scipy.optimize import curve_fit
#%%
#For reading the file for the first time
file = np.loadtxt('protons_and_alphas_with_mcp.txt', dtype=str,delimiter='*',
                  skiprows=3, usecols=(1,2,3,4))
file = np.delete(file,len(file)-1,0)

E = []
T = []
MCP = []
# events = int(input('How many events should be read?: '))
for i, row in enumerate(file):
    # if i == events:
    #     break
    E.append(float(row[1]))
    T.append(float(row[2]))
    MCP.append(float(row[3]))



#%%
#Loading data from already prepared file
Ec = np.loadtxt('ec.txt',dtype=float,skiprows=1)

#Find the peaks
#%%
#Making histograms
E_hist = plt.hist(E,bins = 10000, range=(0,200))
plt.xlabel('energy channel')
plt.ylabel('events')
plt.show()
#%%
def ff(x, *params):
    pdf = norm.pdf(x,params[1],params[2])*params[0] + params[3]
    return pdf
# def ff2(x, *params):
#     pdf1 = norm.pdf(x,params[1],params[2])*params[0] + params[3]
#     pdf2 = norm.pdf(x,params[4],params[5])*params[6] + params[7]
#     return pdf1+pdf2


peaks = find_peaks(E_hist[0],height=50, prominence=30,distance=10)
peak_index = peaks[0]
peak_heights = peaks[1]['peak_heights']
left_bases = peaks[1]['left_bases']
right_bases = peaks[1]['right_bases']
stds = []
means = []
savefig = input('Save figures? [y/n]: ')
skip = False

for i, peak in enumerate(peak_index):
    if skip:
        skip = False
        continue
    
    if (i==len(peak_index)-1) or (peak - peak_index[i+1] < -100):
        peak_data = []
        x=[]
        for j in range(left_bases[i],right_bases[i]+1):
            peak_data.append(E_hist[0][j])
            x.append(E_hist[1][j])
            
        p0 = [np.sum(peak_data),E_hist[1][peak],np.std(peak_data),0]                   
        # pdf = norm.pdf(x, peak_channels[1],stds[1])
        try:
            fitted = curve_fit(ff,x,peak_data, p0)
            params, cov = fitted
            means.append(params[1]) #Extract channel using index from params
            stds.append(params[2])
        except RuntimeError:
            print(f'Could not find optimal parameters for peak {i+1}')
            means.append(E_hist[1][peak])
            params = p0
        
        plt.figure(figsize=(12,10))
        plt.plot(x,peak_data,'b')
        plt.plot(x,ff(x,*params),'r')
        plt.title(f'Peak {i+1}')
        plt.xlabel('Channel number')
        plt.ylabel('Pulse height')
        if savefig=='y':
            plt.savefig(f'Peak_{i}.png',format='png')
    else:
        print('Double peak detected')
        peak_data1 = []
        peak_data2 = []
        x1=[]
        x2=[]
        cross_point1 = right_bases[i]-round((right_bases[i]-peak_index[i]) / 2)
        cross_point2 = right_bases[i]+round((peak_index[i+1]-right_bases[i]) / 2)
        for j in range(left_bases[i],cross_point1):
            peak_data1.append(E_hist[0][j])
            x1.append(E_hist[1][j])
        for j, index in enumerate(range(cross_point1,cross_point2)):
            peak_data1.append(E_hist[0][index]*(1- j/len(range(left_bases[i],cross_point1))))
            x1.append(E_hist[1][index])
       
        for j, index in enumerate(range(cross_point1,cross_point2)):
            peak_data2.append(E_hist[0][index]*(j/len(range(left_bases[i],cross_point1))))
            x2.append(E_hist[1][index])
        for j in range(cross_point2,right_bases[i+1]+1):
            peak_data2.append(E_hist[0][j])
            x2.append(E_hist[1][j])
        # for j in range(left_bases[i],right_bases[i]+1):
        #     peak_data1.append(E_hist[0][j])
        #     x1.append(E_hist[1][j])
        # for j in range(right_bases[i]+1,right_bases[i+1]+1):
        #     peak_data2.append(E_hist[0][j])
        #     x2.append(E_hist[1][j])
        
        # x = x1+x2
        # peak_data = peak_data1+peak_data2
        
        # for k in range(left_bases[i],right_bases[i+1]+1,1):
        #     x.append(E_hist[1][k])
        p01 = [np.sum(peak_data1),E_hist[1][peak],np.std(peak_data1),0]
        p02 = [np.sum(peak_data2),E_hist[1][peak_index[i+1]],np.std(peak_data2),0]
        # p0 = p01+p02
        try:
            fitted1 = curve_fit(ff,x1,peak_data1, p01)
            fitted2 = curve_fit(ff,x2,peak_data2, p02)
            params1 = fitted1[0]
            params2 = fitted2[0]
            # params = fitted[0]
            means.append(params1[1]) #Extract channel using index from params
            means.append(params2[1])
            stds.append(params1[2])
            stds.append(params2[2])
        except RuntimeError:
            print(f'Could not find optimal parameters for peak {i+1}')
            means.append(E_hist[1][peak])
            means.append(E_hist[1][peak_index[i+1]])
            params1 = p01
            params2 = p02
            # params = p0

        
        plt.figure(figsize=(12,10))
        plt.plot(x1,peak_data1,'b')
        plt.plot(x2,peak_data2,'g')
        plt.plot(x1,ff(x1,*params1),'y')
        plt.plot(x2,ff(x2,*params2),'c')
        # plt.plot(x,ff2(x,*params),'r')
        plt.title(f'Peak {i+1} and {i+2}')
        plt.xlabel('Channel number')
        plt.ylabel('Pulse height')
        if savefig=='y':
            plt.savefig(f'Peak_{i}.png',format='png')
        
        skip = True
        
        
    
    #Create x value in channels instead of index
    


# plt.hist(E,bins=10000,range=(0,150))
# plt.show()
# Plot
#%%
# x1 = np.arange(0, len(E))
# plt.plot(x1,E,'.')
# plt.xlabel('events')
# plt.ylabel('energy channel')
# plt.show()

y_p = []
x_p = []
for p in peaks[0]:
    y_p.append(E_hist[0][p]+5)
    x_p.append(E_hist[1][p])
x = np.delete(E_hist[1],0)    
plt.plot(x,E_hist[0])
plt.plot(x_p,y_p,color='r',marker='v',linestyle='None')

plt.xlabel('energy channel')
plt.ylabel('events')
plt.show()
#%%
# Look at each event 
# print('(PH, ToF)')
# for n in ev:
#     print(ev[n])
channels = [means[0],means[3]]
energies = [577,1815]
dc = channels[1]-channels[0]
de = energies[1]-energies[0]
k = de/dc
m = energies[0]-k*channels[0]

plt.plot(x*k+m,E_hist[0],'-')
plt.xlabel('Energy [keV]')
plt.ylabel('Events')
