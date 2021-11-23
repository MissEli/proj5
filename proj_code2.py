# -*- coding: utf-8 -*-

# A program that will read data from file and plot the channels and events
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm as norm
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
# #%%
# #For reading the file for the first time
# file = np.loadtxt('protons_and_alphas_with_mcp.txt', dtype=str,delimiter='*',
#                   skiprows=3, usecols=(1,2,3,4))
# file = np.delete(file,len(file)-1,0)

# E = [] #Pulse heights
# T = [] #Time of flights
# mcp = [] #MCP pulse heights
# # events = int(input('How many events should be read?: '))
# for i, row in enumerate(file):
#     # if i == events:
#     #     break
#     E.append(float(row[1]))
#     T.append(float(row[2]))
#     mcp.append(float(row[3]))
fname = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']


#%%
#Loading data from already prepared file
Ec = np.loadtxt('ec.txt',dtype=float,skiprows=1)

#%%
#Function for fitting a normal distribution over span x. Parameters are ordered
#as [mean,standard deviation,area under curve, offset in y]
def ff(x, *params):
    pdf = norm.pdf(x,params[1],params[2])*params[0] + params[3]
    return pdf

#Function taking in data and finding peaks from defined parameters
#Height is minimum peak height, prominence it's required height over background 
#and distance the maximum allowed difference to neighbouring peaks
#The function returns a matrix with rows representing peak index, height, and left-/right bases
def peak_finder(ydata,height=50,prominence=30,distance=10):
    peaks = find_peaks(ydata,height=height,prominence=prominence,distance=distance)
    peak_index = peaks[0]
    peak_heights = peaks[1]['peak_heights']
    left_bases = peaks[1]['left_bases']
    right_bases = peaks[1]['right_bases']

    peak_data = np.vstack((peak_index,peak_heights,left_bases,right_bases))
    return peak_data
    

def pulse_plotter(xdata,ydata,fit):
    plt.subplot(222)
    plt.plot(xdata,ydata,'b')
    plt.plot(xdata,fit,'r')
    
    plt.xlabel('Channel number')
    plt.ylabel('Pulse height')
 #%%   
# def pulse_plotter2(xdata,ydata,xdata1,ydata1,fit1,xdata2,ydata2,fit2,title):
#     plt.figure(figsize=(12,10))
#     plt.plot(xdata1,ydata1,'b')
#     plt.plot(xdata2,ydata2,'g')
#     plt.plot(xdata,ydata,'r')
#     plt.plot(xdata1,fit1,'y')
#     plt.plot(xdata2,fit2,'c')
#     plt.title(title)
#     plt.xlabel('Channel number')
#     plt.ylabel('Pulse height')

# def data_fit(peaks,xdata,ydata):
#     #Separate peak data for easy handling
#     peak_index = peaks[0].astype(int)
#     peak_heights = peaks[1].astype(int)
#     left_bases = peaks[2].astype(int)
#     right_bases = peaks[3].astype(int)
#     #Create arrays to hold standard deviation and mean of fitted peaks
#     stds = []
#     means = []
    
#     savefig = input('Save figures? [y/n]: ') #User choses to save figures or not
#     skip = False #Boolean to handle double peaks

#     for i, peak in enumerate(peak_index):
#         if skip:
#             skip = False #Reset boolean
#             continue
#         #Check if last peak or if current and next peak are close enough to 
#         #be a double peak (handled differently)
#         if (i==len(peak_index)-1) or (peak - peak_index[i+1] < -100):
#             #Store y and x values
#             peak_data = []
#             x=[]
#             for j in range(left_bases[i],right_bases[i]+1): #Loop over peak
#                 peak_data.append(ydata[j])
#                 x.append(xdata[j])
#             #Initial guess for fitting using known parameters
#             p0 = [np.sum(peak_data),xdata[peak],np.std(peak_data),0]                   
#             try:
#                 fitted = curve_fit(ff,x,peak_data, p0)
#                 params, cov = fitted
#                 means.append(params[1]) #Extract channel using index from params
#                 stds.append(params[2])
#             #Sometimes fitting doesn't work but we don't want the program to stop
#             except RuntimeError:
#                 print(f'Could not find optimal parameters for peak {i+1}')
#                 means.append(xdata[peak]) #Just add peak location as mean
#                 params = p0 #Keep initial guess as fitting parameters for Gaussian
        
#         #Plotting
#             pulse_plotter(x,peak_data,ff(x,*params),f'Peak {i+1}')
#         #Save figures if savefig is given as "y"
#             if savefig=='y':
#                 plt.savefig(f'Peak_{i}.png',format='png')
            
#         else: #Double peak
#             print('Double peak detected')
#             #This time store x,y values separately and together for both peaks
#             peak_data1 = []
#             peak_data2 = []
#             peak_data=[]
#             x=[]
#             x1=[]
#             x2=[]
#             #To handle overlap between peaks estimate how far in they reach
#             cross_point1 = right_bases[i]-round((right_bases[i]-peak_index[i]) / 2)
#             cross_point2 = right_bases[i]+round((peak_index[i+1]-right_bases[i]) / 2)
            
#             #Loop for peak 1, keeping full strength until cross_point1
#             for j in range(left_bases[i],cross_point1):
#                 peak_data1.append(ydata[j])
#                 peak_data.append(ydata[j])
#                 x1.append(xdata[j])
#                 x.append(xdata[j])
#             #Add weight to y-values depending on depth into second peak region
#             for j, index in enumerate(range(cross_point1,cross_point2)):
#                 peak_data1.append(ydata[index]*(1- j/len(range(left_bases[i],cross_point1))))
#                 x1.append(xdata[index])
#                 peak_data.append(ydata[index])
#                 x.append(xdata[index])
#                 peak_data2.append(ydata[index]*(j/len(range(left_bases[i],cross_point1))))
#                 x2.append(xdata[index])
#             for j in range(cross_point2,right_bases[i+1]+1):
#                 peak_data2.append(ydata[j])
#                 peak_data.append(ydata[j])
#                 x2.append(xdata[j])
#                 x.append(xdata[j])
#             #Initial guesses based on separate peak data    
#             p01 = [np.sum(peak_data1),xdata[peak],np.std(peak_data1),0]
#             p02 = [np.sum(peak_data2),xdata[peak_index[i+1]],np.std(peak_data2),0]
#             try:
#                 fitted1 = curve_fit(ff,x1,peak_data1, p01)
#                 fitted2 = curve_fit(ff,x2,peak_data2, p02)
#                 params1 = fitted1[0]
#                 params2 = fitted2[0]
#                 means.append(params1[1]) #Extract channel using index from params
#                 means.append(params2[1])
#                 stds.append(params1[2])
#                 stds.append(params2[2])
#             except RuntimeError:
#                 print(f'Could not find optimal parameters for peak {i+1}')
#                 means.append(xdata[peak])
#                 means.append(xdata[peak_index[i+1]])
#                 params1 = p01
#                 params2 = p02
#                 # params = p0
    
            
#             pulse_plotter2(x,peak_data,x1,peak_data1,ff(x1,*params1),
#                            x2,peak_data2,ff(x2,*params2),f'Peaks {i+1} and {i+2}')
            
#             if savefig=='y':
#                 plt.savefig(f'Peak_{i}.png',format='png')
        
#             skip = True
#     results = np.vstack((means,stds))
#     return results
    
#%%
def data_fit(xdata,ydata,plot):
    #Create arrays to hold standard deviation and mean of fitted peaks
    peak = peak_finder(ydata)
    std = np.std(ydata)
    mean = xdata[peak[0][0].astype(int)]
    area = np.sum(ydata)
    x1 = peak[2][0].astype(int)
    x2 = peak[3][0].astype(int)
    x = []
    y=[]
    for i in range(x1,x2+1):
        x.append(xdata[i])
        y.append(ydata[i])
    #Initial guess for fitting using known parameters
    p0 = [area,mean,std,0]                   
    try:
        fitted = curve_fit(ff,x,y, p0)
        params, cov = fitted
        mean = params[1] #Extract channel using index from params
        std = params[2]
    #Sometimes fitting doesn't work but we don't want the program to stop
    except RuntimeError:
        print('Could not find optimal parameters for peak')
        params = p0 #Keep initial guess as fitting parameters for Gaussian
    
    #Plotting
    if plot:    
        pulse_plotter(x,y,ff(x,*params))
    #Save figures if savefig is given as "y"
        
            
    
    result = [mean,std]
    return result
#%%
#Peak plotter
def histogrammer(xdata,ydata,peaks):
    y_p = []
    x_p = []
    for p in peaks:
        y_p.append(ydata[p]+5)
        x_p.append(xdata[p])
        x = np.delete(xdata,0)    
    plt.plot(x,ydata)
    plt.plot(x_p,y_p,color='r',marker='v',linestyle='None')

    plt.xlabel('Energy channel')
    plt.ylabel('Events')
    plt.show()
#%%
def calibration(xdata,calipeaks,calienergies):
    dc = calipeaks[1]-calipeaks[0]
    de = calienergies[1]-calienergies[0]
    k = de/dc
    m = calienergies[0]-k*calipeaks[0]
    return [k,m]

def E_calib(k,m,xdata,ydata,peak,std):
    energies = [x*k+m for x in xdata]
    E_peak = peak*k+m
    E_std = std*k+m
    plt.subplot(212)
    plt.plot(energies,ydata,'-')
    plt.xlabel('Time [ns]')
    plt.ylabel('Pulse height')
    output = [E_peak,E_std]
    return output

def PHD(E,peaks,Eloss):
    E_exp=[] #Energy with losses accounted
    for i in range(len(peaks)):
        E_exp.append(peaks[i]-Eloss[i])

    phd = [element-E_exp[i] for i, element in enumerate(E)] 
    return phd

def load(fname):
    file = np.loadtxt(fname, dtype=float,delimiter=',', skiprows=1)
    E = file[:,0]
    return E

def main():
    #Eloss = [-6.1909968, -3.1023139999999997, -32.09591, -31.536043, -27.466562999999996,
    #     -16.584065999999998, -4.685778]
    Eloss = [-0.0061909968, -0.0031023139999999997, -0.03209591, -0.031536043, -0.027466562999999996, -0.016584065999999998, -0.004685778]
    Eloss = [e*1000 for e in Eloss]
    calienergies = [577+Eloss[0],1815+Eloss[1]]
    Etrue = [0.577,1.815,1.4, 1.464, 2.05, 4.75,2.73]
    Etrue = [element*1000 for element in Etrue]
    energies = []
    peaks = []
    calib_peaks = []
    calib_means = []
    stds = []
    titles = ['p577','p1815','a1400','a1464','a2050','a4750','t2730']
    for name in fname:
        energy = load(name)
        energies.append(energy)
    # rows, columns = times.shape
    results=[]
    phd = []
    savefig = input('Save figures? [y/n]: ') #User choses to save figures or not
    for i in range(len(energies)):
        E_hist = np.histogram(energies[i],bins=5000,range=(0,200))
        xdata = np.delete(E_hist[1],0)
        ydata = E_hist[0]
        result = data_fit(xdata,ydata,0)
        stds.append(result[1])
        peaks.append(result[0])
        if (i==0) or (i==1):
            calib_peaks.append(result[0])
    
    k, m = calibration(xdata,calib_peaks,calienergies)
    
    for i in range(len(energies)):
        plt.figure(figsize=(15,15))
        plt.subplot(221)
        E_hist = plt.hist(energies[i],bins=5000,range=(0,200))
        plt.suptitle(f'Peak for {titles[i]}')
        plt.ylabel('Pulse height')
        plt.xlabel('Channel number')
        xdata = np.delete(E_hist[1],0)
        ydata = E_hist[0]
        result = data_fit(xdata,ydata,1)
        calib_result = E_calib(k,m,xdata,ydata,peaks[i],stds[i])
        results.append(calib_result)
        calib_means.append(calib_result[0])
        if savefig=='y':
            plt.savefig(f'timePeak_{titles[i]}.png',format='png')
    
    phd = PHD(Etrue,calib_means,Eloss)
    
    return [results,phd]
    # E_hist = plt.hist(E,bins = 5000, range=(0,200))
    # xdata = E_hist[1]
    # # xdata = np.delete(xdata,1,0)
    # ydata = E_hist[0]
    # plt.xlabel('energy channel')
    # plt.ylabel('events')
    # plt.show()
    # peaks = peak_finder(ydata)
    # peak_means = peaks[0].astype(int) #Un-fitted and un-calibrated
    # histogrammer(xdata,ydata,peak_means)
    # stats = data_fit(peaks,xdata,ydata)
    # means = stats[0]
    # stds = stats[1]
    # fitting = calibration(xdata,[means[0],means[3]],calienergies)
    # E_peaks = E_calib(fitting[0],fitting[1],np.delete(xdata,0),ydata,means,stds)
    # phd = PHD(Etrue,E_peaks[0],Eloss)
    # result = np.vstack((E_peaks,phd))
    # return result

# mass = [1,4,4,1,4,3,4]
# plt.plot(mass,phd,'.')
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot(mass,phd,E,'.b')
# plt.rc('font', size=8)
# ax.set_xlabel('A')
# ax.set_ylabel('PHD [keV]')
# ax.set_zlabel('Energy [MeV]')
# plt.plot(x*k+m,E_hist[0],'-')
# plt.xlabel('Energy [keV]')
# plt.ylabel('Events')
