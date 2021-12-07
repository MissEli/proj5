# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:38:10 2021

Code for analysing energy data
"""
#%% Importing packages
#Import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltp
from scipy.signal import find_peaks
import scipy.constants as sc
from scipy.stats import norm as norm
from scipy.optimize import curve_fit

#%% Functions
#load will read a given ascii file and return the energy in a vector
def load(fname):
    file = np.loadtxt(fname, dtype=float,delimiter=',', skiprows=1)
    E = file[:,0]
    return E
#Function which loads data from a set of files using load function.
def get_data(fnames):
    energies = []
    for name in fnames:
        energy = load(name)
        energies.append(energy)
    ydata = []
    for i in range(len(energies)):
        E_hist = plt.hist(energies[i],bins=5000,range=(0,200))  
        if i == 0:
            xdata = [e+0.02 for e in E_hist[1]]
            xdata = np.delete(xdata,len(xdata)-1)
        ydata.append(E_hist[0])
    return np.vstack((xdata,ydata))


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
    
#Plots a pulse with a data fit distribution. Subplotted with original histogram peak
def pulse_plotter(xdata,ydata,fit):
    plt.subplot(222)
    plt.plot(xdata,ydata,'b')
    plt.plot(xdata,fit,'r')
    
    plt.xlabel('Channel number')
    plt.ylabel('Pulse height')
    
#Function to fit a normal distribution onto a histogram peak
def data_fit(xdata,ydata,plot):
    #Create arrays to hold standard deviation and mean of fitted peaks
    peak = peak_finder(ydata) #Find the peak
    std = np.std(ydata)
    mean = xdata[peak[0][0].astype(int)]
    area = np.sum(ydata)
    #Make new x-vector only over peak area
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
        std = params[2] #Standard deviation of the fitted distribution
        mean_err = np.sqrt(cov[1,1]) #Uncertainty of the mean channel
    #Sometimes fitting doesn't work but we don't want the program to stop
    except RuntimeError:
        print('Could not find optimal parameters for peak')
        params = p0 #Keep initial guess as fitting parameters for Gaussian
    
    #Plotting
    if plot:    
        pulse_plotter(x,y,ff(x,*params))
    #Save figures if savefig is given as "y"
        
    result = [mean,std,mean_err]
    return result

#Calibration functions. 
#calibration uses given peaks and energies and uses a linear fit to two points
#to get a gain and intercept.
def calibration(calipeaks,calienergies):
    dc = calipeaks[1]-calipeaks[0]
    de = calienergies[1]-calienergies[0]
    k = de/dc
    m = calienergies[0]-k*calipeaks[0]
    return [k,m]

#E_calib takes in gain and intercept for calibration and converts the given
#x-vector values. It will also convert values of a peaks channel and it's standard
#deviation into units of energy instead of channel.
def E_calib(k,m,xdata,ydata,peak,std):
    energies = [x*k+m for x in xdata]
    E_peak = peak*k+m
    E_std = std*k
    # plt.subplot(212)
    # plt.plot(energies,ydata,'-')
    # plt.xlabel('Time [ns]')
    # plt.ylabel('Pulse height')
    output = [E_peak,E_std,energies]
    return output

#PHD calculates the pulse height defect as the difference between a peaks energy
#and the expected energy in the detector.
def PHD(E_det,peaks):
    phd = [element-peaks[i] for i, element in enumerate(E_det)] 
    return phd

#%% Declare true energies and losses
"""
Data to be loaded before running program
 
Particles are ordered by type first and second by energy in rising order:
    p0557, p1815, 
    a1400, a1464, a2050, a4750,
    t2730
"""
fnames = ['p577.txt','p1815.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']
#True energies
Etrue = [0.577,1.815,1.4, 1.464, 2.05, 4.75,2.73]
Etrue = [element*1000 for element in Etrue]
#Energy losses
Eloss = [-0.0061909968, -0.0031023139999999997, -0.03209591, -0.031536043, -0.027466562999999996, -0.016584065999999998, -0.004685778]
Eloss = [e*1000 for e in Eloss]
#Expected energy in detector
E_det = [E+Eloss[i] for i, E in enumerate(Etrue)]
calienergies = [E_det[0],E_det[1]]
#Masses in kg
mass = [938.27, 938.27, 3728.4, 3728.4, 3728.4, 3728.4, 2809.41] # MeV
mass = [sc.e*mass*10**6/(sc.c**2) for mass in mass]
#%% Load data from files and perform linear calibration
#Getting data for peaks and performing fitting.
titles = ['p577','p1815','a1400','a1464','a2050','a4750','t2730'] #Used in plotting
peaks_C = [] #Array to hold the channels with peaks
calib_peaks = [] #Channel for peaks used in calibration
stds = [] #Vector to hold the standard deviation in channels
mean_errs = [] #Error in mean estimated

peaks_E = [] #Vector to hold the means of peaks in keV   
results=[] #Vector for final results
phd = [] #Pulse height defect vector

#Load data and store x and y values
data = get_data(fnames)
xdata = data[0,:]
ydata = data[1:,:]
for i in range(len(ydata)):
    result = data_fit(xdata,ydata[i],0)
    stds.append(result[1])
    peaks_C.append(result[0])
    mean_errs.append(result[2])
    if (i==0):
        calib_peaks.append(peaks_C[i]) #, peaks[i]-stds[i],peaks[i]+stds[i]])
    elif (i==1):
        calib_peaks.append(peaks_C[i]) #, peaks[i]+stds[i],peaks[i]-stds[i]])
k, m = calibration(calib_peaks,calienergies)
print([k,m])

#Try error for k and m using normal propagation
p1 = -(calienergies[1]-calienergies[0])/(peaks_C[1]-peaks_C[0])**(2)
p2 = +(calienergies[1]-calienergies[0])/(peaks_C[1]-peaks_C[0])**(2)
k_err= np.sqrt((p1*mean_errs[0])**2 + (p2*mean_errs[1]**2))
m_err = np.sqrt((peaks_C[0]*k_err)**2 + (mean_errs[0]*k)**2)

#%% Translate from channels to energy and get fitting plots
for i in range(len(ydata)):
        plt.figure(figsize=(15,15))
        plt.subplot(221)
        plt.plot(xdata,ydata[i])
        plt.suptitle(f'Peak for {titles[i]}')
        plt.ylabel('Pulse height')
        plt.xlabel('Channel number')
        result = data_fit(xdata,ydata[i],1)
        calib_result = E_calib(k,m,xdata,ydata[i],peaks_C[i],stds[i])
        results.append(calib_result[0:2])
        peaks_E.append(calib_result[0])
        
#%%
#Plot calibrations and get PHD       
plt.figure(figsize=(12,10))
plt.plot(peaks_C,peaks_E,'o',markersize=3,label='Particle energies (calibrated)')
y = [k*x + m for x in xdata]
plt.plot(xdata,y,label='Calibration line')
plt.title('Calibration with only protons')
plt.xlabel('Pulse height')
plt.ylabel('Energy [keV]')
plt.legend()
#plt.savefig(f'calib_{i}.pdf',format='pdf')
phd.append(PHD(E_det,peaks_E))

plt.figure()
plt.plot(peaks_E,E_det,'*',markersize=3,label='Energy_true at calibrated peak energy')
plt.plot(E_det,E_det,linewidth=1,label='y=x line')
plt.title('Calibration using only protons')
plt.legend()
plt.ylabel('Energy_true')
plt.xlabel('Energy_measured')
#%% Fitting with all points and comparing k,m between fits
newfit = np.polyfit(peaks_C,E_det,1,rcond=None,cov=True)
gain, intercept = newfit[0]
sigma_gain = np.sqrt(newfit[1][0,0])
sigma_inter = np.sqrt(newfit[1][1,1])
plt.figure()
yy = [gain*x + intercept for x in xdata]
plt.plot(xdata,yy,linewidth=1,label='Calibration line')
plt.plot(peaks_C,E_det,'*',markersize=2,label='True energy')
plt.title('Calibration using all particles')
plt.ylabel('Energy_true')
plt.xlabel('Pulse Height')
plt.legend()

peakpeak = [gain*p+m for p in peaks_C]
plt.figure()
plt.plot(peakpeak,E_det,'*',markersize=3,label='Energy_true at calibrated peak energy')
plt.plot(E_det,E_det,linewidth=1,label='y=x line')
plt.title('Calibration using all particles')
plt.legend()
plt.ylabel('Energy_true')
plt.xlabel('Energy_measured')
plt.figure()
plt.title('Comparing gain and intercept between calibration methods')
plt.ylabel('Intercept')
plt.xlabel('Gain')
plt.errorbar(k,m,yerr=m_err,xerr=k_err,label='Proton only calibration +/- 1\u03c3')
plt.errorbar(gain,intercept,yerr=sigma_inter,xerr=sigma_gain,label='All particles calibration +/- 1\u03c3')
ax = plt.gca()
ellipse1 = pltp.Ellipse((k,m),2*k_err,2*m_err,edgecolor='r',fc='None')
ellipse2 = pltp.Ellipse((gain,intercept),2*sigma_gain,2*sigma_inter,edgecolor='r',fc='None')
ax.add_patch(ellipse1)
ax.add_patch(ellipse2)
plt.legend()

