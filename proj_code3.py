# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:38:10 2021

Code for analysing energy data
"""
#%% Importing packages
#Import packages
import math as math
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes',titlesize=20)
plt.rc('axes',labelsize=20)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
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
    labels = ['Proton577','Proton1815','\u03b11400','\u03b11464','\u03b12050','\u03b14750','Triton2730']
    for i in range(len(energies)):
        E_hist = plt.hist(energies[i],bins=5000,range=(0,200),label=labels[i])  
        if i == 0:
            xdata = [e+0.02 for e in E_hist[1]]
            xdata = np.delete(xdata,len(xdata)-1)
        ydata.append(E_hist[0])
    plt.xlabel('Pulse height')
    plt.ylabel('Count')
    plt.legend()
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
def pulse_plotter(xdata,ydata,fit,xlabel,title):
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.plot(xdata,ydata,'b')
    plt.plot(xdata,fit,'r')
    plt.xlabel(xlabel)
    plt.ylabel('Counts')
    
#Function to fit a normal distribution onto a histogram peak
def data_fit(xdata,ydata,plot,xlabel='',title=''):
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
    #Initial guess fory fitting using known parameters
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
        mean_err = 0
    
    #Plotting
    if plot:    
        pulse_plotter(x,y,ff(x,*params),xlabel,title)
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
#mass = [sc.e*mass*10**6/(sc.c**2) for mass in mass]
#%% Load data from files and perform linear calibration
#Getting data for peaks and performing fitting.
filenames = ['p577','p1815','a1400','a1464','a2050','a4750','t2730']
titles = ['Proton of 577 keV', 'Proton of 1815 keV',
          '\u03b1 of 1400 keV', '\u03b1 of 1464 keV', '\u03b1 of 2050 keV', '\u03b1 of 4750 keV',
          'Triton of 2730 keV']
labels = ['Proton577','Proton1815','\u03b11400','\u03b11464','\u03b12050','\u03b14750','Triton2730'] #Used in plotting


peaks_C = [] #Array to hold the channels with peaks
calib_peaks = [] #Channel for peaks used in calibration
stds = [] #Vector to hold the standard deviation in channels
mean_errs = [] #Error in mean estimated

peaks_E = [] #Vector to hold the means of peaks in keV   
results=[] #Vector for final results
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
p3 = +1/(peaks_C[1]-peaks_C[0])
p4 = -1/(peaks_C[1]-peaks_C[0])
k_err= np.sqrt((p1*mean_errs[0])**2 + (p2*mean_errs[1]**2) + 2*(0.04*p3)**2)
m_err = np.sqrt(0.04**2 + (peaks_C[0]*k_err)**2 + (mean_errs[0]*k)**2)

peak_errors = [np.sqrt((k*mean_errs[i])**2 + \
                       (k_err*peaks_C[i])**2 + \
                           m_err**2) for i in range(len(peaks_C))]

#%% Translate from channels to energy and get fitting plots
savefig = input('Save figures?: [y/n]')
for i in range(len(ydata)):
        result = data_fit(xdata,ydata[i],1,'Pulse Height',titles[i])
        if savefig:
            plt.savefig(f'{filenames[i]}.png',bbox_inches='tight')
        calib_result = E_calib(k,m,xdata,ydata[i],peaks_C[i],stds[i])
        results.append(calib_result[0:2])
        peaks_E.append(calib_result[0])
phd = PHD(E_det,peaks_E)
phd_err = [np.sqrt(peak_errors[i]**2 + 0.04**2) for i in range(len(peak_errors))]


#%%
#Plot calibrations      
plt.figure(figsize=(12,10))
plt.plot(peaks_C,peaks_E,'o',markersize=3,label='Particle energies (calibrated)')
y = [k*x + m for x in xdata]
plt.plot(xdata,y,label='Calibration line')
plt.title('Calibration with only protons')
plt.xlabel('Pulse height')
plt.ylabel('Energy [keV]')
plt.legend()
#plt.savefig(f'calib_{i}.pdf',format='pdf')
#phd.append(PHD(E_det,peaks_E))

plt.figure()
plt.plot(peaks_E,E_det,'*',markersize=3,label='Energy_true at calibrated peak energy')
plt.plot(E_det,E_det,linewidth=1,label='y=x line')
plt.title('Calibration using only protons')
plt.legend()
plt.ylabel('Energy_true')
plt.xlabel('Energy_measured')
x_energy = calib_result[2]

for i in range(len(ydata)):
        result = data_fit(x_energy,ydata[i],1,'Energy [keV]',titles[i])
        if savefig:
            plt.savefig(f'{filenames[i]}_E1.png',bbox_inches='tight')
#%% Final histogram, calibration 1
plt.figure(figsize=(15,10))
# log_energy = [m.log(x) for x in x_energy]
for i in range(len(ydata)):
    plt.plot(x_energy,ydata[i],label=labels[i])
plt.xlim((500,5000))
plt.title('Energy histogram')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts / bin')
plt.legend(loc='best',prop={'size': 15})
plt.savefig('energy_hist.png')
#%% Fitting with all points and comparing k,m between fits
new_results = []
new_peaks_E = []
newfit = np.polyfit(peaks_C,E_det,1,rcond=None,cov=True)
gain, intercept = newfit[0]
sigma_gain = np.sqrt(newfit[1][0,0])
sigma_inter = np.sqrt(newfit[1][1,1])

for i in range(len(ydata)):
    new_calib_result = E_calib(gain,intercept,xdata,ydata[i],peaks_C[i],stds[i])
    new_results.append(new_calib_result[0:2])
    new_peaks_E.append(new_calib_result[0])
new_x_energy = new_calib_result[2]

new_peak_errors = [np.sqrt((gain*mean_errs[i])**2 + \
                       (sigma_gain*peaks_C[i])**2 + \
                           sigma_inter**2) for i in range(len(peaks_C))]

for i in range(len(ydata)):
    new_result = data_fit(new_x_energy,ydata[i],1,'Energy [keV]',titles[i])
    if savefig:
        plt.savefig(f'{filenames[i]}_E2.png',bbox_inches='tight')
        
new_phd = PHD(E_det,new_peaks_E)
new_phd_err = [np.sqrt(new_peak_errors[i]**2 + 0.04**2) for i in range(len(peak_errors))]


   
plt.figure()
yy = [gain*x + intercept for x in xdata]
plt.plot(xdata,yy,linewidth=1,label='Calibration line')
plt.plot(peaks_C,E_det,'*',markersize=2,label='True energy')
plt.title('Calibration using all particles')
plt.ylabel('Energy_true')
plt.xlabel('Pulse Height')
plt.legend()

peakpeak = [gain*p+intercept for p in peaks_C]
plt.figure()
plt.plot(new_peaks_E,E_det,'*',markersize=3,label='Energy_true at calibrated peak energy')
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
#%% Final histogram with calibration 2
plt.figure(figsize=(15,10))
# log_energy = [m.log(x) for x in new_x_energy]
for i in range(len(ydata)):
    plt.plot(new_x_energy,ydata[i],label=labels[i])
plt.xlim((500,5000))
plt.title('Energy histogram')
plt.xlabel('Energy [keV]')
plt.ylabel('Counts/bin')
plt.legend()
plt.savefig('energy_hist2.png')
#%% PHD plots with errorbars
# plt.figure()
# for i in range(len(E_det)):
#     plt.errorbar(E_det[i],new_peaks_E[i],yerr=new_peak_errors[i],marker='*',markersize=1,capsize=3,label=labels[i])
# plt.plot(E_det,E_det,linewidth=1,label='y=x line')
# plt.title('Calibration using all particles')
# plt.legend(bbox_to_anchor=(1.05, 1),
#                          loc='upper left', borderaxespad=0.)
# plt.ylabel('Energy_true')
# plt.xlabel('Energy_measured')
# plt.savefig('errorbar_calib.svg',bbox_inches='tight',format='svg')
shortlabels = ['p577','p1815','\u03b11400','\u03b11464','\u03b12050','\u03b14750','t2730']
colors = ['r','b','y','c','m','g','k']
colors2 = ['tab:purple','tab:blue','tab:orange','tab:pink','tab:brown','tab:olive','tab:gray']

f, (ax1,ax2) = plt.subplots(1,2,sharex=True, sharey = True,figsize=(12,6))

for i in range(len(E_det)):
    ax2.errorbar(new_peaks_E[i],new_phd[i],yerr=new_phd_err[i],color=colors[i],marker='*',capsize=5,label=f'{shortlabels[i]}')
    ax1.errorbar(new_peaks_E[i],phd[i],yerr=phd_err[i],color=colors[i],marker='o',markersize=3,capsize=5,label=f'{shortlabels[i]}')
plt.legend()
ax1.set_title('Calibrating with only protons')
ax2.set_title('Calibrating with all particles')
ax1.set_ylabel('Pulse height defect [keV]')
ax1.set_xlabel('Energy measured [keV]')
ax2.set_xlabel('Energy measured [keV]')
plt.savefig('phds.png',bbox_inches='tight')

#%% Peak translated to time
s = 38.35*10**(-3)
ToF=[3.664777595671451,2.0569616549298337,4.719136521132885,4.611566904522071,3.880989880003998,2.536898146663733,2.9022070076326107]

x_v = [np.sqrt(2*e*10**(-3)/mass[1])*sc.c for e in new_x_energy]
x_time = [10**9*s/v for v in x_v]
peak = peak_finder(ydata[1]) #Find the peak
std = np.std(ydata[1])
mean = new_x_energy[peak[0][0].astype(int)]
area = np.sum(ydata[1])
#Make new x-vector only over peak area
x1 = peak[2][0].astype(int)
x2 = peak[3][0].astype(int)
x = []
y=[]
for i in range(x1,x2+1):
    x.append(new_x_energy[i])
    y.append(ydata[1][i])
x_t = [10**9*s/(np.sqrt(2*e*10**(-3)/mass[1])*sc.c) for e in x]
#Initial guess fory fitting using known parameters
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
    mean_err = 0
   
plt.figure()
plt.title('Time of flight for p1815 converted from energy',fontsize=15)
plt.plot(x_t,y,'b')
plt.plot(x_t,ff(x,*params),'r')
plt.axvline(x=ToF[1],color='k')
plt.xlabel('Time [ns]')
plt.ylabel('Counts')    
  
plt.axvline(x=ToF[1],color='k',label='Calculated ToF')
plt.legend()
plt.tight_layout()
plt.savefig('e_to_t_peak.png')