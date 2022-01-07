# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:13:01 2021

@author: ivewi
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes',titlesize=20)
plt.rc('axes',labelsize=20)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
from scipy.signal import find_peaks
from scipy.stats import norm as norm
from scipy.optimize import curve_fit
#%% Functions
def ff(x, *params):
    pdf = norm.pdf(x,params[1],params[2])*params[0] + params[3]
    return pdf

def pulse_plotter(xdata,ydata,fit,xlabel,title):
    plt.figure()
    plt.title(title)
    plt.plot(xdata,ydata,'b')
    plt.plot(xdata,fit,'r')
    plt.xlabel(xlabel)
    plt.ylabel('Pulse height')
    plt.tight_layout()

def peak_finder(ydata,height=15,prominence=15,distance=50):
    peaks = find_peaks(ydata,height=height,prominence=prominence,distance=distance)
    peak_index = peaks[0]
    peak_heights = peaks[1]['peak_heights']
    left_bases = peaks[1]['left_bases']
    right_bases = peaks[1]['right_bases']

    peak_data = np.vstack((peak_index,peak_heights,left_bases,right_bases))
    return peak_data
    
def data_fit(xdata,ydata,plot,xlabel='',title=''):
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
        fitted = curve_fit(ff,xdata,ydata, p0)
        params, cov = fitted
        mean = params[1] #Extract channel using index from params
        std = params[2]
        mean_err = np.sqrt(cov[1,1])
    #Sometimes fitting doesn't work but we don't want the program to stop
    except RuntimeError:
        print('Could not find optimal parameters for peak')
        params = p0 #Keep initial guess as fitting parameters for Gaussian
        
    if plot:    
        pulse_plotter(x,y,ff(x,*params),xlabel,title)
        
            
    
    result = [mean,std,mean_err]
    return result

def calibration(xdata,calipeak,calitime):
    k = 1
    m = calitime-calipeak
    return [k,m]

def T_calib(k,m,xdata,ydata,peak,std):
    times = [x*k+m for x in xdata]
    T_peak = peak*k+m
    T_std = std
    output = [T_peak,std,times]
    return output

def PDT(ToF,T_peaks):
    pdt=[] #Energy with losses accounted
    for i in range(len(T_peaks)):
        pdt.append(T_peaks[i]-ToF[i])
    return pdt

def load(fname):
    file = np.loadtxt(fname, dtype=float,delimiter=',', skiprows=1)
    T = file[:,1]
    return T

def get_data(fnames):
    times = []
    for name in fnames:
        time = load(name)
        times.append(time)
    ydata = []
    
    for i in range(len(times)):
        T_hist = plt.hist(times[i],bins=2000, range=(0,150))
        if i == 0:
            xdata = [t+0.075 for t in T_hist[1]]
            xdata = np.delete(xdata,len(xdata)-1)

        ydata.append(T_hist[0])
        
    plt.xlabel('Pulse height')
    plt.ylabel('Count')
    plt.legend()
    return np.vstack((xdata,ydata))

#%% Load parameters
#p p a a a a t
fnames = ['p1815.txt','p577.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']
ToF=[2.0569616549298337, 3.664777595671451, 4.719136521132885, 4.611566904522071, 3.880989880003998, 2.536898146663733, 2.9022070076326107]
ToF_err =[0.11239114904430238,0.2002412461030443,0.25785078435268605,0.25197324512672664,0.2120549554229963,0.13861459061648096, 0.15857484730962676]
E = [1811.8976859999998, 570.8090032, 1370.5776578369482, 1428.936015304666, 2023.2640121302848, 4751.052015174993, 2714.916596558814]

#%% Load data and fit
filenames = ['p1815','p577','a1400','a1464','a2050','a4750','t2730']
titles = [ 'Proton of 1815 keV','Proton of 577 keV',
          '\u03b1 of 1400 keV', '\u03b1 of 1464 keV', '\u03b1 of 2050 keV', '\u03b1 of 4750 keV',
          'Triton of 2730 keV']
labels = ['Proton1815','Proton577','\u03b11400','\u03b11464','\u03b12050','\u03b14750','Triton2730'] #Used in plotting

peaks_C = []
calib_peaks = []
stds = []
mean_errs = []

peaks_T = []
results=[]
pdt = []

data = get_data(fnames)
xdata = data[0,:]
ydata = data[1:,:]

for i in range(len(ydata)):
    result = data_fit(xdata,ydata[i],1,'Pulse Height',titles[i])
    stds.append(result[1])
    peaks_C.append(result[0])
    mean_errs.append(result[2])
    if i==0:
        k, m = calibration(xdata,peaks_C[0],ToF[0])
    calib_result = T_calib(k,m,xdata,ydata,peaks_C[i],stds[i])
    results.append(calib_result[0:2])
    calib_peaks.append(calib_result[0])
#%% calculate pdt
x_time = calib_result[2]
for i in range(len(ydata)):
    result = data_fit(x_time,ydata[i],1,'Time [ns]',titles[i])
    plt.savefig(f'time_{filenames[i]}.png')
pdt = PDT(ToF,calib_peaks)

#Calculate pdt error
err_m = ToF_err[0]
err_k = 0
peak_errs = [np.sqrt(err_x**2 + err_m**2) for err_x in mean_errs]
pdt_err = [np.sqrt(ToF_err[i]**2 + peak_errs[i]**2) for i in range(len(peak_errs))]
print(pdt)
print(calib_peaks)
#%%
c = ['b', 'g','r','c','m','k','y']
plt.figure()
plt.plot(E[:2],pdt[:2], label='proton')
plt.plot(E[2:6],pdt[2:6],label='alpha')
for i in range(len(pdt)):
    plt.plot(E[i],pdt[i],color=c[i],marker='*', label=titles[i])
plt.legend()
plt.xlabel('measured energy')
plt.ylabel('PDT')

plt.figure()
for i in range(len(calib_peaks)):
    plt.plot(E[i],calib_peaks[i],color=c[i] ,marker='*', label=titles[i]+' m')
    plt.plot(E[i],ToF[i],color=c[i],marker='o', label=titles[i]+' t')
plt.legend()
plt.xlabel('measured energy')
plt.ylabel('ToF')
plt.show()

peak = peak_finder(ydata[0])
std = np.std(ydata[0])
mean = x_time[peak[0][0].astype(int)]
area = np.sum(ydata[0])

x1 = peak[2][0].astype(int)
x2 = peak[3][0].astype(int)
x = []
y=[]
for i in range(x1,x2+1):
    x.append(x_time[i])
    y.append(ydata[0][i])
#Initial guess for fitting using known parameters
p0 = [area,mean,std,0]                   
try:
    fitted = curve_fit(ff,x,y, p0)
    params, cov = fitted
    mean = params[1] #Extract channel using index from params
    std = params[2]
    mean_err = np.sqrt(cov[1,1])
#Sometimes fitting doesn't work but we don't want the program to stop
except RuntimeError:
    print('Could not find optimal parameters for peak')
    params = p0 #Keep initial guess as fitting parameters for Gaussian
    
plt.figure()
plt.title('Time of flight for p1815')
plt.plot(x,y,'b')
plt.plot(x,ff(x,*params),'r')
plt.axvline(x=ToF[0],color='k')
plt.xlabel('Time [ns]')
plt.ylabel('Counts')   
plt.tight_layout() 
plt.savefig('timepeak.png')



        

    

