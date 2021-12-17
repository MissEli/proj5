# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:13:01 2021

@author: ivewi
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm as norm
from scipy.optimize import curve_fit
#%%
fname = ['p1815.txt','p577.txt','a1400.txt','a1464.txt','a2050.txt','a4750.txt','t2730.txt']

def ff(x, *params):
    pdf = norm.pdf(x,params[1],params[2])*params[0] + params[3]
    return pdf

def pulse_plotter(xdata,ydata,lims,fit):
    plt.subplot(222)
    plt.plot(xdata,ydata,'b')
    plt.plot(xdata,fit,'r')
    plt.xlim(lims)
    plt.xlabel('Channel number')
    plt.ylabel('Pulse height')

def peak_finder(ydata,height=15,prominence=15,distance=20):
    peaks = find_peaks(ydata,height=height,prominence=prominence,distance=distance)
    peak_index = peaks[0]
    peak_heights = peaks[1]['peak_heights']
    left_bases = peaks[1]['left_bases']
    right_bases = peaks[1]['right_bases']

    peak_data = np.vstack((peak_index,peak_heights,left_bases,right_bases))
    return peak_data
    
def data_fit(xdata,ydata):
    #Create arrays to hold standard deviation and mean of fitted peaks
    peak = peak_finder(ydata)
    std = np.std(ydata)
    mean = xdata[peak[0][0].astype(int)]
    area = np.sum(ydata)

    #Initial guess for fitting using known parameters
    p0 = [area,mean,std,0]                   
    try:
        fitted = curve_fit(ff,xdata,ydata, p0)
        params, cov = fitted
        mean = params[1] #Extract channel using index from params
        std = params[2]
    #Sometimes fitting doesn't work but we don't want the program to stop
    except RuntimeError:
        print('Could not find optimal parameters for peak')
        params = p0 #Keep initial guess as fitting parameters for Gaussian
    lims = [peak[2][0],peak[3][0]]
    #Plotting
    #pulse_plotter(xdata,ydata,lims,ff(xdata,*params))
    #Save figures if savefig is given as "y"
        
            
    
    result = [mean,std]
    return result

def calibration(xdata,calipeak,calitime):
    k = 1
    m = calitime-calipeak
    print(m)
    return [k,m]

def T_calib(k,m,xdata,ydata,peak,std):
    times = [x*k+m for x in xdata]
    T_peak = peak*k+m
    T_std = std
    plt.subplot(212)
    plt.plot(times,ydata,'-')
    plt.xlabel('Time [ns]')
    plt.ylabel('Pulse height')
    output = [T_peak,std]
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

def main():
    #p p a a a a t
    ToF=[2.0569616549298337, 3.664777595671451, 4.719136521132885, 4.611566904522071, 3.880989880003998, 2.536898146663733, 2.9022070076326107]
    E = [1811.8976859999998, 570.8090032, 1370.5776578369482, 1428.936015304666, 2023.2640121302848, 4751.052015174993, 2714.916596558814]
    times = []
    peaks = []
    calib_peaks = []
    stds = []
    titles = ['p1815','p577','a1400','a1464','a2050','a4750','t2730']
    for name in fname:
        time = load(name)
        times.append(time)
    # rows, columns = times.shape
    results=[]
    pdt = []
    savefig = input('Save figures? [y/n]: ') #User choses to save figures or not
    for i in range(len(times)):
        # plt.figure(figsize=(15,15))
        # plt.subplot(221)
        T_hist = plt.hist(times[i],bins=2000, range=(0,150))
        # plt.suptitle(f'Peak for {titles[i]}')
        # plt.ylabel('Pulse height')
        # plt.xlabel('Channel number')
        xdata = np.delete(T_hist[1],0)
        ydata = T_hist[0]
        result = data_fit(xdata,ydata)
        stds.append(result[1])
        peaks.append(result[0])
        if i==0:
            k, m = calibration(xdata,peaks[0],ToF[0])
        calib_result = T_calib(k,m,xdata,ydata,peaks[i],stds[i])
        results.append(calib_result)
        calib_peaks.append(calib_result[0])
        if savefig=='y':
            plt.savefig(f'timePeak_{titles[i]}.png',format='png')
    pdt = PDT(ToF,calib_peaks)
    print(pdt)
    print(calib_peaks)

    c = ['b', 'g','r','c','m','k','y']
    plt.figure()
    plt.plot(E[:2],pdt[:2],'k-', label='proton')
    plt.plot(E[2:6],pdt[2:6],'k--', label='alpha')
    for i in range(len(pdt)):
        plt.plot(E[i],pdt[i],color=c[i],marker='*', label=titles[i])
    plt.legend()
    plt.xlabel('measured energy [keV]')
    plt.ylabel('PDT [ns]')

    plt.figure()
    for i in range(len(calib_peaks)):
        plt.plot(E[i],calib_peaks[i],color=c[i] ,marker='*', label=titles[i]+' m')
        plt.plot(E[i],ToF[i],color=c[i],marker='o', label=titles[i]+' t')
    plt.legend()
    plt.xlabel('measured energy [keV]')
    plt.ylabel('ToF [ns]')

    plt.show()
    return [results,pdt]
        
if __name__ == "__main__":
    main()   


    

