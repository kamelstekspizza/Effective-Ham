#import general libraries
import sys
import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import scipy.fft as sft
from scipy.special import erf

#Import local code
from constants import *
from effective_Hamiltonian import complex_Rabi
from plot_utils import format_plot

class two_level_system:
    #Solve two level system with envelope

    def __init__(self,omega_ba,z_ba,omega,E_0,envelope,phi):
        #solve two-level system with arbitrary envelope funciton and CEP
        self.omega_ba = omega_ba
        self.z_ba = z_ba
        self.omega = omega
        self.E_0 = E_0
        self.f = envelope
        self.phi = phi

        self.Rabi_0 = self.E_0*self.z_ba

        self.H = np.zeros((2,2),dtype = np.complex128)
        self.H[0,0] = 0
        self.H[1,1] = self.omega_ba

        return

    def E(self,t):
        return self.f(t)*self.E_0*np.cos(self.omega*t+self.phi)

    def H_update(self,t):
        interaction = self.E(t)*self.z_ba
        self.H[1,0] = interaction
        self.H[0,1] = interaction

        return
    
    def H_eval(self,t):
        self.H_update(t)
        return self.H
    
    def RHS(self,t,alpha):
        self.H_update(t)
        return -1j*self.H@alpha


class effective_two_level:
    #Solve two level system with effective Hamiltonian used in Olofsson and Dahlström PRR (2023)

    def __init__(self,E_1,E_2,E_0,dipole,omega,shifts,envelope,delay = 0):

        self.E_1 = E_1
        self.E_2 = E_2
        self.E_0 = E_0
        self.z_ba = dipole
        self.omega = omega
        self.shifts = shifts
        self.envelope = envelope
        self.delay = delay

        self.Rabi_0 = self.E_0*self.z_ba
        self.omega_ba = self.E_2

        self.H = np.zeros((2,2),dtype = np.complex128)
        

        return
    
    def f(self,t):
        return self.envelope(t-self.delay)
    
    def E(self,t):
        return self.f(t)*self.E_0*np.cos(self.omega*(t-self.delay))
    
    def E_deriv(self,t,h=1e-7):
        return (self.E(t+h)-self.E(t-h))/(2*h)
     
    
    def H_update(self,t):
        E = self.E_0*self.f(t)
        
        self.c_R = complex_Rabi(self.E_1,self.E_2,E,self.z_ba,self.omega)
        self.c_R.shifts_to_order_4(self.shifts[0],self.shifts[1],self.shifts[2],self.shifts[3],self.shifts[4],disp = False)
        self.c_R.compute_eigenvalues(disp = False)

        self.H[0,0] = self.c_R.h_11
        self.H[1,0] = self.c_R.h_12*np.exp(1j*self.omega*self.delay)
        self.H[0,1] = self.c_R.h_12*np.exp(-1j*self.omega*self.delay)
        self.H[1,1] = self.c_R.h_22

        return

    def H_eval(self,t):
        self.H_update(t)
        return self.H
    
    def RHS(self,t,alpha):
        self.H_update(t)
        return -1j*self.H@alpha



def solve(system,a_0,b_0,t_vec):

    alpha_0 = np.array([a_0,b_0],dtype = np.complex128)
    limits = (t_vec[0],t_vec[-1])

    print('')
    print('------------------------')
    print('Propagating equations...')
    print('------------------------')
    print('')

    solution = si.solve_ivp(system.RHS,limits,alpha_0,method = 'DOP853',t_eval = t_vec, max_step = 0.1*2*np.pi/system.omega)
    #solution = si.solve_ivp(system.RHS,limits,alpha_0,method = 'RK45',t_eval = t_vec, max_step = 0.1*2*np.pi/system.omega)

    system.time = solution.t
    system.alpha_final = solution.y
    system.population = np.abs(system.alpha_final[0,:])**2 + np.abs(system.alpha_final[1,:])**2

    return

def plot(system):
    #Plot the two-level populations
    fig_pop,ax_pop = plt.subplots()

    ax_pop.plot(system.time/fs_to_au,np.abs(system.alpha_final[0,:])**2,label = 'ground state')
    ax_pop.plot(system.time/fs_to_au,np.abs(system.alpha_final[1,:])**2,'--',label = 'excited state')
    ax_pop.plot(system.time/fs_to_au,system.population,':',label = 'Norm')

    ax_pop.legend(fontsize = 16)

    format_plot(fig_pop,ax_pop,'Time [fs]','Population')

    #Plot the fourier transform of the state amplitudes
    fig_fft,ax_fft = plt.subplots()

    a_fft = sft.fft(system.alpha_final[0,:])
    a_fft = sft.fftshift(a_fft)

    b_fft = sft.fft(system.alpha_final[1,:])
    b_fft = sft.fftshift(b_fft)

    t_fft = sft.fftfreq(system.time.shape[0],system.time[1]-system.time[0])
    t_fft = sft.fftshift(t_fft) 


    ax_fft.semilogy(2*np.pi*t_fft,np.abs(a_fft)/np.max(np.abs(a_fft)),label = '$c_a(\omega)$')
    ax_fft.semilogy(2*np.pi*t_fft,np.abs(b_fft)/np.max(np.abs(b_fft)),'--',label = '$c_b(\omega)$')
    ax_fft.vlines(system.Rabi_0/2,0,1,'c',linestyles='dashed',label = '$\Omega_0$')
    ax_fft.vlines(-system.Rabi_0/2,0,1,'c',linestyles='dashed',label = '$\Omega_0$')

    format_plot(fig_fft,ax_fft,'Frequency [a.u]','$|c_i(\omega)|$')

    #Plot the electric field and its fourier tranform
    fig_E,ax_E = plt.subplots()
    E = system.E(system.time)
    ax_E.plot(system.time/fs_to_au,E)
    format_plot(fig_E,ax_E,'Time [fs]','Electric field [a.u.]')

    E_fft = sft.fft(E)
    E_fft = sft.fftshift(E_fft)
    fig_E_fft,ax_E_fft = plt.subplots()
    ax_E_fft.semilogy(2*np.pi*t_fft,np.abs(E_fft)**2)
    format_plot(fig_E_fft,ax_E_fft,'Frequency[a.u.]','$|E(\omega)|^2$')

    #Plot fourier transform of f(t)b(t) and f(t)^2 a(t)
    fb_fft = sft.fft(system.f(system.time)*system.alpha_final[1,:]*np.exp(1j*system.time*system.H[1,1]))
    fb_fft = sft.fftshift(fb_fft)
    fa_fft = sft.fft(system.f(system.time)**2*system.alpha_final[0,:]*np.exp(1j*system.time*system.H[0,0]))
    fa_fft = sft.fftshift(fa_fft)

    fig_fb,ax_fb = plt.subplots()
    ax_fb.plot(2*np.pi*t_fft,np.abs(fb_fft)**2/np.max(np.abs(fb_fft)**2),label = 'b')
    ax_fb.plot(2*np.pi*t_fft,np.abs(fa_fft)**2/np.max(np.abs(fa_fft)**2),label = 'a')
    ax_fb.vlines(system.Rabi_0/2,0,1,'c',linestyles='dashed',label = '$\Omega_0$')
    ax_fb.vlines(-system.Rabi_0/2,0,1,'c',linestyles='dashed',label = '$\Omega_0$')
    ax_fb.legend()
    format_plot(fig_fb,ax_fb,'Frequency[a.u.]','$|fb(\omega)|^2$')

    return

def compute_spectrum(system,matels,E_range,E_points,I_p,plot=False):
    dt = system.time[1]-system.time[0]
    E_vec = np.linspace(E_range[0],E_range[1],E_points)
    spec = np.zeros(E_points,dtype = np.complex128)
    spec_a  = np.zeros(E_points,dtype = np.complex128)
    spec_b  = np.zeros(E_points,dtype = np.complex128)

    print('')
    print('---------------------')
    print('Computing spectrum...')
    print('---------------------')
    print('')


    for index,E in np.ndenumerate(E_vec):
        #b_contr = 0
        b_integrand = -1j*system.E_0*system.f(system.time)/2*matels[1]*system.alpha_final[1,:]*np.exp(-1j*(2*system.omega-E)*system.time)
        b_contr = si.simps(b_integrand,dx = dt)*np.exp(1j*system.delay*system.omega)

        a_integrand = -1j*(system.E_0*system.f(system.time)/2)**2*matels[0]*system.alpha_final[0,:]*np.exp(-1j*(2*system.omega-E)*system.time)
        a_contr = si.simps(a_integrand,dx = dt)*np.exp(1j*2*system.delay*system.omega)

        spec[index] = b_contr + a_contr
        spec_a[index] = a_contr
        spec_b[index] = b_contr


    if plot:
        fig_spec,ax_spec = plt.subplots()
        ax_spec.plot(E_vec-I_p,np.abs(spec)**2,label = 'Total')
        ax_spec.plot(E_vec-I_p,np.abs(spec_b)**2,'--',label = 'b contribution')
        ax_spec.plot(E_vec-I_p,np.abs(spec_a)**2,':',label = 'a contribution')
        ax_spec.legend()
        format_plot(fig_spec,ax_spec,'Energy[a.u.]','Photoelectron yield [arb. u.]')

    return spec

def compute_absorption(system,omega_vec,t_1,sigma):

    print('')
    print('---------------------')
    print('Computing absorption...')
    print('---------------------')
    print('')


    omega_fft = 2*np.pi*sft.fftfreq(system.time.shape[0],system.time[1]-system.time[0])
    omega_fft = sft.fftshift(omega_fft)

    amplitude_factor = 2*np.real(system.alpha_final[0,:]*np.conjugate(system.alpha_final[1,:]*np.exp(-1j*system.omega*system.time)))
    amplitude_factor_2 = (system.alpha_final[0,:]*np.conjugate(system.alpha_final[1,:]*np.exp(-1j*system.omega*system.time)) 
                        + np.conjugate(system.alpha_final[0,:])*system.alpha_final[1,:]*np.exp(-1j*system.omega*system.time))
    dipole = system.z_ba*amplitude_factor
    dipole_2 = system.z_ba*amplitude_factor_2

    #dipole = np.transpose(dipole)

    dipole_fft = sft.fft(dipole)
    dipole_fft = sft.fftshift(dipole_fft)

    dipole_ft = np.zeros(omega_vec.shape,dtype=np.complex128)
    E_ft = np.zeros(omega_vec.shape,dtype=np.complex128)
    prod_ft = np.zeros(omega_vec.shape,dtype = np.complex128)
    #absorption = np.zeros(omega_vec.shape,dtype=np.complex128)

    E = system.E(system.time)
    E_fft = sft.fft(E)
    E_fft = sft.fftshift(E_fft)

    prod_fft = sft.fft(dipole*system.E_deriv(system.time))
    prod_fft = sft.fftshift(prod_fft)
    

    step = 1-0.5*(1+erf((system.time-t_1)/(sigma*np.sqrt(2))))

    for index,omega in np.ndenumerate(omega_vec):
        dip_int = dipole*np.exp(1j*omega*system.time)*step
        dipole_ft[index] = si.simps(dip_int,dx = system.time[1]-system.time[0])/np.sqrt(2*np.pi)
        E_int = E*np.exp(1j*omega*system.time)
        E_ft[index] = si.simps(E_int,dx = system.time[1]-system.time[0])/np.sqrt(2*np.pi)
        prod_ft[index] = si.simps(dipole*system.E_deriv(system.time)*np.exp(1j*omega*system.time),dx = system.time[1]-system.time[0])/np.sqrt(2*np.pi)

    abs_tot = si.simps(dipole*system.E_deriv(system.time),dx = system.time[1]-system.time[0])

    print(f'Total absorption: {abs_tot}')

    absorption = -np.imag(2*omega_vec*dipole_ft*np.conjugate(E_ft))
    #absorption = prod_ft/omega_vec
    #absorption = np.imag(omega_vec*dipole_ft/E_ft)

    fig,ax = plt.subplots()
    #ax.plot(system.time,dipole)
    #ax.plot(system.time,step)
    #ax.plot(system.time,dipole*system.f(system.time))
    #ax.plot(system.time,dipole*system.E_deriv(system.time))
    #.plot(system.time,system.E(system.time))
    #ax.plot(omega_vec,np.abs(prod_ft))
    #ax.plot(omega_vec,np.abs(E_ft))
    ax.plot(omega_fft,prod_fft)
    return absorption


