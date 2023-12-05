import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bloch_trajectories(coeffs,time):
    #Plot the Bloch-vector of a two-level system as a function of time

    #State coefficients
    a = coeffs[0,:]
    b = coeffs[1,:]

    #Initialize the density matrix and the Pauli matrices
    rho = np.zeros((2,2),dtype = np.complex128)
    sigma_z = np.array([[1,0],[0,-1]],dtype = np.complex128)
    sigma_y = np.array([[0,-1j],[1j,0]],dtype = np.complex128)
    sigma_x = np.array([[0,1],[1,0]],dtype = np.complex128)

    #Initialize the Bloch-vector components
    r_x = np.zeros(time.shape[0],dtype=np.float64)
    r_y = np.zeros(time.shape[0],dtype=np.float64)
    r_z = np.zeros(time.shape[0],dtype=np.float64)

    #Compute the elements of the density matrix and the Bloch vector for each time step
    for index,t in np.ndenumerate(time):
        rho[0,0] = np.abs(a[index])**2
        rho[1,1] = np.abs(b[index])**2
        rho[0,1] = b[index]*np.conjugate(a[index])
        rho[1,0] = np.conjugate(rho[0,1])
        r_x[index] = np.trace(sigma_x@rho)
        r_y[index] = np.trace(sigma_y@rho)
        r_z[index] = np.trace(sigma_z@rho)

    #Plot the Bloch vector in 3D
    fig,ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(r_x,r_y,r_z)
    
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    ax.set_xlabel('$r_x$')
    ax.set_ylabel('$r_y$')
    ax.set_zlabel('$r_z$')

    return
