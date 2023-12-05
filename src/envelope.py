import numpy as np

#Collects the envelopes used for Rabi with envelopes calculations
#In future use some kind of polymorphism?

class gauss:
    def __init__(self,params):
        self.tau = params[0]
        return

    def env(self,t):
        return np.exp(-2*np.log(2)*(t/self.tau)**2)
    

class n_gauss:
    def __init__(self,params):
        self.tau = params[0]
        self.n = params[1]
        return
    
    def env(self,t):
        return np.exp(-np.log(2)*2**(2*self.n-1)*(t/self.tau)**(2*self.n))

class double_gauss:
    def __init__(self,params):
        self.tau = params[0]
        self.delay = params[1]
        return
    
    def env(self,t):
        env_1 = np.exp(-2*np.log(2)*(t/self.tau)**2)
        env_2 = np.exp(-2*np.log(2)*((t-self.delay)/self.tau)**2)

        return env_1+env_2
    

class flat_top:
    def __init__(self,params):
        self.tau = params[0]
        return
    
    def env(self,t):
        return np.heaviside(t+0.5*self.tau,1.0)-np.heaviside(t-0.5*self.tau,1.0)
    

class double_flat_top:
    def __init__(self,params):
        self.tau = params[0]
        self.delay = params[1]
        self.ft = flat_top([self.tau])
        return
    
    def env(self,t):
        return self.ft.env(t) + self.ft.env(t-self.delay)