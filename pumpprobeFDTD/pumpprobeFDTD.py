import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.signal import convolve2d
import scipy
from . import myFFT

# unit conversion factors
ps = 299.79246
kV_cm = 1/3767.30314
THz2k = 1e-2*(2*np.pi)/2.9979246 # k in um^-1 when n = 1

class component_index:
    def __init__(self):
        self.Ex = 0
        self.Ey = 1
        self.Ez = 2
        self.Dx = 3
        self.Dy = 4
        self.Dz = 5
        self.Hx = 6
        self.Hy = 7
        self.Hz = 8
        self.Bx = 9
        self.By = 10
        self.Bz = 11
        self.Pz = [12,13]
        self.Jpz = [14,15]
        self.curlEzx = 0
        self.curlEzy = 1
        self.curlB_z = 2
        self.Fz = [0,1]
        self.paramz = [2,3]
comp = component_index()
class simulation:
    def __init__(self,cell,Courant):
        self.fields = np.zeros((16,cell["y"]+1,cell["x"]+1))
        self.curls = np.zeros((3,cell["y"]+1,cell["x"]+1))
        self.forces = np.zeros((4,cell["y"]+1,cell["x"]+1))
        self.time = 0
        # self.fields[comp.Ez]
        self.Courant = Courant
        self.dt = Courant
        self.Mur_r = (1-Courant/2.4)/(1+Courant/2.4)
        self.epsilon = np.ones((cell["y"]+1,cell["x"]+1))
        self.Lorentzian = []
        self.Lorentzian_pristine = []
        self.coupled = []
        self.force = []
        self.fparam = []
        self.yf = np.zeros((4,cell["y"]+1,cell["x"]+1)) # y coordinates used to create force array
        self.ys =  np.ones((cell["y"]+1,cell["x"]+1))*np.arange(cell["y"]+1)[:,np.newaxis] # y coordinates across the cell
        self.cell = cell
    def curl_E(self):
        '''
        Compute the curl field of E, taking care of boundary conditions
        '''
        self.curls[comp.curlEzx][1:] = convolve2d(self.fields[comp.Ez],[[1],[-1]],mode="valid")
        self.curls[comp.curlEzx][0] =  self.fields[comp.Ez][0]-0 # Metallic BC along y direction
        self.curls[comp.curlEzy][:,1:] = -convolve2d(self.fields[comp.Ez],np.array([[1,-1]]),mode="valid")
        self.curls[comp.curlEzy][:,0] = -(self.fields[comp.Ez][:,0] - self.fields[comp.Ez][:,-1]) # PBC along x direction
    def curl_B(self):
        '''
        Compute the curl field of B, taking care of boundary conditions
        '''
        self.curls[comp.curlB_z][:,:-1] = convolve2d(self.fields[comp.By],np.array([[1,-1]]),mode="valid")
        self.curls[comp.curlB_z][:,-1] = self.fields[comp.By][:,0] - self.fields[comp.By][:,-1] # PBC along x direction
        self.curls[comp.curlB_z][:-1] -= convolve2d(self.fields[comp.Bx],np.array([[1],[-1]]),mode="valid")
        self.curls[comp.curlB_z][-1] -= (0-self.fields[comp.Bx][-1]) # Magnetic BC along y direction
    def add_Lorentzian(self,f,b1E,bE1,gamma,xstart,xend,ystart,yend):
        '''
        Add a lorentzian oscillator with parameters f, b1E, bE1, gamma defined in a rectangel xstart<x<xend, ystart<y<yend
        '''
        cell = self.cell
        self.Lorentzian.append({"f":f,"b1E":b1E,"bE1":bE1,"gamma":gamma,"block":np.zeros((cell["y"]+1,cell["x"]+1))})
        L = self.Lorentzian[-1]
        L.update({"w":L["f"]*(2*np.pi),"Gamma":L["gamma"]*(2*np.pi)})
        L.update({"c1":(2-self.dt*L["Gamma"])/(2+self.dt*L["Gamma"])})
        L.update({"c2":(2*self.dt)/(2+self.dt*L["Gamma"])})
        for i in range(int(ystart),int(yend)+1):
            for j in range(int(xstart),int(xend)+1):
                L["block"][i,j] = 1
        self.Lorentzian_pristine.append(L.copy())
        
    def add_coupledmode(self,f,g,gamma,xstart,xend,ystart,yend):
        '''
        Add a mode that couples to the Lorentzian mode. Not implemented for NiI2
        '''
        cell = self.cell
        self.coupled.append({"f":f,"g":g,"gamma":gamma,"block":np.zeros((cell["y"]+1,cell["x"]+1))})
        C = self.coupled[-1]
        C.update({"w":C["f"]*(2*np.pi),"Gamma":C["gamma"]*(2*np.pi)})
        C.update({"c1":(2-self.dt*C["Gamma"])/(2+self.dt*C["Gamma"])})
        C.update({"c2":(2*self.dt)/(2+self.dt*C["Gamma"])})
        for i in range(int(ystart),int(yend)+1):
            for j in range(int(xstart),int(xend)+1):
                C["block"][i,j] = 1
                    
    def force_profile(self):
        '''
        Add a force term F(t,y) that drives the Lorentzian oscillator
        '''
        F = self.force
        for i in range(len(self.force)):
            if F[i]["type"]=="ISRS":
                # the first part is a Gaussian whole center moves at a speed of vg; the second part is the propagation decay
                self.forces[comp.Fz[i]] = np.exp(-(self.yf[i]-F[i]["y0"]-F[i]["vg"]*(self.time*self.dt-F[i]["t0"]))**2/F[i]["w_space"]**2-F[i]["alpha"]*(self.yf[i]-F[i]["y0"]))*F[i]["F0"]
            if F[i]["type"]=="DECP":
                teff = -(self.yf[i] - F[i]["y0"])/F[i]["vg"] + (self.dt*self.time-F[i]["t0"])
                new_force = (1+scipy.special.erf(teff/F[i]["trise"]))/2*np.exp(-teff/F[i]["tdecay"])*F[i]["F0"]
                new_force *= np.exp(-F[i]["alpha"]*(self.yf[i]-F[i]["y0"]))
                self.forces[comp.Fz[i]] = new_force
    
    def add_force(self, osc,F0,t0,y0,vg,w,alpha):
        '''
        Define the parameters of the ISRS force term
        '''
        cell = self.cell
        F = self.force
        F.append({"osc":osc,"F0":F0,"t0":t0,"y0":y0,"vg":vg,"w":w,"alpha":alpha,"type":"ISRS"})
        i = len(F)-1
        F[i].update({"w_space":F[i]["w"]*vg})
        self.yf[i] = np.ones((cell["y"]+1,cell["x"]+1))*y0 # y0 outside the sample
        self.yf[i] += np.ones((cell["y"]+1,cell["x"]+1))*(np.arange(cell["y"]+1)[:,np.newaxis]-y0)*self.Lorentzian[F[i]["osc"]]["block"] # y inside the sample

    def add_force_DECP(self, osc,F0,t0,y0,vg,trise,tdecay,alpha):
        '''
        Define the parameters of the DECP force term
        '''
        cell.self.cell
        F = self.force
        F.append({"osc":osc,"F0":F0,"t0":t0,"y0":y0,"vg":vg,"trise":trise,"tdecay":tdecay,
                   "alpha":alpha,"type":"DECP"})
        i = len(F)-1
        self.yf[i] = np.ones((cell["y"]+1,cell["x"]+1))*y0 # y0 outside the sample
        self.yf[i] += np.ones((cell["y"]+1,cell["x"]+1))*(np.arange(cell["y"]+1)[:,np.newaxis]-y0)*self.Lorentzian[F[i]["osc"]]["block"] # y inside the sample

    def add_fparam(self,osc,F0,t0,y0,vg,trise,tdecay,alpha,param):
        '''
        Define the parameters of a changing Lorentzian oscillator parameter (one of "b1E","w","Gamma")
        '''
        Fp = self.fparam
        Fp.append({"osc":osc,"F0":F0,"t0":t0,"y0":y0,"vg":vg,"trise":trise,"tdecay":tdecay,
                   "alpha":alpha,"param":param})
        # "param" should be one of "b1E","w","Gamma"
        
    def param_profile(self):
        """
        Calculate the Lorentzian parameter as a function of space and time
        """
        Fp = self.fparam
        for i in range(len(Fp)):
            teff = -(self.ys - Fp[i]["y0"])/Fp[i]["vg"] + (self.dt*self.time-Fp[i]["t0"])
            self.forces[comp.paramz[i]] = (1+scipy.special.erf(teff/Fp[i]["trise"]))/2*np.exp(-teff/Fp[i]["tdecay"])*Fp[i]["F0"]
            self.forces[comp.paramz[i]] *= np.exp(-Fp[i]["alpha"]*(self.ys-Fp[i]["y0"]))
    
    def update_Lorentzian(self):
        '''
        Update the Lorentzain model based  on param_profile
        '''
        Fp = self.fparam
        L = self.Lorentzian
        L_p = self.Lorentzian_pristine
        for i in range(len(Fp)):
                osc = Fp[i]["osc"]
                L[osc][Fp[i]["param"]] = L_p[osc][Fp[i]["param"]]*(1+self.forces[comp.paramz[i]])
                if Fp[i]["param"] == "Gamma":
                    L[osc].update({"c1":(2-self.dt*L[osc]["Gamma"])/(2+self.dt*L[osc]["Gamma"])})
                    L[osc].update({"c2":(2*self.dt)/(2+self.dt*L[osc]["Gamma"])})
    
    def get_sides(self):
        """
        Handle boundaries
        """
        side_L = self.fields[comp.By,:,1]
        side_B = self.fields[comp.Bx,1]
        side_R = self.fields[comp.Ez,:,-2]
        side_T = self.fields[comp.Ez,-2]
        return (side_L,side_R,side_B,side_T)
    
    def step(self):
        """
        Simulating one time step in FDTD
        """
        self.curl_E()
        if self.source["tstart"]<self.time*self.dt<self.source["tend"]:
            Jz = self.source["block"]*self.source["func"](self.time*self.dt)
        else:
            Jz = 0
        (side_L,side_R,side_B,side_T) = self.get_sides()
        self.fields[comp.Bx,1:] += (-self.curls[comp.curlEzx]*self.dt+Jz)[1:]
        self.fields[comp.Bx,0] = self.Mur_r*(self.fields[comp.Bx,0]-self.fields[comp.Bx,1])+side_B # Mur ABC at the bottom
        self.fields[comp.By] += (-self.curls[comp.curlEzy]*self.dt)
        L = self.Lorentzian
        F = self.force
        Lnum = len(L)
        Fnum = len(F)
        self.param_profile()
        self.update_Lorentzian()
        for i in range(Lnum):
            self.fields[comp.Jpz[i]] = L[i]["c1"]*self.fields[comp.Jpz[i]]
            self.fields[comp.Jpz[i]] += L[i]["c2"]*(L[i]["b1E"]*self.fields[comp.Ez]-L[i]["w"]**2*self.fields[comp.Pz[i]])*L[i]["block"]
        self.force_profile()
        for j in range(Fnum):
            i = F[j]["osc"]
            self.fields[comp.Jpz[i]] += L[i]["c2"]*(self.forces[comp.Fz[j]])*L[i]["block"]
        self.curl_B()
        self.fields[comp.Ez,:-1] += (self.curls[comp.curlB_z]*self.dt/self.epsilon)[:-1]
        for i in range(Lnum):
            self.fields[comp.Ez,:-1] += ((-self.fields[comp.Jpz[i]]*L[i]["bE1"]*self.dt)/self.epsilon)[:-1]*L[i]["block"][:-1]
         # Mur ABC at the top
        self.fields[comp.Ez,-1] = self.Mur_r*(self.fields[comp.Ez,-1]-self.fields[comp.Ez,-2])+side_T
        for i in range(Lnum):
            self.fields[comp.Pz[i]] += self.fields[comp.Jpz[i]]*self.dt
        self.fields[comp.Dz] = self.fields[comp.Ez]*self.epsilon
        self.time += 1
    
    def source_block(self,source_func,tstart,tend,xstart,xend,ystart,yend,integrated=True):
        """
        Add a source in the simulation
        """
        cell = self.cell
        self.source = {"block":np.zeros((cell["y"]+1,cell["x"]+1)),"func":source_func,
                       "tstart":tstart,"tend":tend,"inte":integrated}
        for i in range(int(ystart),int(yend)+1):
            for j in range(int(xstart),int(xend)+1):
                self.source["block"][i,j] = 1
                
    def add_epsilon_block(self,epsilon,xstart,xend,ystart,yend):
        """
        Add a constant epsilon block to the simulation
        """
        for i in range(int(ystart),int(yend)+1):
            for j in range(int(xstart),int(xend)+1):
                self.epsilon[i,j] = epsilon