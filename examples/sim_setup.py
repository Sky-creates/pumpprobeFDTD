## Sample code for setting up a simulation with pumpprobe FDTD ##

from pumpprobeFDTD import *

# Edit the following parameters for your simulation
kind = "PP" # what kind of experiment to simulate (either "PP","Reference","Emission","Transmission")
tpps = np.linspace(1,61,300) # an np array of the pump probe time delays to simulate
probe_shift = 30 # Offset the pump and probe pulse so that they arrives earlier to make best use of simulation time

setlabel="set0"
# change setlabel for every set!!
import sys
if len(sys.argv) > 1:
    kind = sys.argv[1]

# ideal Gaussian electric field as input THz pulse
def E_gaussian(t):
    return 300*np.exp(-(t)**2/(2*(sigma)**2))
def d2E_gaussian(t):
    return -100*np.exp(-(t)**2/(2*(sigma)**2))*(-2/(2*(sigma)**2)+4*t**2/(2*(sigma)**2)**2)
def d2E_gaussian_FDTD(t):
    return kV_cm*d2E_gaussian(t/ps-1-probe_shift)*correction
def E_input(t):
    return d2E_gaussian_FDTD(t)
def zero_input(t):
    return t*0

## Simulation cell size
sample_size = 700
cell = {"x":1,"y":sample_size+200}
sample_range = (100,100+sample_size)


correction = 0.7920113294015154
b1E = 1.01/ps**2
bE1 = 1
f1 = 1.024
gamma1 = 0.03
b2E = 0.46/ps**2
bE2 = 1
f2 = 1.12
gamma2 = 0.022
sigma = 0.2
probe_E_PP = []
sample_loc = sample_size+150
n_NIR = 2.86
k_NIR = 0.075
eps_THz = 10.7
alpha_NIR = 4*np.pi*k_NIR/1.698 # 0.73 eV
F0p = -1

if kind != "PP":
    tpps = [1]
    probe_shift = 30
for tpp in tpps:
    # Simulation for all different pump probe time delays tpps
    # Simulations are performed independently. It is recommended to replace this loop with parallelization
    sim = simulation(cell,Courant=0.4)
    run = {"sample_every":0.02*ps,"time":60*ps}
    run.update({"sample_every_step":int(run["sample_every"]/sim.dt),"step":int(run["time"]/sim.dt)})
    if kind == "Emission":
        sim.source_block(zero_input,0*ps,2*ps,0,cell["x"],5,5)
        sim.add_epsilon_block(eps_THz,0,cell["x"],*sample_range)
        sim.add_Lorentzian(f1/ps,b1E,bE1,gamma1/ps,0,cell["x"],*sample_range)
        # sim.add_Lorentzian(f2/ps,b2E,bE2,gamma2/ps,0,cell["x"],*sample_range)
        sim.add_force(0,2e-3,tpp*ps,100,1,0.1*ps,alpha_NIR)
        # sim.add_force(1,0.01,tpp*ps,100,0.1,0.1*ps,0.02)
        sim.add_fparam(0,F0p,tpp*ps,100,1,0.1*ps,1.6*ps,alpha_NIR,"b1E")
    
    if kind == "Transmission":
        sim.source_block(E_input,(0+probe_shift)*ps,(2+probe_shift)*ps,0,cell["x"],5,5)
        sim.add_epsilon_block(eps_THz,0,cell["x"],*sample_range)
        sim.add_Lorentzian(f1/ps,b1E,bE1,gamma1/ps,0,cell["x"],*sample_range)
        # sim.add_Lorentzian(f2/ps,b2E,bE2,gamma2/ps,0,cell["x"],100,600)
    
    if kind == "Reference":
        sim.source_block(E_input,(0+probe_shift)*ps,(2+probe_shift)*ps,0,cell["x"],5,5)

    if kind=="PP":
        sim.source_block(E_input,(0+probe_shift)*ps,(2+probe_shift)*ps,0,cell["x"],5,5)
        sim.add_epsilon_block(eps_THz,0,cell["x"],*sample_range)
        sim.add_Lorentzian(f1/ps,b1E,bE1,gamma1/ps,0,cell["x"],*sample_range)
        # sim.add_Lorentzian(f2/ps,b2E,bE2,gamma2/ps,0,cell["x"],100,600)
        sim.add_force(0,2e-3,tpp*ps,100,1,0.1*ps,alpha_NIR)
        # sim.add_force(1,0.01,tpp*ps,100,0.1,0.1*ps,0.02)
        sim.add_fparam(0,F0p,tpp*ps,100,1,0.1*ps,1.6*ps,alpha_NIR,"b1E")
    probe_E = []
    for tstep in range(run["step"]):
        if tstep%run["sample_every_step"]==0:
            probe_E.append(sim.fields[comp.Ez][1:,int(cell["x"]/2)].copy())
        sim.step()  # simulation run
    run.update({"sample_ps":np.arange(0,len(probe_E),1)*run["sample_every_step"]*sim.dt/ps})
    probe_E = np.array(probe_E)
    if kind!="PP":
        np.savetxt(setlabel+"_results/"+kind+"_time.txt",run["sample_ps"])
        np.savetxt(setlabel+"_results/"+kind+"_probeE.txt",probe_E[:,sample_loc])
    else:
        probe_E_PP.append(probe_E[:,sample_loc])
if kind=="PP":
    np.savetxt(setlabel+"_results/"+kind+"_kernal%d_tpp.txt"%kernal,tpps)
    np.save(setlabel+"_results/"+kind+"_kernal%d_probeE"%kernal,np.array(probe_E_PP))