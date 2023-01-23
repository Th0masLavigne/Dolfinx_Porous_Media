from Double_poro_functions_FE import *

# Thomas Lavigne
# 27/09/2022
#
# NonLinear Consolidation Problem
# Hyper-elastic solid scaffold
# Sample width= Width
# Sample Height= Height
#
#
# Csv results are stored in the folder Plotting.
# The objective is to identify the best parameters equivalent to those proposed in:
# http://dx.doi.org/10.1016/j.jmps.2016.03.005
#
# Boundary conditions:
#			   p=0
#			--------
#			|      |
#			|      |
#			|      |
#			|      |
#			|      |
#			|      |
#	    ux=0|      |ux=0
#			|      |
#			|      |
#			|      |
#			|      |
#			|      |
#			|      |
#			--------
#			  uy=0
#
# 
#
###################################################
#################### Libraries ####################
###################################################
# 
import time
import os
import glob
import shutil
import csv
from mpi4py            import MPI
import numpy as np
# 
###################################################
################### Application ###################
###################################################
# 
## Create the output directories and files
# Output directory
directory = "Results"
# Parent Directory path
parent_dir = "./"
# Path
path       = os.path.join(parent_dir, directory)
try:
	os.mkdir(path)
except:
	pass
# 
## Geometrical definition
Height= 1e-4 #[m]
Width = 1e-5 #[m]
Length = 1e-5 #[m]
## Discretization
nx, ny, nz = 2, 2, 40
# 
###################################################
#### Parameters : time and load and material ######
###################################################
# 
t_ramp = 5   
t_sust = 125.
## Time parametrization
t_init    = 0                # Start time
t_refine  = 2*t_ramp
Tf        = t_ramp+t_sust    # End time
dt_refine = 0.1
dt_coarse = 0.1
t=t_init
num_steps = int((t_refine-t_init)/dt_refine)+int((Tf-t_refine)/dt_coarse) 
# 
## Material parameters
Er            = 5000  
nur           = 0.2  
permeabilityr = 1e-14 
mu_lr         = 1
alphar        = 1.0
porosityr     = 0.5
# 
ksr = 1e10
kfr = 2.2e9
# Vessels data
Comp_br   = 1000		    	#compressibility of the vessels
mu_br     = 0.004    			#dynamic mu_l of the blood
# 
# total number of steps
time_param = t, t_init, dt_refine, Tf, num_steps, t_ramp
# 
## Mechanical loading 
pinit = 200 #[Pa]
# 
###################################################
#################### FEM-processing ###############
###################################################
# 
# Evaluate all the constitutive laws. Compute the required time for each
# atol, rtol, maxiter
solver_options = 5e-10, 1e-11, 50
# 
# 0%
poro_b_0r =	1e-10   		#initial porosity of vascular part
k_br      = 2e-16			#intrinsic permeability of vessels
# Young's modulus, Poisson ratio, permeability, dynamic viscosity, porosity, vessel bulk modulus, blood dynamic viscosity, vascular permeability, vascular porosity, soil grain bulk, solid bulk
material = Er,nur,permeabilityr,mu_lr,porosityr,Comp_br,mu_br,k_br,poro_b_0r,ksr,kfr
# Set time counter
begin_t = time.time()
pressure_y_0, disp_init_y0, poro_b_H0, pressure_y_0_blood = FE_solving([Length,Width,Height],[nx,ny, nz], material, teff_Elastic, time_param, pinit, solver_options,'epsb0','rows0.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# 2%, 
poro_b_0r = 0.02	 	#initial porosity of vascular part
k_br      = 2e-16			#intrinsic permeability of vessels
# Young's modulus, Poisson ratio, permeability, dynamic viscosity, porosity, vessel bulk modulus, blood dynamic viscosity, vascular permeability, vascular porosity
material = Er,nur,permeabilityr,mu_lr,porosityr,Comp_br,mu_br,k_br,poro_b_0r,ksr,kfr
# Set time counter
begin_t = time.time()
pressure_y_0, disp_init_y0, poro_b_H0, pressure_y_0_blood = FE_solving([Length,Width,Height],[nx,ny, nz], material, teff_Elastic, time_param, pinit, solver_options,'epsb2','rows2.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# 4%, 
poro_b_0r = 0.04        #initial porosity of vascular part
k_br      = 4e-16			#intrinsic permeability of vessels
# Young's modulus, Poisson ratio, permeability, dynamic viscosity, porosity, vessel bulk modulus, blood dynamic viscosity, vascular permeability, vascular porosity
material = Er,nur,permeabilityr,mu_lr,porosityr,Comp_br,mu_br,k_br,poro_b_0r,ksr,kfr
# Set time counter
begin_t = time.time()
pressure_y_0, disp_init_y0, poro_b_H0, pressure_y_0_blood = FE_solving([Length,Width,Height],[nx,ny, nz], material, teff_Elastic, time_param, pinit, solver_options,'epsb4','rows4.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
###################################################
################### Post-processing ###############
###################################################
if MPI.COMM_WORLD.rank == 0:
	timelist = np.asarray([np.round(element,1) for element in np.linspace(t_init,Tf,num_steps+1)])
	plot_all('rows0.csv','rows2.csv','rows4.csv',timelist)
