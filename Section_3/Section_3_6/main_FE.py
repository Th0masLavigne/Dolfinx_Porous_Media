from functions_FE import *

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
###################################################
######### Import the experimental data ############
###################################################
# 
file_y1 = open('./REF_csv/NH_disp.csv')
csvreader = csv.reader(file_y1)
rows1 = []
for row in csvreader:
        rows1.append(row)
rows_float_y1=[]
for ii in range(len(rows1)):
	# convert strings to float
	rows_float_y1.append([float(x) for x in rows1[ii]])
file_y1.close()
# 
timeNHd=[]
dispNH=[]
for elem in range(len(rows_float_y1)):
	timeNHd.append(rows_float_y1[elem][0])
	dispNH.append(rows_float_y1[elem][1])
# 
file_y1 = open('./REF_csv/NH_press.csv')
csvreader = csv.reader(file_y1)
rows1 = []
for row in csvreader:
        rows1.append(row)
rows_float_y1=[]
for ii in range(len(rows1)):
	# convert strings to float
	rows_float_y1.append([float(x) for x in rows1[ii]])
file_y1.close()
# 
timeNHp=[]
pressNH=[]
for elem in range(len(rows_float_y1)):
	timeNHp.append(rows_float_y1[elem][0])
	pressNH.append(rows_float_y1[elem][1])
# 
###################################################
############# Geometrical Parameters ##############
###################################################
# 
## Geometrical definition
Height= 1 #[m]
Width = 0.1 #[m]
## Discretization
nx, ny = 2, 40
# 
###################################################
#### Parameters : time and load and material ######
###################################################
# 
## Time parametrization
# Initial and final time of the experiment
t         = 0                # Start time
Tf        = 100000           # End time
# Adaptive time steps: three phases are defined with three steps size
t_init    = t
t_refine  = 20000
t_middle  = 60000
dt_refine = 500
dt_middle = 1000
dt_coarse = 10000
# total number of steps
num_steps = int((t_refine-t_init)/dt_refine)+int((t_middle-t_refine)/dt_middle)+int((Tf-t_middle)/dt_coarse) 
time_param = t, t_init, dt_refine, t_refine, dt_middle, t_middle, dt_coarse, Tf, num_steps
# 
## Mechanical loading 
pinit = 300000 #[Pa]
# 
# Young's modulus, Poisson ratio, permeability, dynamic viscosity, porosity, Ks, Kf
material = 600000, 0.3, 4e-14, 1e-3, 0.5, 1e10, 2.2e9
# 
###################################################
#################### FEM-processing ###############
###################################################
# 
# Evaluate all the constitutive laws. Compute the required time for each
# atol, rtol, maxiter
solver_options = 5e-10, 1e-11, 50
# 
# Set time counter
begin_t = time.time()
pressure_LE, displacement_LE, timelist_LE = FE_solving([Width,Height],[nx,ny], material, teff_Elastic, time_param, pinit, solver_options,'Elastic','LinearElastic.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# Set time counter
begin_t = time.time()
pressure_Ibar, displacement_Ibar, timelist_Ibar =FE_solving([Width,Height],[nx,ny], material, teff_NH_Ibar, time_param, pinit, solver_options,'NH_Ibar','NH_Ibar.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# Set time counter
begin_t = time.time()
pressure_Ibar2, displacement_Ibar2, timelist_Ibar2 =FE_solving([Width,Height],[nx,ny], material, teff_NH_Ibar_2, time_param, pinit, solver_options,'NH_Ibar_2','NH_Ibar_2.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# Set time counter
begin_t = time.time()
pressure_UJ1, displacement_UJ1, timelist_UJ1 =FE_solving([Width,Height],[nx,ny], material, teff_NH_UJ1, time_param, pinit, solver_options,'NH_UJ1','NH_UJ1.csv',False)
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
# 
# Set time counter
begin_t = time.time()
pressure_UJ2, displacement_UJ2, timelist_UJ2 =FE_solving([Width,Height],[nx,ny], material, teff_NH_UJ2, time_param, pinit, solver_options,'NH_UJ2','NH_UJ2.csv',False)
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
	plot_all()
	plot_hyper_el()
	plot_data1_data2(timelist_UJ2,displacement_UJ2,'U2JDISP.jpg','xlabel','ylabel')
	plot_data1_data2(timelist_UJ2,pressure_UJ2,'U2JPRESS.jpg','xlabel','ylabel')
