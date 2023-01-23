# Thomas Lavigne
# 27/09/2022
#
# NonLinear Terzaghi Problem
# Sample width= Width
# Sample Height= Height
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
#		ux=0|      |ux=0
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
import numpy as np
import time
import csv
from petsc4py          import PETSc
import dolfinx
from dolfinx           import nls
from dolfinx.io        import XDMFFile
from dolfinx.mesh      import CellType, create_box, locate_entities_boundary, locate_entities, meshtags
from dolfinx.fem       import (Constant, dirichletbc, Function, FunctionSpace, locate_dofs_topological, form, assemble_scalar)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from petsc4py.PETSc    import ScalarType
from mpi4py            import MPI
from ufl               import (FacetNormal, Identity, Measure, TestFunctions, TrialFunction, VectorElement, FiniteElement, dot, dx, inner, grad, nabla_div, div, sym, MixedElement, derivative, split)
# 
###################################################
#################### Functions ####################
###################################################
# Remark: The functions must be in the main since we 
# do not import libraries in them and some parameters
# are made global.
# 
# Compute the deformation vector
# Inputs: Displacement vector
# Outputs: Desformation vector
def epsilon(u):
    return sym(grad(u))
# 
# Compute the effective stress tensor
# Inputs: displacement vector
# Outputs: effective stress
def teff(u):
    return lambda_m * nabla_div(u) * Identity(u.geometric_dimension()) + 2*mu*epsilon(u)
# 
# Compute the Exact Terzaghi solution
# Inputs: coordinates
# Outputs: Fluid pressure 
kmax=1e3
def terzaghi_p(x):
	p0,L=pinit,Height
	cv = permeability.value/viscosity.value*(lambda_m.value+2*mu.value)
	pression=0
	for k in range(1,int(kmax)):
		pression+=p0*4/np.pi*(-1)**(k-1)/(2*k-1)*np.cos((2*k-1)*0.5*np.pi*(x[2]/L))*np.exp(-(2*k-1)**2*0.25*np.pi**2*cv*t/L**2)
	pl=pression
	return pl
# 
# Define the L2_error computation
# Inputs: Mesh, type of element, solution function
# Outputs: L2 error to the analytical solution
def L2_error_p(mesh,pressure_element,__p):
	V2 = FunctionSpace(mesh, pressure_element)
	pex = Function(V2)
	pex.interpolate(terzaghi_p)
	L2_errorp, L2_normp = form(inner(__p - pex, __p - pex) * dx), form(inner(pex, pex) * dx)
	error_localp = assemble_scalar(L2_errorp)/assemble_scalar(L2_normp)
	error_L2p = np.sqrt(mesh.comm.allreduce(error_localp, op=MPI.SUM))
	return error_L2p
# 
# Suitable Evaluations functions for Parallel computation
# Inputs: mesh, function to evaluate, contributing cells to the point, point, output list to store the value, index in the list
# Outputs: the evaluated function value is added at the index location in output list
def evaluate_point(mesh, function, contributing_cells, point, output_list, index):
	from mpi4py            import MPI
	function_eval = None
	if len(contributing_cells) > 0:
		function_eval = function.eval(point, contributing_cells[:1])
	function_eval = mesh.comm.gather(function_eval, root=0)
	# Choose first pressure that is found from the different processors
	if MPI.COMM_WORLD.rank == 0:
		for element in function_eval:
			if element is not None:
				output_list[index]=element[0]
				break
	pass
# Terzaghi analytical solution
def terzaghi(p0,L,cv,z,t,kmax):
	# y as the position, t as the time we are looking to
	# p0 the applied pressure
	# L the sample's length
	# cv the consolidation time
	pression=0
	for k in range(1,kmax):
		#print(k)
		pression+=p0*4/np.pi*(-1)**(k-1)/(2*k-1)*np.cos((2*k-1)*0.5*np.pi*(z/L))*np.exp(-(2*k-1)**2*0.25*np.pi**2*cv*t/L**2)
		#print(p0*4/np.pi*p)
	pl=pression
	return pl
###################################################
################### Application ###################
###################################################
# Compute the overall computation time
# Set time counter
begin_t = time.time()
# 
###################################################
############# Geometrical Definition ##############
###################################################
# 
## Create the domain / mesh
## Geometrical definition
Height= 1e-4 #[m]
Width = 1e-5 #[m]
Length = 1e-5 #[m]
## Discretization
nx, ny, nz = 2, 2, 40
mesh  = create_box(MPI.COMM_WORLD, np.array([[0,0,0],[Length,Width, Height]]), [nx,ny,nz], cell_type=CellType.tetrahedron)
# 
## Define the boundaries:
## Define the boundaries of the domain:
# 1 = bottom, 2 = right, 3=top, 4=left, 5=back, 6=front
boundaries = [(1, lambda x: np.isclose(x[2], 0)),
			  (2, lambda x: np.isclose(x[0], Length)),
			  (3, lambda x: np.isclose(x[2], Height)),
			  (4, lambda x: np.isclose(x[0], 0)),
			  (5, lambda x: np.isclose(x[1], Width)),
			  (6, lambda x: np.isclose(x[1], 0))]
facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
# Identify contributing cells to our points of interest
num_points = 11
z_check = np.linspace(0,Height,num_points)
tree = BoundingBoxTree(mesh, mesh.geometry.dim)
points_for_time = np.array([[Length/2, Width/2, 0.], [Length/2, Width/2, Height/2]])
points_for_space = np.zeros((num_points,3))
for ii in range(num_points):
	points_for_space[ii,0]=Length/2
	points_for_space[ii,1]=Width/2
	points_for_space[ii,2]=z_check[ii]
points = np.concatenate((points_for_time,points_for_space))
cell_candidates = compute_collisions(tree, points)
colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
cells_y_0 = colliding_cells.links(0)
cells_y_H_over_2 = colliding_cells.links(1)

###################################################
#### Parameters : time and load and material ######
###################################################
# 
## Time parametrization
t         = 0                # Start time
Tf        = 6.           # End time
num_steps = 1000             # Number of time steps
dt        = (Tf-t)/num_steps # Time step size
# 
E            = Constant(mesh, ScalarType(5000))  
nu           = Constant(mesh, ScalarType(0.4))
lambda_m     = Constant(mesh, ScalarType(E.value*nu.value/((1+nu.value)*(1-2*nu.value))))
mu           = Constant(mesh, ScalarType(E.value/(2*(1+nu.value))))
rhos         = Constant(mesh, ScalarType(1))
permeability = Constant(mesh, ScalarType(1.8e-15)) 
viscosity    = Constant(mesh, ScalarType(1e-2))  
rhol         = Constant(mesh, ScalarType(1))
beta         = Constant(mesh, ScalarType(1))
porosity     = Constant(mesh, ScalarType(0.2))
Kf           = Constant(mesh, ScalarType(2.2e9))
Ks           = Constant(mesh, ScalarType(1e10))
S            = (porosity/Kf)+(1-porosity)/Ks
# 
## Mechanical loading 
pinit = 100 #[Pa]
T     = Constant(mesh,ScalarType(-pinit))
# 
# Create the surfacic element
dx = Measure("dx", metadata={"quadrature_degree": 4})
ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
# compute the mesh normals to express t^imposed = T.normal
normal = FacetNormal(mesh)
#
###################################################
########## Mixed Space of resolution ##############
###################################################
# 
# Define Mixed Space (R2,R) -> (u,p)
displacement_element  = VectorElement("CG", mesh.ufl_cell(), 2)
pressure_element      = FiniteElement("CG", mesh.ufl_cell(), 1)
MS                    = FunctionSpace(mesh, MixedElement([displacement_element,pressure_element]))
# 
# Create the initial and solution functions of space
X0 = Function(MS)
Xn = Function(MS)
# 
###################################################
######## Dirichlet boundary condition #############
###################################################
# 
# 1 = bottom, 2 = right, 3=top, 4=left, 5=back, 6=front
bcs    = []
fdim = mesh.topology.dim - 1
# uz=0
facets = facet_tag.find(1)
dofs   = locate_dofs_topological(MS.sub(0).sub(2), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(2)))
# ux=0
facets = facet_tag.find(2)
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(0)))
# ux=0
facets = facet_tag.find(4)
dofs   = locate_dofs_topological(MS.sub(0).sub(0), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(0)))
# uy=0
facets = facet_tag.find(5)
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
# uy=0
facets = facet_tag.find(6)
dofs   = locate_dofs_topological(MS.sub(0).sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(0).sub(1)))
# leakage p=0
facets = facet_tag.find(3)
dofs   = locate_dofs_topological(MS.sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(1)))
# 
# 
###################################################
############## Initial Values #####################
###################################################
# 
Un_, Un_to_MS = MS.sub(0).collapse()
FUn_ = Function(Un_)
with FUn_.vector.localForm() as initial_local:
	initial_local.set(ScalarType(0.0)) 
# # Re assign in Xn
Xn.x.array[Un_to_MS] = FUn_.x.array
Xn.x.scatter_forward()

Pn_, Pn_to_MS = MS.sub(1).collapse()
FPn_ = Function(Pn_)
with FPn_.vector.localForm() as initial_local:
	initial_local.set(ScalarType(pinit)) 
# # Re assign in Xn
Xn.x.array[Pn_to_MS] = FPn_.x.array
Xn.x.scatter_forward()
#  
###################################################
############## Variationnal form ##################
###################################################
# 
# Identify the unknowns from the function
u,p    =split(X0)
u_n,p_n=split(Xn)
# Set up the test functions
v,q = TestFunctions(MS)
# Equation 17
F  = (1/dt)*nabla_div(u-u_n)*q*dx + (permeability/viscosity)*dot(grad(p),grad(q))*dx  + ( S/dt )*(p-p_n)*q*dx
# Equation 18
F += inner(grad(v),teff(u))*dx - beta * p * nabla_div(v)*dx - T*inner(v,normal)*ds(3)
# Non linear problem definition
dX0     = TrialFunction(MS)
J       = derivative(F, X0, dX0)
Problem = NonlinearProblem(F, X0, bcs = bcs, J = J)
#
###################################################
######################## Solver ###################
###################################################
# 
# set up the non-linear solver
solver  = nls.petsc.NewtonSolver(mesh.comm, Problem)
# Absolute tolerance
solver.atol = 5e-10
# relative tolerance
solver.rtol = 1e-11
solver.maxit = 200
solver.convergence_criterion = "incremental"
# 
###################################################
################ Pre-processing ###################
###################################################
# 
# Create an output xdmf file to store the values
# xdmf = XDMFFile(mesh.comm, "Column.xdmf", "w")
# xdmf.write_mesh(mesh)
# Create output lists in time and space for the IF pressure
pressure_z_0 = np.zeros(num_steps+1, dtype=PETSc.ScalarType)
pressure_z_Height_over_2 = np.zeros(num_steps+1, dtype=PETSc.ScalarType)
pressure_space0 = np.zeros(num_points, dtype=PETSc.ScalarType)
pressure_space1 = np.zeros(num_points, dtype=PETSc.ScalarType)
pressure_space2 = np.zeros(num_points, dtype=PETSc.ScalarType)
# initial conditions
pressure_z_0[0] = pinit
pressure_z_Height_over_2[0] = pinit
# 
###################################################
################ Processing #######################
###################################################
# time steps to evaluate the pressure in space:
n0, n1, n2 = 200,400,800
# 
t = 0
L2_p = np.zeros(num_steps, dtype=PETSc.ScalarType)
for n in range(num_steps):
	t += dt
	try:
		num_its, converged = solver.solve(X0)
	except:
		if MPI.COMM_WORLD.rank == 0:
			print("*************") 
			print("Solver failed")
			print("*************") 
			pass
	X0.x.scatter_forward()
	# Update Value
	Xn.x.array[:] = X0.x.array
	Xn.x.scatter_forward()
	__u, __p = X0.split()
	# 
	# Export the results
	__u.name = "Displacement"
	__p.name = "Pressure"
	# xdmf.write_function(__u,t)
	# xdmf.write_function(__p,t)
	# 
	# Compute L2 norm for pressure
	error_L2p     = L2_error_p(mesh,pressure_element,__p)
	L2_p[n] = error_L2p
	# 
	# Solve tracking
	if MPI.COMM_WORLD.rank == 0:
		print(f"Time step {n}/{num_steps}, Load {T.value}, L2-error p {error_L2p:.2e}") 
	# Evaluate the functions
	# in time
	evaluate_point(mesh, __p, cells_y_0, points[0], pressure_z_0, n+1)
	evaluate_point(mesh, __p, cells_y_H_over_2, points[1], pressure_z_Height_over_2, n+1)
	# in space
	if n == n0:
		for ii in range(num_points):
			evaluate_point(mesh, __p, colliding_cells.links(ii+2), points[ii+2], pressure_space0, ii)
		t0 = t
	elif n==n1:
		for ii in range(num_points):
			evaluate_point(mesh, __p, colliding_cells.links(ii+2), points[ii+2], pressure_space1, ii)
		t1 = t
	elif n==n2:
		for ii in range(num_points):
			evaluate_point(mesh, __p, colliding_cells.links(ii+2), points[ii+2], pressure_space2, ii)
		t2 = t
# 
# xdmf.close()
# 
###################################################
################ Post-Processing ##################
###################################################
# 
if MPI.COMM_WORLD.rank == 0:
	print(f"L2 error p, min {np.min(L2_p):.2e}, mean {np.mean(L2_p):.2e}, max {np.max(L2_p):.2e}, std {np.std(L2_p):.2e}")
# 
# Evaluate final time
end_t = time.time()
t_hours = int((end_t-begin_t)//3600)
tmin = int(((end_t-begin_t)%3600)//60)
tsec = int(((end_t-begin_t)%3600)%60)
if MPI.COMM_WORLD.rank == 0:
	print(f"FEM operated with {num_steps} iterations, in {t_hours} h {tmin} min {tsec} sec")
	# 
	###################################################
	################ Analytical solutions #############
	###################################################
	# 
	cv = permeability.value/viscosity.value*(lambda_m.value+2*mu.value)
	z=0
	t=np.linspace(0,Tf,num_steps+1)
	pressure4 = np.zeros(num_steps+1)
	kmax=1e3
	for i in range(num_steps+1):
		pressure4[i] = terzaghi(pinit,Height,cv,z,t[i],int(kmax))
	# 
	z=Height/2
	t=np.linspace(0,Tf,num_steps+1)
	pressure5 = np.zeros(num_steps+1)
	for i in range(num_steps+1):
		pressure5[i] = terzaghi(pinit,Height,cv,z,t[i],int(kmax))
	# 
	pressure0 = np.zeros(num_points)
	for i in range(num_points):
		pressure0[i] = terzaghi(pinit,Height,cv,z_check[i],t0,int(kmax))
	# 
	pressure1 = np.zeros(num_points)
	for i in range(num_points):
		pressure1[i] = terzaghi(pinit,Height,cv,z_check[i],t1,int(kmax))
	# 
	pressure2 = np.zeros(num_points)
	for i in range(num_points):
		pressure2[i] = terzaghi(pinit,Height,cv,z_check[i],t2,int(kmax))
	# 
	###################################################
	################ Plots ############################
	###################################################
	# 
	import matplotlib.pyplot as plt
	# 
	plt.rcParams.update({'font.size': 15})
	plt.rcParams.update({'legend.loc':'upper right'})
	# 
	fig1, ax1 = plt.subplots()
	ax1.plot(t,pressure4,linestyle='-',linewidth=2,label='Analytic y=0',color='powderblue')
	ax1.plot(t,pressure5,linestyle='-',linewidth=2,label='Analytic y=h/2',color='bisque')
	ax1.plot(t,pressure_z_0,linestyle=':',linewidth=2,label='FEniCSx y=0',color='cornflowerblue')
	ax1.plot(t,pressure_z_Height_over_2,linestyle=':',linewidth=2,label='FEniCSx y=h/2',color='salmon')
	ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	ax1.set_xlabel('time (s)')
	ax1.set_ylabel('Pressure (Pa)')
	ax1.legend()
	fig1.tight_layout()
	fig1.savefig('Figure_2a.jpg')
	# 
	fig2, ax2 = plt.subplots()
	ax2.plot(pressure0,z_check,linestyle='-',linewidth=2,color='lightgreen',label='Analytic')
	ax2.plot(pressure1,z_check,linestyle='-',linewidth=2,color='lightgreen')
	ax2.plot(pressure2,z_check,marker='',linestyle='-',linewidth=2,color='lightgreen')
	ax2.plot(pressure_space0,z_check,linestyle=':',linewidth=2,color='olivedrab',label='FEniCSx')
	ax2.plot(pressure_space1,z_check,linestyle=':',linewidth=2,color='olivedrab')
	ax2.plot(pressure_space2,z_check,linestyle=':',linewidth=2,color='olivedrab')
	ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	tt=plt.text(12, 4e-5, f't={np.round(t2,1)}s', fontsize = 12 )
	tt=plt.text(35, 4e-5,f't={np.round(t1,1)}s', fontsize = 12)
	tt.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	tt=plt.text(60, 4e-5, f't={np.round(t0,1)}s', fontsize = 12)
	tt.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	ax2.set_xlabel('Pressure (Pa)')
	ax2.set_ylabel('Height (m)')
	ax2.legend()
	fig2.tight_layout()
	fig2.savefig('Figure_2b.jpg')