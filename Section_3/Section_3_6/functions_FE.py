###################################################
#################### Functions ####################
###################################################
# 
def teff_Elastic(u,lambda_m,mu):
	"""
	Compute the cauchy stress tensor from the displacement and lame coefficients
	Inputs:  
	- lambda_m, mu : Lame_Coefficients
	- u : displacement
	Outputs: 
	- lambda_m * nabla_div(u) * Identity(len(u)) + 2*mu*sym(grad(u)) : stress tensor 
	"""
	from ufl import sym, grad, nabla_div, Identity
	## Deformation
	epsilon = sym(grad(u))
	return lambda_m * nabla_div(u) * Identity(u.geometric_dimension()) + 2*mu*epsilon
# 
def deviatoric_part(tensor,u):
	"""
	Compute the deviatoric part of a 2D/3D tensor tensor-1/len(u)*tr(tensor)*Identity(len(u))
	Inputs:
	- tensor: tensor which we are interested in
	- u: displacement
	Outputs:
	- dev(tensor)
	"""
	from ufl import variable, Identity, tr
	return variable(tensor-1/len(u)*tr(tensor)*Identity(len(u)))
# 
def teff_NH_Ibar(u,lambda_m,mu):
	"""
	Compute the stress from the generalized nearly-incompressible definition of abaqus: W = C10*(J^-2/3 I1 - 3) + 1/D1 (J-1)^2
	Please note that such fromulation is only defined for a 3D case.
	(Treloar,  L. R. G., The Physics of Rubber Elasticity, Clarendon Press, Oxford, Third Edition, 1975)  
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- diff(W,F): First Piola-Kirchoff stress tensor 
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	from dolfinx.fem       import Constant
	from petsc4py.PETSc    import ScalarType
	bulk  = ScalarType(lambda_m.value+2/3*mu.value)
	D1abq = ScalarType(2/bulk)
	C10abq = ScalarType(mu.value/2)
	## Deformation gradient
	F = variable(Identity(len(u)) + grad(u))
	J  = variable(det(F))
	##
	Fbar= variable(J**(-1/3)*F)
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	B = variable(F * F.T)
	##
	Bbar = variable(J**(-2/3)*B)
	##Invariants of deformation tensors
	I1 = variable(tr(B))
	I1bar = variable(J**(-2/3)*I1)
	# 2.10 
	W = C10abq* (I1bar - tr(Identity(u.geometric_dimension()))) + 1/D1abq* (J-1)**2
	return diff(W,F)
# 
def teff_NH_Ibar_2(u,lambda_m,mu):
	"""
	Compute the stress from the generalized nearly-incompressible definition of abaqus: W = C10*(J^-2/3 I1 - 3) + 1/D1 (J-1)^2
	Please note that such fromulation is only defined for a 3D case.
	(Treloar,  L. R. G., The Physics of Rubber Elasticity, Clarendon Press, Oxford, Third Edition, 1975)  
	This formulation is developped in Selvadurai et al (http://dx.doi.org/10.1016/j.jmps.2016.03.005) and allows to avoid the derivative
	It contains the use of the deviatoric_part(tensor,u) function.
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- 2/J*mu/2*deviatoric_part(Bbar,u) - (-bulk*(J-1)*Identity(len(u))): cauchy stress tensor 
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	from dolfinx.fem       import Constant
	from petsc4py.PETSc    import ScalarType
	bulk  = ScalarType(lambda_m.value+2/3*mu.value)
	D1abq = ScalarType(2/bulk)
	C10abq = ScalarType(mu.value/2)
	## Deformation gradient
	F = variable(Identity(len(u)) + grad(u))
	J  = variable(det(F))
	##
	Fbar= variable(J**(-1/3)*F)
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	B = variable(F * F.T)
	##
	Bbar = variable(J**(-2/3)*B)
	##Invariants of deformation tensors
	I1 = variable(tr(B))
	I1bar = variable(J**(-2/3)*I1)
	# 2.10 
	sprime = 2/J*mu/2*deviatoric_part(Bbar,u)
	pprime = -bulk*(J-1)*Identity(len(u))
	sigma = variable(sprime - pprime)
	return sigma
# 
def teff_NH_UJ1(u,lambda_m,mu):
	"""
	Compute the stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (ln(J))^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- diff(W, F): First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	F = variable(Identity(len(u)) + grad(u))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	W = (mu / 2) * (Ic - tr(Identity(u.geometric_dimension()))) - mu * ln(J) + (lambda_m / 2) * (ln(J))**2
	return diff(W, F)
# 
def teff_NH_UJ2(u,lambda_m,mu):
	"""
	Compute the stress from a compressible neo-Hookean formulation: W = (mu / 2) * (Ic - tr(Identity(len(u)))) - mu * ln(J) + (lambda_m / 2) * (J-1)^2
	Inputs:
	- lambda_m, mu : Lame_Coefficients
	- u : displacement 
	Outputs: 
	- diff(W, F): First Piola Kirchoff stress tensor
	"""
	from ufl import variable, Identity, grad, det, tr, ln, diff
	## Deformation gradient
	F = variable(Identity(len(u)) + grad(u))
	J  = variable(det(F))
	## Right Cauchy-Green tensor
	C = variable(F.T * F)
	##Invariants of deformation tensors
	Ic = variable(tr(C))
	W = (mu / 2) * (Ic - tr(Identity(u.geometric_dimension()))) - mu * ln(J) + (lambda_m / 2) * (J-1)**2
	return diff(W, F)
#  
#  
###################################################
############### FE Problem Functions ##############
###################################################
# 
def mechanical_load(t,magnitude):
	"""
	Temporal evolution function of the load
	Inputs:
	- t: current time
	- magnitude: magnitude in pascal 
	Outputs: 
	- Instantaneous load value
	"""
	return -magnitude
# 
def create_geometry(dimensions,discretization):
	"""
	Create the geometry of the problem: a column
	Inputs:
	- dimensions: list containing [Length, Width, Height]
	- discretization: lists containing [nx, ny, nz]
	- timeref: list of referent times
	Outputs: 
	- mesh, dh, facet_tag:  mesh, maximum element size, all defined tags for boundary conditions
	"""
	## libraries
	import dolfinx
	import numpy
	from dolfinx.mesh      import create_box, CellType, locate_entities, meshtags
	from mpi4py            import MPI
	## Mesh generation
	mesh  = create_box(MPI.COMM_WORLD, numpy.array([[0.0,0.0,0.0],dimensions]), discretization, cell_type=CellType.hexahedron)
	## Useful checks (for convergence) 
	# Maximum element size
	tdim      = mesh.topology.dim
	num_cells = mesh.topology.index_map(tdim).size_local
	h         = dolfinx.cpp.mesh.h(mesh, tdim, range(num_cells))
	dh        = h.max()
	## Define the boundaries of the domain:
	# 1 = bottom, 2 = right, 3=top, 4=left, 5=back, 6=front
	boundaries = [(1, lambda x: numpy.isclose(x[2], 0)),
				  (2, lambda x: numpy.isclose(x[0], dimensions[0])),
				  (3, lambda x: numpy.isclose(x[2], dimensions[2])),
				  (4, lambda x: numpy.isclose(x[0], 0)),
				  (5, lambda x: numpy.isclose(x[1], dimensions[1])),
				  (6, lambda x: numpy.isclose(x[1], 0))]
	facet_indices, facet_markers = [], []
	fdim = mesh.topology.dim - 1
	for (marker, locator) in boundaries:
	    facets = locate_entities(mesh, fdim, locator)
	    facet_indices.append(facets)
	    facet_markers.append(numpy.full_like(facets, marker))
	facet_indices = numpy.hstack(facet_indices).astype(numpy.int32)
	facet_markers = numpy.hstack(facet_markers).astype(numpy.int32)
	sorted_facets = numpy.argsort(facet_indices)
	facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
	return mesh, dh, facet_tag
# 
def evaluate_point(mesh, function, contributing_cells, point, output_list, index):
	"""
	Evaluate the value of a function at a physical point
	Inputs:
	- mesh: the mesh object
	- function: the function to be evaluated
	- contributing_cells: contributing cells to the point
	- point: the physical point
	- output_list: the list to store the value in
	- index: the index in the output_list
	Outputs: 
	- None
	"""
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
#
def FE_solving(dimensions, discretization, material, constitutive_law, Time_param, load_magnitude, solver_options,XDMFNAME, filename, plot_or_not):
	"""
	Solve the FE problem for confined compression of the column on its top surface. The writing of the xdmf is currently commented.
	Calls evaluate_point(), create_geometry(), mechanical_load()
	Inputs:
	- dimensions: list containing [Length, Width, Height]
	- discretization: lists containing [nx, ny, nz]
	- material: [Young modulus Pa, Poisson ratio, IF Permeability m^2, IF dynamic viscosity Pa.s, IF Porosity, ...
	vessel compressibility Pa, Blood dynamic viscosity Pa.s, Blood permeability m^2, Vascular porosity, Soil grain bulk modulus, fluid bulk modulus]
	- constitutive_law: the constitutive law name of the function
	- Time_param: initial time s, initial time s, dt s, final time s, number of steps, ramp time s
	- load_magnitude: load magnitude Pa
	- solver_options: options for the solver [atol, rtol, maxit]
	- XDMFNAME: name of the xdmf output file (without .xdmf)
	- filename: name of the csv output file (with .csv)
	- plot_or_not: variable to plot the following of the solving
	Outputs: 
	- bottom point IF pressure, top solid displacement
	"""
	# 
	import numpy
	from dolfinx           import nls
	from dolfinx.geometry  import BoundingBoxTree, compute_collisions, compute_colliding_cells
	from dolfinx.fem.petsc import NonlinearProblem
	from ufl               import VectorElement, FiniteElement, MixedElement, TestFunctions, TrialFunction
	from ufl               import Measure, FacetNormal
	from ufl               import nabla_div, dx, dot, inner, grad, derivative, split
	from petsc4py.PETSc    import ScalarType
	from mpi4py            import MPI
	from dolfinx.fem       import (Constant, dirichletbc, Function, FunctionSpace, locate_dofs_topological)
	from dolfinx.io        import XDMFFile
	# 
	# Assign the inputs in the variables
	t, t_init, dt_refine, t_refine, dt_middle, t_middle, dt_coarse, Tf, num_steps = Time_param 
	E_r,nu_r,k_r,mu_r,eps_r, ksr, kfr = material
	mesh, dh, facet_tag = create_geometry(dimensions,discretization)
	# 
	###################################################
	############ Material Definition ##################
	###################################################
	# 
	# Young's modulus [Pa]
	E          = Constant(mesh, ScalarType(E_r))
	# Poisson's ratio [-]
	nu         = Constant(mesh, ScalarType(nu_r))
	# Bulk modulus [Pa]
	K             = Constant(mesh, ScalarType(E.value/(3*(1-2*nu.value))))
	# Lame Coefficients [Pa]
	# mu           = Constant(mesh, ScalarType(E.value/(2*(1+nu.value))))
	mu           = Constant(mesh, ScalarType(E.value/(2*(1+nu.value))))
	# lambda_m     = Constant(mesh, ScalarType(E.value*nu.value/((1+nu.value)*(1-2*nu.value))))
	lambda_m     = Constant(mesh, ScalarType(K.value-2/3*mu.value))
	# Intrinsic permeability [m^2]
	permeability = Constant(mesh, ScalarType(k_r)) 
	# Dynamic viscosity [Pa.s]
	viscosity    = Constant(mesh, ScalarType(mu_r))  
	# Porosity []
	porosity     = Constant(mesh, ScalarType(eps_r))
	# Fluid bulk modulus [Pa]
	Kf           = Constant(mesh, ScalarType(kfr))
	# Soil grains bulk modulus [Pa]
	Ks           = Constant(mesh, ScalarType(ksr))
	# Biot's coefficient []
	beta         = Constant(mesh, ScalarType(1-K.value/Ks.value))
	# Storativity coefficient []
	S            = (porosity/Kf)+(1-porosity)/Ks
	# 
	if MPI.COMM_WORLD.rank == 0:
		print(f"""E={E.value} ; nu= {nu.value} ; Biot coefficient = {beta.value}; porosity = {porosity.value}; permeability = {permeability.value}""")
	#
	###################################################
	########## Mixed Space of resolution ##############
	###################################################
	# 
	# Define Mixed Space MS=(R2,R) -> (u,p)
	displacement_element  = VectorElement("CG", mesh.ufl_cell(), 2)
	pressure_element      = FiniteElement("CG", mesh.ufl_cell(), 1)
	MS                    = FunctionSpace(mesh, MixedElement([displacement_element,pressure_element]))
	# Create the solution and previous step functions of space
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
	###################################################
	############## Initial Values #####################
	###################################################
	# 
	# U0
	Un_, Un_to_MS = MS.sub(0).collapse()
	FUn_ = Function(Un_)
	with FUn_.vector.localForm() as initial_local:
		initial_local.set(ScalarType(0.0)) 
	# Assign in Xn
	Xn.x.array[Un_to_MS] = FUn_.x.array
	Xn.x.scatter_forward()
	# P0
	Pn_, Pn_to_MS = MS.sub(1).collapse()
	FPn_ = Function(Pn_)
	with FPn_.vector.localForm() as initial_local:
		initial_local.set(ScalarType(-mechanical_load(t_init,load_magnitude))) 
	# Assign in Xn
	Xn.x.array[Pn_to_MS] = FPn_.x.array
	Xn.x.scatter_forward()
	#  
	###################################################
	############## Variationnal form ##################
	###################################################
	# 
	dt        = Constant(mesh, ScalarType(dt_refine)) # initial time step
	# surfacic load variable (Neumann BCs)
	T     = Constant(mesh,ScalarType(mechanical_load(t_init,load_magnitude)))
	if MPI.COMM_WORLD.rank == 0:
		print(f'Total number of steps: {num_steps}')
		print(f'Applied load magnitude: {T.value} Pa')
	# Create the integral surfacic element
	ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)	
	dx = Measure("dx", metadata={"quadrature_degree": 4})
	# compute the mesh normals to express t^imposed = T.normal
	normal = FacetNormal(mesh)
	# Identify the unknowns from the function
	u,p    =split(X0)
	u_n,p_n=split(Xn)
	# Define the test functions
	v,q = TestFunctions(MS)
	# Equation 17
	F  = (1/dt)*nabla_div(u-u_n)*q*dx + (permeability/viscosity)*dot(grad(p),grad(q))*dx  + ( S/dt )*(p-p_n)*q*dx
	# Equation 18
	F += inner(grad(v),constitutive_law(u,lambda_m,mu))*dx - beta * p * nabla_div(v)*dx - T*inner(v,normal)*ds(3)
	# 
	###################################################
	######################## Solver ###################
	###################################################
	# 
	# Tune the solver
	# Non linear problem definition
	dX0     = TrialFunction(MS)
	J       = derivative(F, X0, dX0)
	Problem = NonlinearProblem(F, X0, bcs = bcs, J = J)
	# Non-linear solver
	solver  = nls.petsc.NewtonSolver(mesh.comm, Problem)
	# Absolute tolerance
	solver.atol = solver_options[0]
	# relative tolerance
	solver.rtol = solver_options[1]
	solver.max_it = solver_options[2]
	solver.convergence_criterion = "incremental"
	# 
	###################################################
	################ Pre-processing ###################
	###################################################
	# 
	# Create an output xdmf file to store the values
	# xdmf = XDMFFile(mesh.comm, "./Results/"+XDMFNAME+'.xdmf', "w")
	# xdmf.write_mesh(mesh)
	# Identify the contributory cells to a top point (displacement) and a bottom point (pressure)
	# Create an array of the points to be estimated (pressure, displacement)
	points = numpy.array([[dimensions[0]/2, dimensions[1]/2, 0.], [dimensions[0]/2, dimensions[1]/2, dimensions[2]]])
	# Identify the contributing cells for each point
	tree               = BoundingBoxTree(mesh, mesh.geometry.dim)
	cell_candidates    = compute_collisions(tree, points)
	colliding_cells    = compute_colliding_cells(mesh, cell_candidates, points)
	cells_pressure     = colliding_cells.links(0)
	cells_displacement = colliding_cells.links(1)
	# Create the lists for evaluation
	pressure_z_0          = numpy.zeros(num_steps+1, dtype=ScalarType)
	displacement_z_Height = numpy.zeros(num_steps+1, dtype=ScalarType)
	# store initial values
	__uinit, __pinit = Xn.split()
	evaluate_point(mesh, __pinit, cells_pressure, points[0], pressure_z_0, 0)
	evaluate_point(mesh, __uinit.sub(2), cells_displacement, points[1], displacement_z_Height, 0)
	# 
	###################################################
	################ Processing #######################
	###################################################
	# 
	# Enusre being at the right time step
	t  = t_init  
	dt.value = dt_refine
	timelist=[]
	timelist.append(t)
	for n in range(num_steps):
		# assign the correct value of the time step
		if t>= t_middle:
			dt.value = dt_coarse
		elif t>= t_refine:
			dt.value = dt_middle
		t += dt.value
		t = round(t,2)
		timelist.append(t)
		# update the load
		T.value = mechanical_load(t,load_magnitude)
		# Solve
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
		# Export the results
		__u.name = "Displacement"
		__p.name = "Pressure"
		# xdmf.write_function(__u,t)
		# xdmf.write_function(__p,t)
		# Evaluate the functions at given points
		evaluate_point(mesh, __p, cells_pressure, points[0], pressure_z_0, n+1)
		evaluate_point(mesh, __u.sub(2), cells_displacement, points[1], displacement_z_Height, n+1)
		# Solve tracking
		if plot_or_not:   
			if MPI.COMM_WORLD.rank == 0:
				print(f"Time step {n}, time {t}, dt: {dt.value}, Number of iterations {num_its}, Load {T.value}")  
				print(f"Time {t}, dt: {dt.value}, p {pressure_z_0[n+1]}, u_y {displacement_z_Height[n+1]}")  
	# xdmf.close()
	if MPI.COMM_WORLD.rank == 0:
		import csv
		###################################################
		################ Post-Processing ##################
		###################################################
		print("Last Pressure at 0", pressure_z_0[-1])
		print("Last displacement at H",displacement_z_Height[-1])
		rows = pressure_z_0, displacement_z_Height, timelist
		with open('./Results/'+filename, 'w', encoding='UTF8', newline='') as f:
			writer = csv.writer(f)
			# write multiple rows
			writer.writerows(rows)
	else:
		rows=None
	rows=MPI.COMM_WORLD.bcast(rows, root=0)
	return rows
# 
# 
###################################################
################ Post-Processing ##################
###################################################
def plot_data1_data2(data1,data2,filename,xlabel,ylabel):
	"""
	Plotting
	- data1: absicea data
	- data2: values for the vertical axis
	- filename: output filename + extension
	- point: the physical point
	- xlabel
	- ylabel
	Outputs: 
	- None
	"""
	import matplotlib.pyplot as plt	
	# 
	plt.rcParams.update({'font.size': 15})
	plt.rcParams.update({'legend.loc':'upper right'})
	# 
	fig1, ax1 = plt.subplots()
	ax1.plot(data1,data2,linestyle='-',linewidth=2,color='gray')
	ax1.grid(color='lightgray', linestyle=':', linewidth=0.8)
	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(ylabel)
	ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	fig1.tight_layout()
	fig1.savefig(filename)
	pass
# function plot from csvs
def plot_all():
	"""
	Plotting
	- filename0: case 0%% filename
	- filename1: case 2%% filename
	- filename2: case 4%% filename
	- timelist
	Outputs: 
	- None
	"""
	import csv
	import numpy as np
	import matplotlib.pyplot as plt
	# 
	file_y1 = open('./REF_csv/LE_disp.csv')
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
	timeLEd=[]
	dispLE=[]
	for elem in range(len(rows_float_y1)):
		timeLEd.append(rows_float_y1[elem][0])
		dispLE.append(rows_float_y1[elem][1])
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
	file_y1 = open('./REF_csv/LE_press.csv')
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
	timeLEp=[]
	pressLE=[]
	for elem in range(len(rows_float_y1)):
		timeLEp.append(rows_float_y1[elem][0])
		pressLE.append(rows_float_y1[elem][1])
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
	file_y1 = open('./Results/NH_UJ1.csv')
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float.append([float(x) for x in rows1[ii]])
	file_y1.close()
	# 
	file_y1 = open('./Results/NH_UJ2.csv')
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float2=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float2.append([float(x) for x in rows1[ii]])
	file_y1.close()
	# 
	file_y1 = open('./Results/LinearElastic.csv')
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float3=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float3.append([float(x) for x in rows1[ii]])
	file_y1.close()
	# 
	file_y1 = open('./Results/NH_Ibar.csv')
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float4=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float4.append([float(x) for x in rows1[ii]])
	file_y1.close()
	# 
	file_y1 = open('./Results/NH_Ibar_2.csv')
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float5=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float5.append([float(x) for x in rows1[ii]])
	file_y1.close()
	# 
	plt.rcParams.update({'font.size': 15})
	plt.rcParams.update({'legend.loc':'upper right'})
	# 
	fig1, ax1 = plt.subplots()
	ax1.plot(rows_float3[2],rows_float3[1],linestyle='-',linewidth=2,label=r'$2 \mu \varepsilon + \lambda tr(\varepsilon) I_d$',color='gray')
	ax1.plot(rows_float[2],rows_float[1],linestyle='--',linewidth=2,label=r'$\frac{\mu}{2}(I_1-3-2\log[J])+\frac{\lambda}{2} \log[J]^2$',color='cornflowerblue')
	ax1.plot(rows_float2[2],rows_float2[1],linestyle=':',linewidth=2,label=r'$\frac{\mu}{2}(I_1-3-2\log(J))+\frac{\lambda}{2} (J-1)^2$',color='navy')
	ax1.plot(rows_float4[2],rows_float4[1],linestyle='-',linewidth=2,label=r'$\frac{\mu}{2}(J^{-2/3}I_1-3)+\left(\frac{\lambda}{2}+\frac{\mu}{3}\right)*(J-1)^2$',color='salmon')
	ax1.plot(timeLEd,dispLE,linestyle='None',marker='o',label='LE [29]',markeredgewidth=0.5,markeredgecolor='black',markerfacecolor='black')
	ax1.plot(timeNHd,dispNH,linestyle='None',marker='s',markeredgewidth=0.5,label='NH [29]',color='blue')
	ax1.grid(color='lightgray', linestyle=':', linewidth=0.8)
	ax1.set_xlabel('time (s)')
	ax1.set_ylabel('Displacement (m)')
	ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	fig1.tight_layout()
	fig1.savefig('Figure_4a.jpg')
	fig2, ax2 = plt.subplots()
	ax2.plot(rows_float3[2],rows_float3[0],linestyle='-',linewidth=2,label=r'$2 \mu \varepsilon + \lambda tr(\varepsilon) I_d$',color='gray')
	ax2.plot(rows_float[2],rows_float[0],linestyle='--',linewidth=2,label=r'$\frac{\mu}{2}(I_1-3-2\log(J))+\frac{\lambda}{2} \log(J)^2$',color='cornflowerblue')
	ax2.plot(rows_float2[2],rows_float2[0],linestyle=':',linewidth=2,label=r'$\frac{\mu}{2}(I_1-3-2\log(J))+\frac{\lambda}{2} (J-1)^2$',color='navy')
	ax2.plot(rows_float4[2],rows_float4[0],linestyle='-',linewidth=2,label=r'$\frac{\mu}{2}(J^{-2/3}I_1-3)+\left(\frac{\lambda}{2}+\frac{\mu}{3}\right)*(J-1)^2$',color='salmon')
	ax2.plot(timeLEp,pressLE,linestyle='None',marker='o',markeredgewidth=0.5,label='Linear Elastic [29]',color='black')
	ax2.plot(timeNHp,pressNH,linestyle='None',marker='s',markeredgewidth=0.5,label='Neo-Hooke [29]',color='blue')
	ax2.grid(color='lightgray', linestyle=':', linewidth=0.8)
	ax2.set_xlabel('time (s)')
	ax2.set_ylabel('Pressure (Pa)')
	ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	ax2.legend(fontsize= 12)
	fig2.tight_layout()
	fig2.savefig('Figure_4b.jpg')
	pass










