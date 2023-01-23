###################################################
################ Constitutive laws ################
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
###################################################
############### FE Problem Functions ##############
###################################################
# 
def mechanical_load(t,t_ramp,magnitude):
	"""
	Temporal evolution function of the load
	Inputs:
	- t,t_ramp: current time, duration of the ramp in seconds
	- magnitude: magnitude in pascal 
	Outputs: 
	- Instantaneous load value
	"""
	import numpy as np
	if t < t_ramp:
		f1 = 0.5 * (1 - np.cos(np.pi*t/t_ramp))
	else:
		f1 = 1
	return -magnitude*f1
# 
def RMSE(x,xref):
	"""
	Compute the Root Mean Square Error (RMSE)
	Inputs:
	- x: numpy array to be evaluated
	- xref: referent numpy array of same size than x
	Outputs: 
	- RMSE(x, xref)
	"""
	import numpy as np
	return np.sqrt(1/len(xref)*np.sum((x-xref)**2))
# 
def NRMSE(x,xref):
	"""
	Compute the Normalized Root Mean Square Error (NRMSE)
	Inputs:
	- x: numpy array to be evaluated
	- xref: referent numpy array of same size than x
	Outputs: 
	- RMSE(x, xref)/mean(xref)
	"""
	import numpy as np
	return 1/np.mean(np.abs(xref))*np.sqrt(1/len(xref)*np.sum((x-xref)**2))
# 
def select_data(rows,timelist,timeref):
	"""
	Create lists of same length to the referent ones based on the evaluation time
	Inputs:
	- rows: list of lists composed of the L1, L2, L3, L4
	- timelist: times corresponding to the rows lists
	- timeref: list of referent times
	Outputs: 
	- L1', L2', L3', L4' evaluated on the same times as timeref
	"""
	import numpy as np
	rpl_0, ruy_0, repsb_0, rpb_0 = rows
	# Identify the required elements for RMSE computation
	pl, uy, epsb, pb   =np.zeros(len(timeref)),np.zeros(len(timeref)),np.zeros(len(timeref)),np.zeros(len(timeref))
	# 
	for k in range(len(timeref)):
		for index in range(len(timelist)):
			if timelist[index] == timeref[k]:
				pl[k]=rpl_0[index]
				uy[k]=ruy_0[index]
				pb[k]=rpb_0[index]
				epsb[k]=repsb_0[index]
				break
	pl = np.asarray(pl)
	uy = np.asarray(uy)
	pb = np.asarray(pb)
	epsb = np.asarray(epsb)
	return pl, uy, epsb, pb
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
	mesh   = create_box(MPI.COMM_WORLD, numpy.array([[0.0,0.0,0.0],dimensions]), discretization, cell_type=CellType.hexahedron)
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
	- bottom point IF pressure, top solid displacement, vascular porosity at the bottom, blood pressure at the bottom
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
	t, t_init, dt_refine, Tf, num_steps,t_ramp = Time_param 
	E_r,nu_r,k_r,mu_r,eps_r,comp_br,mu_br,k_br,eps_br, ksr, kfr = material
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
	# Vessels data
	#compressibility of the vessels [Pa]
	Comp_b   = Constant(mesh, ScalarType(comp_br))
	#dynamic viscosity of the blood [Pa s] 		    
	mu_b     = Constant(mesh, ScalarType(mu_br))
	#initial porosity of vascular part []
	poro_b_0=Constant(mesh, ScalarType(eps_br))	
	#intrinsic permeability of vessels [m2]
	k_b=Constant(mesh, ScalarType(k_br))
	# Coefficient required in the weak form			
	def beta(pl,pb):
		return poro_b_0.value*(1-2*(pl-pb)/Comp_b.value)			
	#
	###################################################
	########## Mixed Space of resolution ##############
	###################################################
	# 
	# Define Mixed Space MS=(R2,R) -> (u,p)
	displacement_element  = VectorElement("CG", mesh.ufl_cell(), 2)
	pressure_element      = FiniteElement("CG", mesh.ufl_cell(), 1)
	MS                    = FunctionSpace(mesh, MixedElement([displacement_element,pressure_element,pressure_element]))
	# Create the solution and previous step functions of space
	X0 = Function(MS)
	Xn = Function(MS)
	# Internal variables
	Poro_space       = FunctionSpace(mesh,pressure_element)
	poro_b   = Function(Poro_space)
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
	# leakage pl=pb==0
	facets = facet_tag.find(3)
	dofs   = locate_dofs_topological(MS.sub(1), fdim, facets)
	bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(1)))
	dofs   = locate_dofs_topological(MS.sub(2), fdim, facets)
	bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(2)))
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
		initial_local.set(ScalarType(-mechanical_load(t_init,t_ramp,load_magnitude))) 
	# Assign in Xn
	Xn.x.array[Pn_to_MS] = FPn_.x.array
	Xn.x.scatter_forward()
	#
	Pbn_, Pbn_to_MS = MS.sub(2).collapse()
	FPbn_ = Function(Pbn_)
	with FPbn_.vector.localForm() as initial_local:
		initial_local.set(ScalarType(0)) 
	# Update all the threads
	Xn.x.array[Pbn_to_MS] = FPbn_.x.array
	Xn.x.scatter_forward()
	#  
	#
	# other quantities (internal variables)
	with poro_b.vector.localForm() as initial_local:
		initial_local.set(ScalarType(poro_b_0.value)) 
	# Update all the threads
	poro_b.x.scatter_forward()
	poro_b.name="poro_b"
	# 
	###################################################
	############## Variationnal form ##################
	###################################################
	# 
	dt        = Constant(mesh, ScalarType(dt_refine)) # initial time step
	# surfacic load variable (Neumann BCs)
	T     = Constant(mesh,ScalarType(mechanical_load(t_ramp,t_ramp,load_magnitude)))
	if MPI.COMM_WORLD.rank == 0:
		print(f'Total number of steps: {num_steps}')
		print(f'Applied load magnitude: {T.value} Pa')
	# Create the integral surfacic element
	ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
	dx = Measure("dx", metadata={"quadrature_degree": 4})
	# compute the mesh normals to express t^imposed = T.normal
	normal = FacetNormal(mesh)
	# Identify the unknowns from the function
	u, pl, pb       = split(X0)
	u_n, pl_n, pb_n = split(Xn)
	# Define the test functions
	v, ql, qb = TestFunctions(MS)
	# 
	F = (1-poro_b)*(1/(dt.value))*nabla_div(u-u_n)*ql*dx + ( permeability/(viscosity) )*dot( grad(pl),grad(ql) )*dx - (poro_b_0/Comp_b)*( (1/(dt.value))*(pb-pb_n-pl+pl_n) )*ql*dx
	F += poro_b*(1/(dt.value))*nabla_div(u-u_n)*qb*dx + ( k_b/(mu_b) )*dot( grad(pb),grad(qb) )*dx + (poro_b_0/Comp_b)*( (1/(dt.value))*(pb-pb_n-pl+pl_n) )*qb*dx
	F += inner(grad(v),constitutive_law(u,lambda_m,mu))*dx - (1-beta(pl,pb))*pl*nabla_div(v)*dx - beta(pl,pb)*pb*nabla_div(v)*dx - T*inner(v,normal)*ds(3)
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
	pressureblood_z_0     = numpy.zeros(num_steps+1, dtype=ScalarType)
	porosity_blood_z_0    = numpy.zeros(num_steps+1, dtype=ScalarType)
	displacement_z_Height = numpy.zeros(num_steps+1, dtype=ScalarType)
	# store initial values
	__uinit, __pinit, __pbinit = Xn.split()
	evaluate_point(mesh, __pinit, cells_pressure, points[0], pressure_z_0, 0)
	evaluate_point(mesh, __pbinit, cells_pressure, points[0], pressureblood_z_0, 0)
	evaluate_point(mesh, poro_b, cells_pressure, points[0], porosity_blood_z_0, 0)
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
		t += dt.value
		t = round(t,2)
		timelist.append(t)
		# update the load
		T.value = mechanical_load(t,t_ramp,load_magnitude)
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
		# update porosity
		poro_b.x.array[:] = poro_b_0.value*(1-(1/Comp_b.value)*(X0.x.array[Pn_to_MS]-X0.x.array[Pbn_to_MS]))
		poro_b.x.scatter_forward()
		# Update Value
		Xn.x.array[:] = X0.x.array
		Xn.x.scatter_forward()
		__u, __p, __pb = X0.split()
		# Export the results
		__u.name = "Displacement"
		__p.name = " IF Pressure"
		__pb.name = "Blood Pressure"
		# xdmf.write_function(__u,t)
		# xdmf.write_function(__p,t)
		# xdmf.write_function(__pb,t)
		# xdmf.write_function(poro_b,t)
		# Evaluate the functions at given points
		evaluate_point(mesh, __p, cells_pressure, points[0], pressure_z_0, n+1)
		evaluate_point(mesh, __pb, cells_pressure, points[0], pressureblood_z_0, n+1)
		evaluate_point(mesh, poro_b, cells_pressure, points[0], porosity_blood_z_0, n+1)
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
		rows = pressure_z_0, displacement_z_Height, porosity_blood_z_0, pressureblood_z_0
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
#
def plot_all(filename0,filename1,filename2,timelist):
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
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	# 
	###########################################
	################ Load #####################
	###########################################
	# time / Load
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVLOAD_TIME', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	load          = data[0:,1]
	time_load     = data[0:,0]
	# 
	###########################################
	################ Vascular #################
	###########################################
	# time / vascular porosity, case == 2%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVEPSBTIME_CASE2.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	time          = data[0:,0]
	epsb_castem_2 = data[0:,1]
	# time / vascular porosity, case == 4%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVEPSBTIME_CASE3.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	epsb_castem_4 = data[0:,1]
	# time / vascular pressure, case == 2%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVPBTIME_CASE2.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	pb_castem_2 = data[0:,1]
	# time / vascular pressure, case == 4%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVPBTIME_CASE3.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	pb_castem_4 = data[0:,1]
	# 
	###########################################
	################ AVascular ################
	###########################################
	# time / avascular pressure, case == 0%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVPTIME_CASE1.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	p_castem_0 = data[0:,1]
	# time / avascular pressure, case == 2%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVPTIME_CASE2.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	p_castem_2 = data[0:,1]
	# time / avascular pressure, case == 4%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVPTIME_CASE3.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	p_castem_4 = data[0:,1]
	# 
	###########################################
	################ Displacement #############
	###########################################
	# time / solid displacement, case == 0%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVUYTIME_CASE1.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	uy_castem_0 = data[0:,1]
	# time / solid displacement, case == 2%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVUYTIME_CASE2.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	uy_castem_2 = data[0:,1]
	# time / solid displacement, case == 4%
	csv_data      = pd.read_csv('./Sciume_2020_Data/DATA_SCIUME_2020/EVUYTIME_CASE3.csv', sep=';')
	data          = np.asarray(pd.DataFrame(csv_data))
	uy_castem_4 = data[0:,1]
	# 
	timeref = time
	file_y1 = open('./Results/'+filename0)
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float.append([float(x) for x in rows1[ii]])
	file_y1.close()
	pl_0, uy_0, epsb_0, pb_0 = select_data(rows_float,timelist,timeref)

	file_y1 = open('./Results/'+filename1)
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float.append([float(x) for x in rows1[ii]])
	file_y1.close()
	pl_2, uy_2, epsb_2, pb_2 = select_data(rows_float,timelist,timeref)
	file_y1 = open('./Results/'+filename2)
	csvreader = csv.reader(file_y1)
	rows1 = []
	for row in csvreader:
	        rows1.append(row)
	rows_float=[]
	for ii in range(len(rows1)):
		# convert strings to float
		rows_float.append([float(x) for x in rows1[ii]])
	file_y1.close()
	pl_4, uy_4, epsb_4, pb_4 = select_data(rows_float,timelist,timeref)
	print(f'RMSE pl: 0% {RMSE(pl_0,p_castem_0)}; NRMSE {NRMSE(pl_0,p_castem_0)}')
	print(f'RMSE uy: 0% {RMSE(uy_0,uy_castem_0)}; NRMSE {NRMSE(uy_0,uy_castem_0)}')
	print(f'RMSE pl: 2% {RMSE(pl_2,p_castem_2)}; NRMSE {NRMSE(pl_2,p_castem_2)}')
	print(f'RMSE uy: 2% {RMSE(uy_2,uy_castem_2)}; NRMSE {NRMSE(uy_2,uy_castem_2)}')
	print(f'RMSE pb: 2% {RMSE(pb_2,pb_castem_2)}; NRMSE {NRMSE(pb_2,pb_castem_2)}')
	print(f'RMSE epsb: 2% {RMSE(epsb_2,epsb_castem_2)}; NRMSE {NRMSE(epsb_2,epsb_castem_2)}')
	print(f'RMSE pl: 4% {RMSE(pl_4,p_castem_4)}; NRMSE {NRMSE(pl_4,p_castem_4)}')
	print(f'RMSE uy: 4% {RMSE(uy_4,uy_castem_4)}; NRMSE {NRMSE(uy_4,uy_castem_4)}')
	print(f'RMSE pb: 4% {RMSE(pb_4,pb_castem_4)}; NRMSE {NRMSE(pb_4,pb_castem_4)}')
	print(f'RMSE epsb: 4% {RMSE(epsb_4,epsb_castem_4)}; NRMSE {NRMSE(epsb_4,epsb_castem_4)}')
	plt.rcParams.update({'font.size': 15})
	###########################################
	################## Display ################
	###########################################
	fig1a, ax1a = plt.subplots()
	ax1a.plot([-10, 30],[0, 0],linewidth=1,linestyle='-',color='darkgray')
	ax1a.plot([0, 0],[-20, 220],linewidth=1,linestyle='-',color='darkgray')
	ax1a.plot(time_load-5,load[0:126],linewidth=1.5,color='black')
	t=plt.text(20, 195, r'$Load$', fontsize = 12, color = 'black' )
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	ax1a.plot(time[0:126]-5,p_castem_0[0:126],label='pl, 0%',linewidth=1.5,color='forestgreen')
	ax1a.plot(time[0:126]-5,p_castem_2[0:126],label='pl, 2%',linewidth=1.5,linestyle=':',color='forestgreen')
	ax1a.plot(time[0:126]-5,p_castem_4[0:126],label='pl, 4%',linewidth=1.5,linestyle='--',color='forestgreen')
	ax1a.plot(time[0:126]-5,pb_castem_2[0:126],linestyle=':',linewidth=1.5,label='pb, 2%',color='forestgreen')
	ax1a.plot(time[0:126]-5,pb_castem_4[0:126],linestyle='--',linewidth=1.5,label='pb, 4%',color='forestgreen')
	ax1a.plot(time[0:126]-5,pl_0[0:126],label='pl, 0%',linewidth=1.5,color='cornflowerblue')
	ax1a.plot(time[0:126]-5,pl_2[0:126],label='pl, 2%',linewidth=1.5,linestyle=':',color='cornflowerblue')
	ax1a.plot(time[0:126]-5,pl_4[0:126],label='pl, 4%',linewidth=1.5,linestyle='--',color='cornflowerblue')
	ax1a.plot(time[0:126]-5,pb_2[0:126],linestyle=':',linewidth=1.5,label='pb, 2%',color='salmon')
	ax1a.plot(time[0:126]-5,pb_4[0:126],linestyle='--',linewidth=1.5,label='pb, 4%',color='salmon')
	ax1a.grid(color='lightgray', linestyle=':', linewidth=0.8)
	plt.text(7, 120, r'$p^l~at~the~bottom~points$', fontsize = 12, color = 'cornflowerblue')
	plt.text(7, 30, r'$p^b~at~the~bottom~points$', fontsize = 12, color = 'salmon')
	t=plt.text(25, 170, r'$0\%$', fontsize = 12, color = 'cornflowerblue' )
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	t=plt.text(10, 175, r'$2\%$', fontsize = 12, color = 'cornflowerblue')
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	t=plt.text(5, 157, r'$4\%$', fontsize = 12, color = 'cornflowerblue')
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	ax1a.set_xlim([-10,30])
	ax1a.set_ylim([-20,220])
	ax1a.set_xlabel('time (s)')
	ax1a.set_ylabel('Pressure (Pa)')
	fig1a.tight_layout()
	fig1a.savefig('Figure_5a.jpg')
	#
	fig1b, ax1b = plt.subplots()
	ax1b.plot([-10, 30],[0, 0],linewidth=1,linestyle='-',color='darkgray')
	ax1b.plot([0, 0],[-3e-6, 1e-6],linewidth=1,linestyle='-',color='darkgray')
	ax1b.plot([elem-5 for elem in time[0:126]],uy_castem_0[0:126],label='0%',linewidth=1.5,linestyle='-',color='forestgreen')
	ax1b.plot([elem-5 for elem in time[0:126]],uy_castem_2[0:126],label='2%',linewidth=1.5,linestyle=':',color='forestgreen')
	ax1b.plot([elem-5 for elem in time[0:126]],uy_castem_4[0:126],label='4%',linewidth=1.5,linestyle='--',color='forestgreen')
	ax1b.plot([elem-5 for elem in time[0:126]],uy_0[0:126],label='0%',linewidth=1.5,linestyle='-',color='black')
	ax1b.plot([elem-5 for elem in time[0:126]],uy_2[0:126],label='2%',linewidth=1.5,linestyle=':',color='black')
	ax1b.plot([elem-5 for elem in time[0:126]],uy_4[0:126],label='4%',linewidth=1.5,linestyle='--',color='black')
	ax1b.grid(color='lightgray', linestyle=':', linewidth=0.8)
	ax1b.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	ax1b.set_xlim([-10,30])
	ax1b.set_ylim([-3e-6,1e-6])
	ax1b.set_xlabel('time (s)')
	ax1b.set_ylabel('$u_y^s$ ('+ 'm)')
	plt.text(3, 2e-7, r'$Vertical~Displacement~of~top~points$', fontsize = 12)
	t=plt.text(10, -1e-6, r'$0\%$', fontsize = 12 )
	t=plt.text(7, -1.3e-6, r'$2\%$', fontsize = 12)
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	t=plt.text(3, -1.5e-6, r'$4\%$', fontsize = 12)
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	fig1b.tight_layout()
	fig1b.savefig('Figure_5b.jpg')
	# 
	fig1c, ax1c = plt.subplots()
	ax1c.plot([0, 0],[0, 0.045],linewidth=1,linestyle='-',color='darkgray')
	ax1c.plot(time-5,epsb_castem_2,linestyle=':',linewidth=1.5,label='\u03B5'+'$_b^0=2%$',color='forestgreen')
	ax1c.plot(time-5,epsb_castem_4,linestyle='--',linewidth=1.5,label='\u03B5'+'$_b^0=4%$',color='forestgreen')
	ax1c.plot(time-5,epsb_2,linestyle=':',linewidth=1.5,label='\u03B5'+'$_b^0=2%$',color='salmon')
	ax1c.plot(time-5,epsb_4,linestyle='--',linewidth=1.5,label='\u03B5'+'$_b^0=4%$',color='salmon')
	ax1c.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	ax1c.grid(color='lightgray', linestyle=':', linewidth=0.8)
	ax1c.set_xlabel('time (s)')
	ax1c.set_ylabel('\u03B5'+'$_b$ (-)')
	plt.text(30, 0.01, r'$Vascular~porosity~at~the~bottom~points$', fontsize = 12, color = 'salmon')
	t=plt.text(20, 0.038, r'$2\%$', fontsize = 12, color = 'salmon')
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	t=plt.text(20, 0.02, r'$4\%$', fontsize = 12, color = 'salmon')
	t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))
	ax1c.set_xlim([-10,120])
	ax1c.set_ylim([0,0.045])
	fig1c.tight_layout()
	fig1c.savefig('Figure_5c.jpg')
	# 
	x=[None, None]
	y=[None, None]
	fig1d, ax1d = plt.subplots()
	ax1d.plot(x, y,label='Load',linewidth=3,linestyle='-',color='lightgray')
	ax1d.plot(x,y,linestyle='-',linewidth=1.5,label='Sciumè et al. (2021), '+'\u03B5'+'$_b^0=0%$',color='forestgreen')
	ax1d.plot(x,y,linestyle=':',linewidth=1.5,label='Sciumè et al. (2021), '+'\u03B5'+'$_b^0=2%$',color='forestgreen')
	ax1d.plot(x,y,linestyle='--',linewidth=1.5,label='Sciumè et al. (2021), '+'\u03B5'+'$_b^0=4%$',color='forestgreen')
	ax1d.axis('off')
	fig1d.legend(loc='center', prop={'size': 12})
	fig1d.tight_layout()
	fig1d.savefig('Figure_5d.jpg')
	pass










