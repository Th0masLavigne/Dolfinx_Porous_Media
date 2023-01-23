import numpy as np
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
def epsilon(u):
    return sym(grad(u))
#
def teff(u):
    return lambda_m * nabla_div(u) * Identity(u.geometric_dimension()) + 2*mu*epsilon(u)
#
kmax=1e3
def terzaghi_p(x):
	p0,L=pinit,Height
	cv = permeability.value/viscosity.value*(lambda_m.value+2*mu.value)
	pression=0
	for k in range(1,int(kmax)):
		pression+=p0*4/np.pi*(-1)**(k-1)/(2*k-1)*np.cos((2*k-1)*0.5*np.pi*(x[1]/L))*np.exp(-(2*k-1)**2*0.25*np.pi**2*cv*t/L**2)
	pl=pression
	return pl
#
def L2_error_p(mesh,pressure_element,__p):
	V2 = FunctionSpace(mesh, pressure_element)
	pex = Function(V2)
	pex.interpolate(terzaghi_p)
	L2_errorp, L2_normp = form(inner(__p - pex, __p - pex) * dx), form(inner(pex, pex) * dx)
	error_localp = assemble_scalar(L2_errorp)/assemble_scalar(L2_normp)
	error_L2p = np.sqrt(mesh.comm.allreduce(error_localp, op=MPI.SUM))
	return error_L2p
#
## Create the domain / mesh
Height = 1e-4 #[m]
Width  = 1e-5 #[m]
Length = 1e-5 #[m]
mesh   = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[Length, Width, Height]]), [8, 8, 20], cell_type=CellType.tetrahedron)
#
## Define the boundaries:
# 1 = bottom, 2 = right, 3=top, 4=left, 5=back, 6=front
boundaries = [(1, lambda x: np.isclose(x[2], 0)),
              (2, lambda x: np.isclose(x[0], Length)),
              (3, lambda x: np.isclose(x[2], Height)),
              (4, lambda x: np.isclose(x[0], 0)),
              (5, lambda x: np.isclose(x[1], Width)),
              (6, lambda x: np.isclose(x[1], 0))]
#
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
#
## Time parametrization
t         = 0                # Start time
Tf        = 6                # End time
num_steps = 1000             # Number of time steps
dt        = (Tf-t)/num_steps # Time step size
#
## Material parameters
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
ds = Measure("ds", domain=mesh, subdomain_data=facet_tag)
normal = FacetNormal(mesh)
#
# Define Mixed Space (R2,R) -> (u,p)
displacement_element  = VectorElement("CG", mesh.ufl_cell(), 2)
pressure_element      = FiniteElement("CG", mesh.ufl_cell(), 1)
MS                    = FunctionSpace(mesh, MixedElement([displacement_element,pressure_element]))
#
# Define the Dirichlet condition
# 1 = bottom: uy=0, 2 = right: ux=0, 3=top: pl=0 drainage, 4=left: ux=0
bcs    = []
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
# drainage p=0
facets = facet_tag.find(3)
dofs   = locate_dofs_topological(MS.sub(1), fdim, facets)
bcs.append(dirichletbc(ScalarType(0), dofs, MS.sub(1)))
#
# Create the initial and solution spaces
X0 = Function(MS)
Xn = Function(MS)
#
# Initial values
# 
Un_, Un_to_MS = MS.sub(0).collapse()
FUn_ = Function(Un_)
with FUn_.vector.localForm() as initial_local:
	initial_local.set(ScalarType(0.0)) 
#
# Update Xn
Xn.x.array[Un_to_MS] = FUn_.x.array
Xn.x.scatter_forward()
#
Pn_, Pn_to_MS = MS.sub(1).collapse()
FPn_ = Function(Pn_)
with FPn_.vector.localForm() as initial_local:
	initial_local.set(ScalarType(pinit)) 
#
# Update Xn
Xn.x.array[Pn_to_MS] = FPn_.x.array
Xn.x.scatter_forward()
#
# Variational form
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
# set up the non-linear solver
solver  = nls.petsc.NewtonSolver(mesh.comm, Problem)
# Absolute tolerance
solver.atol = 5e-10
# relative tolerance
solver.rtol = 1e-11
solver.convergence_criterion = "incremental"
#
t = 0
L2_p = np.zeros(num_steps, dtype=PETSc.ScalarType)
for n in range(num_steps):
	t += dt
	num_its, converged = solver.solve(X0)
	X0.x.scatter_forward()
	# Update Value
	Xn.x.array[:] = X0.x.array
	Xn.x.scatter_forward()
	__u, __p = X0.split()
	# Compute L2 norm for pressure
	error_L2p     = L2_error_p(mesh,pressure_element,__p)
	L2_p[n] = error_L2p
	# Solve tracking
	if mesh.comm.rank == 0:
		print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}, L2-error p {error_L2p:.2e}")    
if mesh.comm.rank == 0:
	print(f"L2 error p, min {np.min(L2_p):.2e}, mean {np.mean(L2_p):.2e}, max {np.max(L2_p):.2e}, std {np.std(L2_p):.2e}")
