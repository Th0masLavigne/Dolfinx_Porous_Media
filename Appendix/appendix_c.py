# First, one need to define the points where to evaluate the solution.

import numpy as np 
num_points = 11
y_check = np.linspace(0,Height,num_points)
points_for_time = np.array([[Width/2, 0., 0.], [Width/2, Height/2, 0.]])
points_for_space = np.zeros((num_points,3))
for ii in range(num_points):
	points_for_space[ii,0]=Width/2
	points_for_space[ii,1]=y_check[ii]
points = np.concatenate((points_for_time,points_for_space))

# The following step is to identify the cells contributing to the points. 
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
tree = BoundingBoxTree(mesh, mesh.geometry.dim)
cell_candidates = compute_collisions(tree, points)
colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
# Here is an example to select cells contributing to the first and second points.
cells_y_0 = colliding_cells.links(0)
cells_y_H_over_2 = colliding_cells.links(1)
# Knowing the shape of the functions to evaluate, lists are created and will be updated during the resolution procedure. 
# Regarding parallel computation, these lists are only created on the first kernel.
from mpi4py            import MPI
if MPI.COMM_WORLD.rank == 0:
	pressure_y_0 = np.zeros(num_steps, dtype=PETSc.ScalarType)
	pressure_y_Height_over_2 = np.zeros(num_steps, dtype=PETSc.ScalarType)
	pressure_space0 = np.zeros(num_points, dtype=PETSc.ScalarType)
	pressure_space1 = np.zeros(num_points, dtype=PETSc.ScalarType)
	pressure_space2 = np.zeros(num_points, dtype=PETSc.ScalarType)
#A function is created to evaluate a function given the mesh, the function, the 
# contributing cells to the point and the list with its index to store the evaluated value in.
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
#
#
#Finally, the problem is solved for each time steps. The functions are evaluated for all kernels and gathered on the first one 
# where the first pressure found by the different processors will be uploaded in the here-above lists. 
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
	xdmf.write_function(__u,t)
	xdmf.write_function(__p,t)
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
	if n == n0:
		for ii in range(num_points):
			evaluate_point(mesh, __p, colliding_cells.links(ii+2), points[ii+2], pressure_space0, ii)
		t0 = t
	elif n==n1:
			evaluate_point(mesh, __p, colliding_cells.links(ii+2), points[ii+2], pressure_space1, ii)
		t1 = t
	elif n==n2:
			evaluate_point(mesh, __p, colliding_cells.links(ii+2), points[ii+2], pressure_space2, ii)
		t2 = t
# 
xdmf.close()
