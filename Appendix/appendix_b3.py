from dolfinx.io.gmshio import read_from_msh
from dolfinx.io        import XDMFFile
# set value to 0 if .xdmf, set it to 1 if .msh
mesher = 1
#
if mesher == 0:
	##########################
	##  Read XDMF mesh      ##
	##########################
	filename = "filename.xdmf"
	with XDMFFile(MPI.COMM_WORLD, filename, "r") as file:
		mesh = file.read_mesh()
		mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
		facet_tag = file.read_meshtags(mesh, "tag.name")
#
elif mesher == 1:
	##########################
	## Read gmsh  mesh      ##
	##########################
	mesh, cell_tag, facet_tag = read_from_msh("filename.msh", MPI.COMM_WORLD, 0, gdim=3)
#
else:
	print('The mesh type is wrongly defined. mesher should equal 0 for xdmf and 1 for msh files.')
	exit()
