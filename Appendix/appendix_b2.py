## Librairies
import dolfinx
import numpy as np
from dolfinx.mesh import create_box, CellType, refine, locate_entities, meshtags
from dolfinx.io   import XDMFFile
from mpi4py       import MPI
#
## Box 
# Dimensions of the sample
[Length, Width, Height!] = [6e-4, 2.5e-4, 4e-5]
# Discretization
[nx,ny,nz] = [30,15,8]
mesh = create_box(MPI.COMM_WORLD,np.array([[0.0,0.0,0.0],[Length, Width, Height!]]), [nx,ny,nz], cell_type=CellType.tetrahedron)
def test_on_boundary(x):
	return (np.sqrt(np.power(x[0]-3e-4,2)+np.power(x[1],2))<=1.5e-4)
#
refine_boudaries = [(11, lambda x: test_on_boundary(x))]
#
for _ in np.arange(2):
	# Refinement 
	refine_indices, refine_markers = [], []
	fdim = mesh.topology.dim-2
	for (marker, locator) in refine_boudaries:
		facets = locate_entities(mesh, fdim, locator)
		refine_indices.append(facets)
		refine_markers.append(np.full_like(facets, marker))
	refine_indices = np.hstack(refine_indices).astype(np.int32)
	refine_markers = np.hstack(refine_markers).astype(np.int32)
	# indices in meshtag must be sorted
	sorted_facets_refine = np.argsort(refine_indices)
	refine_tag = meshtags(mesh, fdim, refine_indices[sorted_facets_refine], refine_markers[sorted_facets_refine])
	mesh.topology.create_entities(fdim)
	mesh = refine(mesh, refine_indices[sorted_facets_refine])
  #
  def Omega_top(x):
    return np.logical_and((x[2] == Height), (np.sqrt(np.power(x[0]-3e-4,2)+np.power(x[1],2))<=1.5e-4))
#
def Omega_loading(x):
    return np.logical_and((x[2] == Height), (np.sqrt(np.power(x[0]-3e-4,2)+np.power(x[1],2))>=1.2e-4))
#
# Create the facet tags (identify the boundaries)
# 1 = loading, 2 = top minus loading, 3 = bottom, 4 = left, 5 = right, 6 = Front, 7 = back
boundaries = [(1, lambda x: Omega_loading(x)),
              (2, lambda x: Omega_top(x)),
              (3, lambda x: np.isclose(x[2], 0.0)),
              (4, lambda x: np.isclose(x[0], 0.0)),
              (5, lambda x: np.isclose(x[0], Length)),
              (6, lambda x: np.isclose(x[1], 0.0)),
              (7, lambda x: np.isclose(x[1], Width))]
# Mark them
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
facet_tag.name = "facets"
# Write XDMF
mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
with XDMFFile(mesh.comm, "facet_tags.xdmf", "w") as xdmftag:
    xdmftag.write_mesh(mesh)
    xdmftag.write_meshtags(facet_tag)
xdmftag.close()
