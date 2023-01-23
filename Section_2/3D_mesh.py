# Thomas Lavigne
# 27/09/2022
###################################################
#################### Libraries ####################
###################################################
import dolfinx
import numpy
from dolfinx.mesh      import create_box, CellType, locate_entities, meshtags
from mpi4py            import MPI
###################################################
#################### Geometry  ####################
###################################################
Length, Height, Width = 0.1, 1, 0.1 #[m]
nx, ny, nz = 2, 40, 2
mesh  = create_box(MPI.COMM_WORLD, numpy.array([[0.0,0.0,0.0],[Length, Height, Width]]), [nx, ny, nz], cell_type=CellType.hexahedron)
###################################################
#################### Marking  #####################
###################################################
## Define the boundaries of the domain:
# 1, 2, 3, 4, 5, 6 = bottom, right, top, left, back, front
boundaries = [(1, lambda x: numpy.isclose(x[1], 0)),
			  (2, lambda x: numpy.isclose(x[0], Length)),
			  (3, lambda x: numpy.isclose(x[1], Height)),
			  (4, lambda x: numpy.isclose(x[0], 0)),
			  (5, lambda x: numpy.isclose(x[2], Width)),
			  (6, lambda x: numpy.isclose(x[2], 0))]
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
