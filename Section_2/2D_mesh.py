# Thomas Lavigne
# 27/09/2022
###################################################
#################### Libraries ####################
###################################################
import dolfinx
import numpy as np
from dolfinx.mesh      import create_rectangle, CellType, locate_entities, meshtags
from mpi4py            import MPI
###################################################
#################### Geometry  ####################
###################################################
Width, Height = 1e-5, 1e-4 #[m]
nx, ny        = 2, 40      #[ ]
mesh  = create_rectangle(MPI.COMM_WORLD, np.array([[0,0],[Width, Height]]), [nx,ny], cell_type=CellType.quadrilateral)
###################################################
#################### Marking  #####################
###################################################
# identifiers: 1 , 2, 3, 4 = bottom, right, top, left
boundaries = [(1, lambda x: np.isclose(x[1], 0)),
              (2, lambda x: np.isclose(x[0], Width)),
              (3, lambda x: np.isclose(x[1], Height)),
              (4, lambda x: np.isclose(x[0], 0))]
facet_indices, facet_markers = [], []
# dimension of the elements we are looking for
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
# the meshtags() function requires sorted facet_indices
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
