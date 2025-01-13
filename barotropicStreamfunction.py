from __future__ import absolute_import, division, print_function, \
    unicode_literals
import xarray as xr
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from mpas_analysis.ocean.utility import compute_zmid


def compute_barotropic_streamfunction_vertex(dsMesh, ds, min_depth, max_depth):

        inner_edges, transport = _compute_transport(dsMesh, ds, min_depth, max_depth)

        nvertices = dsMesh.sizes['nVertices']
        cells_on_vertex = dsMesh.cellsOnVertex - 1
        vertices_on_edge = dsMesh.verticesOnEdge - 1
        is_boundary_cov = cells_on_vertex == -1
        boundary_vertices = np.logical_or(is_boundary_cov.isel(vertexDegree=0),
                                          is_boundary_cov.isel(vertexDegree=1))
        boundary_vertices = np.logical_or(boundary_vertices,
                                          is_boundary_cov.isel(vertexDegree=2))

        # convert from boolean mask to indices
        boundary_vertices = np.flatnonzero(boundary_vertices.values)

        n_boundary_vertices = len(boundary_vertices)
        n_inner_edges = len(inner_edges)

        indices = np.zeros((2, 2*n_inner_edges+n_boundary_vertices), dtype=int)
        data = np.zeros(2*n_inner_edges+n_boundary_vertices, dtype=float)

        # The difference between the streamfunction at vertices on an inner
        # edge should be equal to the transport
        v0 = vertices_on_edge.isel(nEdges=inner_edges, TWO=0).values
        v1 = vertices_on_edge.isel(nEdges=inner_edges, TWO=1).values

        ind = np.arange(n_inner_edges)
        indices[0, 2*ind] = ind
        indices[1, 2*ind] = v1
        data[2*ind] = 1.

        indices[0, 2*ind+1] = ind
        indices[1, 2*ind+1] = v0
        data[2*ind+1] = -1.

        # the streamfunction should be zero at all boundary vertices
        ind = np.arange(n_boundary_vertices)
        indices[0, 2*n_inner_edges + ind] = n_inner_edges + ind
        indices[1, 2*n_inner_edges + ind] = boundary_vertices
        data[2*n_inner_edges + ind] = 1.

        rhs = np.zeros(n_inner_edges+n_boundary_vertices, dtype=float)

        # convert to Sv
        ind = np.arange(n_inner_edges)
        rhs[ind] = 1e-6*np.squeeze(transport)

        ind = np.arange(n_boundary_vertices)
        rhs[n_inner_edges + ind] = 0.

        matrix = scipy.sparse.csr_matrix(
            (data, indices),
            shape=(n_inner_edges+n_boundary_vertices, nvertices))

        solution = scipy.sparse.linalg.lsqr(matrix, rhs)
        bsf_vertex = xr.DataArray(-solution[0],
                                  dims=('nVertices',))

        return bsf_vertex


def _compute_transport(dsMesh, ds, min_depth, max_depth):

    cells_on_edge = dsMesh.cellsOnEdge - 1
    inner_edges = np.logical_and(cells_on_edge.isel(TWO=0) >= 0,
                                 cells_on_edge.isel(TWO=1) >= 0)

    # convert from boolean mask to indices
    inner_edges = np.flatnonzero(inner_edges.values)

    cell0 = cells_on_edge.isel(nEdges=inner_edges, TWO=0)
    cell1 = cells_on_edge.isel(nEdges=inner_edges, TWO=1)
    n_vert_levels = ds.sizes['nVertLevels']

    vert_index = xr.DataArray.from_dict(
        {'dims': ('nVertLevels',), 'data': np.arange(n_vert_levels)})
    z_mid = compute_zmid(dsMesh.bottomDepth, dsMesh.maxLevelCell-1,
                         dsMesh.layerThickness)
    z_mid_edge = 0.5*(z_mid.isel(nCells=cell0) + z_mid.isel(nCells=cell1))
    normal_velocity = ds.timeMonthly_avg_normalVelocity.isel(nEdges=inner_edges)
    if 'timeMonthly_avg_normalGMBolusVelocity' in ds.keys():
        normal_velocity = normal_velocity + ds.timeMonthly_avg_normalGMBolusVelocity.isel(nEdges=inner_edges)
    if 'timeMonthly_avg_normalMLEvelocity' in ds.keys():
        normal_velocity = normal_velocity + ds.timeMonthly_avg_normalMLEvelocity.isel(nEdges=inner_edges)
    layer_thickness = ds.timeMonthly_avg_layerThickness
    layer_thickness_edge = 0.5*(layer_thickness.isel(nCells=cell0) +
                                layer_thickness.isel(nCells=cell1))
    mask_bottom = (vert_index < dsMesh.maxLevelCell).T
    mask_bottom_edge = 0.5*(mask_bottom.isel(nCells=cell0) +
                            mask_bottom.isel(nCells=cell1))
    masks = [mask_bottom_edge,
             z_mid_edge <= max_depth,
             z_mid_edge >= min_depth]
    for mask in masks:
        normal_velocity = normal_velocity.where(mask)
        layer_thickness_edge = layer_thickness_edge.where(mask)

    transport = dsMesh.dvEdge[inner_edges] * (layer_thickness_edge * normal_velocity).sum(dim='nVertLevels')

    return inner_edges, transport
