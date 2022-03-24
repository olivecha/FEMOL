import numpy as np
import meshio
import meshzoo
import pygmsh
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import FEMOL.elements
from FEMOL.utils import points_area
import scipy.sparse
from scipy.interpolate import griddata


class Mesh(object):
    """
    A class representing a generic 2D mesh
    """
    point_variables = ['Ux', 'Uy', 'Uz', 'Tx', 'Ty', 'Tz']

    def __init__(self, points, cell_dict, structured=False, quad_element=None, tri_element=None):
        """
        Constructor for the general mesh class
        """
        # Empty point and cell data dict
        self.element = None
        self.rows = None
        self.wrap_factor = None
        self.point_data = {}
        self.cell_data = {}
        self.cell_centers = {}
        self.ElementClasses = {}
        self.vertices = {}
        self.all_vertices = np.empty(shape=(0, 2))
        # Store the cells and points into the Mesh instance
        self.cells = cell_dict
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((points.shape[0], 1))])
        self.points = points
        # Get if the mesh contains triangles, quads or both
        self.contains = [key for key in cell_dict if key in ['triangle', 'quad']]
        # Is the mesh structured ?
        self.structured = structured
        # If the mesh is structured, there is only one cell type and one number of elements
        if self.structured:
            self.cell_type = self.contains[0]
            self.N_ele = self.cells[self.cell_type].shape[0]
        else:
            # Compute the total number of elements
            self.N_ele = np.sum([self.cells[cell_type].shape[0] for cell_type in self.contains])

        # Define the element classes
        if tri_element:
            self.ElementClasses['triangle'] = tri_element
        else:
            self.ElementClasses['triangle'] = FEMOL.elements.T3
        if quad_element:
            self.ElementClasses['quad'] = quad_element
        else:
            self.ElementClasses['quad'] = FEMOL.elements.Q4

        if self.ElementClasses['triangle'] == FEMOL.elements.T6:
            if 'triangle' in self.contains:
                self.make_triangles_quad()

        # Clean the mesh from empty nodes
        self.clean()

        # Compute the number of nodes
        self.N_nodes = self.points.shape[0]

        # Create the plotter instance
        self.plot = MeshPlot(self)

        # Compute all the cells in an array
        all_cells = [self.cells[cell_type] for cell_type in self.contains]
        all_cells = [nodes for node_list in all_cells for nodes in node_list]
        self.all_cells = np.array(all_cells, dtype=object)

        # Compute the element centers
        self.compute_element_centers()

        # Compute the element areas
        self.areas = self.element_areas()

        # Fix the quads if applicable
        if 'quad' in self.contains:
            if (self.areas['quad'] < 0).any():
                self.fix_quadrilaterals()

    def element_areas(self):
        """
        Method to compute the areas of all the elements
        :return: None
        """
        areas = {}
        for cell_type in self.contains:
            areas[cell_type] = []
            for cell in self.cells[cell_type]:
                element = self.ElementClasses[cell_type](self.points[cell])
                areas[cell_type].append(element.area())
            areas[cell_type] = np.array(areas[cell_type])

        return areas

    def fix_quadrilaterals(self):
        """
        Method to fix the quadrilateral elements having negative area
        """
        self.cells['quad'][self.areas['quad'] < 0] = np.flip(self.cells['quad'][self.areas['quad'] < 0], axis=1)

    def display(self, backend='matplotlib', color='#D1E8FF', plot_nodes=False):
        """
        Plot the mesh using the specified backend
        Supported are :
        - 'matplotlib'
        - #TODO 'Pyvista'
        """
        if backend == 'matplotlib':
            ax = plt.gca()
            ax.set_title('Mesh')
            ax.set_aspect('equal')
            ax.set_axis_off()

            for cell_type in self.contains:
                for cell in self.cells[cell_type]:
                    cell_points = self.points[cell]
                    x = cell_points.T[0]
                    y = cell_points.T[1]
                    ax.fill(x, y, color, edgecolor='k', zorder=-1)

            if plot_nodes:
                plt.scatter(*self.points.T[:2], color='k')

    def compute_N_nodes(self):
        """
        Compute the number of nodes used by cells in the mesh
        """
        all_nodes = []
        for cell_type in self.contains:
            all_nodes.append(np.unique(self.cells[cell_type].flatten()))
        N_nodes = np.unique(np.concatenate(all_nodes)).shape[0]
        return N_nodes

    def compute_vertices(self):
        """
        Computes the vertices of the mesh
        """
        self.vertices = {}
        self.all_vertices = np.empty(shape=(0, 2))
        for cell_type in self.contains:
            vertices = []
            for cell in self.cells[cell_type]:
                for i, _ in enumerate(cell[:-1]):
                    vertices.append([cell[i], cell[i + 1]])
                vertices.append([cell[-1], cell[0]])
            self.vertices[cell_type] = np.array(vertices, dtype=int)
            self.all_vertices = np.vstack([self.all_vertices, np.array(vertices, dtype=int)])

    def capture_boundary(self, separate_subsets=False):
        """
        Finds the nodes on the boundary of the mesh
        :param separate_subsets: if true a list of separated boundaries is return
        :return: boundary nodes (list)
        """
        # compute the vertices in the mesh
        self.compute_vertices()
        # get the unique vertices
        unique_vertices, vertices_count = np.unique(np.sort(self.all_vertices), return_counts=True, axis=0)
        # find the unique vertices (only present in one element)
        boundary_vertices = unique_vertices[vertices_count == 1]
        # find the nodes a part of the boundary vertices
        nodes = boundary_vertices.flatten().astype(int)
        # if the separated subsets are required, compute them
        if separate_subsets:
            # start with the first node
            node = nodes[0]
            # empty list to store the groups
            boundary_groups = []
            # All the boundary vertices and their flipped copy
            vertices = np.vstack([boundary_vertices, np.flip(boundary_vertices, axis=1)])
            while True:
                try:
                    # First node of the group is the initial node
                    current_b_nodes = [node]
                    while True:
                        try:
                            # find the nodes close to this node
                            close_nodes = vertices[vertices[:, 0] == node][:, 1]
                            # the next node is the neighbour that is not already in the group
                            node = close_nodes[~np.isin(close_nodes, current_b_nodes)][0]
                            # add it to the group
                            current_b_nodes.append(node)
                        except IndexError:
                            # break when no more node is found
                            break
                    # store the current group
                    boundary_groups.append(np.array(current_b_nodes, dtype=int))
                    # find the next starting node
                    node = np.array(nodes)[~np.isin(nodes, np.hstack(boundary_groups))][0]
                except IndexError:
                    break

            return boundary_groups

        # if not return all the boundary nodes
        else:
            return nodes

    def clean(self):
        """
        Remove the unused nodes from the mesh
        """
        # compute the full nodes
        self.compute_full_nodes()
        # get the node map
        full_node_map = self.get_full_nodes_map()
        # Transform the node numbers
        for celltype in self.contains:
            cellsize = self.cells[celltype].shape[1]
            new_cells = full_node_map[self.cells[celltype].reshape(-1)].reshape(-1, cellsize)
            self.cells[celltype] = new_cells
        # Transform the points
        self.points = self.points[self.full_nodes]

    def compute_full_nodes(self):
        """
        Computes the nodes indexes used in cells
        """
        full_nodes = np.hstack([self.cells[cell_type].reshape(-1) for cell_type in self.contains])
        self.full_nodes = np.unique(full_nodes)

    def get_full_nodes_map(self):
        """
        Computes the map between the complete node set and the nodes used in cells
        """
        full_node_map = [0] + [0] * self.full_nodes[0].astype(int)
        for i in range(1, self.full_nodes.shape[0]):
            full_node_map += [i] * (self.full_nodes[i] - self.full_nodes[i - 1])
        full_node_map = np.array(full_node_map)
        return full_node_map.astype(int)

    def save(self, file):
        """
        Save the mesh to VTK format using meshio
        :param file: filename
        :return: None
        """
        new_cell_data = {}
        for data in self.cell_data:
            new_data = []
            for cell_type in self.contains:
                new_data.append(self.cell_data[data][cell_type])
            new_cell_data[data] = new_data

        cellkeys = ['triangle', 'quad', 'T6']
        # Only keep the cell having data
        cells = {celltype:self.cells[cell_type] for celltype in self.cells if celltype in cellkeys}

        meshio_mesh = meshio.Mesh(
            self.points,
            cells,
            # Optionally provide extra data on points, cells, etc.
            point_data=self.point_data,
            # Each item in cell data must match the cells array
            cell_data=new_cell_data)

        meshio_mesh.write(file + '.vtk')

    def domain_nodes(self, domain):
        """
        Return the nodes numbers of the nodes in the domain
        Parameters
        ----------
        domain : A callable with the signature domain(x, y)

        Returns
        -------
        nodes : The mesh nodes inside of the domain
        """
        # Get the nodes inside the domain
        nodes = np.array([domain(*coord[:2]) for coord in self.points]).nonzero()[0]
        return nodes

    def global_matrix_indexes(self, N_dof):
        """
        Method computing the global stiffness matrix indexes of the mesh
        """
        # Empty row and col indexes
        rows = []
        cols = []

        # Loop over quad and triangles
        for cell_type in self.contains:
            # create an element instance to get the indexing data
            element_points = self.points[self.cells[cell_type][0]]
            element = self.ElementClasses[cell_type](element_points, N_dof=N_dof)
            base_range = np.tile(np.arange(0, element.N_dof), element.N_nodes)
            for nodes in self.cells[cell_type]:
                base_index = base_range + np.repeat(nodes, element.N_dof) * element.N_dof
                cols.append(np.tile(base_index, element.size))
                rows.append(np.repeat(base_index, element.size))

        # if the mesh is structured we store the element as they are all the same
        if self.structured:
            self.element = element

        self.rows = np.hstack(rows)
        self.cols = np.hstack(cols)

    def dimensions(self):
        """
        Compute the general mesh dimensions to adjusts plots
        """
        x, y = self.points.T[:2]
        Lx = x.max() - x.min()
        Ly = y.max() - y.min()
        return Lx, Ly

    def element_size(self):
        """
        Computes the element size (distance) for filtering
        """
        # Exact for structured meshes
        if self.structured:
            x, y = self.points[self.cells[self.contains[0]][0]].T[:2]
            x_size = x.max() - x.min()
            y_size = y.min() - y.max()
            ele_size = np.max([x_size, y_size])

        # Approximation for unstructured mesh
        elif not self.structured:
            dx = (self.dimensions()[0] / np.sqrt(self.N_nodes))
            dy = (self.dimensions()[1] / np.sqrt(self.N_nodes))
            ele_size = np.mean([dx, dy])

        return ele_size

    def get_quality(self):
        """
        Computes the mesh quality as cell data series
        """
        mesh_quality = {}
        for cell_type in self.contains:
            mesh_quality[cell_type] = []
            for cell in self.cells[cell_type]:
                element = self.ElementClasses[cell_type](self.points[cell], 2)
                mesh_quality[cell_type].append(element.quality())
        self.cell_data['quality'] = mesh_quality

    def wrap(self, name='', factor=1):
        """
        Wrap the x, y components of the mesh according to a specified point data entry
        """
        # 2D wrapped points
        if name == '':
            wrapped_x = self.points.T[0] + factor * self.point_data['Ux']
            wrapped_y = self.points.T[1] + factor * self.point_data['Uy']
            self.wrap_data = {'x': 'Ux', 'y': 'Uy'}
        else:
            wrapped_x = self.points.T[0] + factor * self.point_data[name + '_' + 'Ux']
            wrapped_y = self.points.T[1] + factor * self.point_data[name + '_' + 'Uy']
            self.wrap_data = {'x': name + '_' + 'Ux', 'y': name + '_' + 'Uy'}
        self.wrap_factor = factor
        self.wrapped = True
        self.wrapped_points_2D = np.vstack([wrapped_x, wrapped_y]).T

    def unwrap(self):
        """
        Unwraps the mesh with the saved point data key
        """
        if self.wrapped:
            factor = self.wrap_factor
            wrapped_x = self.points.T[0] - factor * self.point_data[self.wrap_data['x']]
            wrapped_y = self.points.T[1] - factor * self.point_data[self.wrap_data['y']]
            self.wrapped_points_2D = np.vstack([wrapped_x, wrapped_y]).T
            self.wrapped = False

    def add_displacement(self, U, N_dof):
        """
        Adds a displacement result vector to the mesh as point data
        """
        self.U = U
        for dof in np.arange(N_dof):
            self.point_data[self.point_variables[dof]] = U[np.arange(dof, len(U), N_dof)]
        self.wrap()

    def stress_from_displacement(self, *tensors, which=None, N_dof=2):
        """
        Computes the stress as cell data for the mesh
        :param tensors: Stiffness tensors
        """
        if N_dof == 2:
            self.cell_data['Sx'] = {}
            self.cell_data['Sy'] = {}
            self.cell_data['Sxy'] = {}
            self.cell_data['Sv'] = {}

            for celltype in self.contains:
                S = []
                for i, cell in enumerate(self.cells[celltype]):
                    if which:
                        ux = self.point_data[which + '_Ux'][cell]
                        uy = self.point_data[which + '_Uy'][cell]
                    else:
                        ux = self.point_data['Ux'][cell]
                        uy = self.point_data['Uy'][cell]
                    u = np.empty(ux.size * 2)
                    u[::2] = ux
                    u[1::2] = uy
                    element = self.ElementClasses[celltype](self.points[cell], N_dof=N_dof)
                    if 'X' in self.cell_data.keys():
                        try:
                            S.append(element.stress(u, tensors[0] + tensors[1] * (self.cell_data['X'][celltype][i])**3))
                        except IndexError:
                            S.append(element.stress(u, tensors[0] * (self.cell_data['X'][celltype][i])**3))
                    else:
                        S.append(element.stress(u, tensors[0]))

                S = np.array(S)

                Sv = np.sqrt(((S[:, 0] - S[:, 1]) ** 2 + (S[:, 1] - S[:, 2]) ** 2 + (S[:, 2] - S[:, 0]) ** 2) / 2)
                self.cell_data['Sx'][celltype] = S[:, 0]
                self.cell_data['Sy'][celltype] = S[:, 1]
                self.cell_data['Sxy'][celltype] = S[:, 2]
                self.cell_data['Sv'][celltype] = Sv

    def add_mode(self, name, V, N_dof):
        """
        Adds a eigen vector to the mesh as point data
        """
        data_names = [name + '_' + vi for vi in self.point_variables]

        for dof in np.arange(N_dof):
            self.point_data[data_names[dof]] = V[np.arange(dof, len(V), N_dof)]
        self.wrap(name)

    def compute_element_centers(self):
        """
        Computes the center of every element in the mesh
        """

        def cell_center(points):
            x, y = points.T[:2]
            x_center = np.mean(x)
            y_center = np.mean(y)
            return [x_center, y_center]

        # triangle points
        triangle_centers = []
        if 'triangle' in self.contains:
            for nodes in self.cells['triangle']:
                element_points = self.points[nodes]
                triangle_centers.append(cell_center(element_points))
            self.cell_centers['triangle'] = triangle_centers

        # triangle points
        quad_centers = []
        if 'quad' in self.contains:
            for nodes in self.cells['quad']:
                element_points = self.points[nodes]
                quad_centers.append(cell_center(element_points))
            self.cell_centers['quad'] = quad_centers

        all_cell_centers = [self.cell_centers[cell_type] for cell_type in self.contains]
        self.all_cell_centers = np.array([center for center_list in all_cell_centers for center in center_list])

    def _get_Xi_keys(self):
        """
        Generates the array containing all the density iterations of the topology
        optimization
        """
        keys = []
        # Check that the mesh has a density field
        if 'X' in self.cell_data:
            i = 1
            while True:
                key = 'X{}'.format(i)
                if key in self.cell_data:
                    keys.append(key)
                    i += 1
                else:
                    break
            return keys

    def make_triangles_quad(self):
        """
        Convert the triangles cells to T6 cells with 6 points
        """
        # Create the nodes between the existing corners
        new_nodes = []
        new_nodes_ids = []
        # Create a new node between every node for each cell
        for cell in self.cells['triangle']:
            # Add the first corner at the end
            cell = np.append(cell, cell[0])
            # For each pair of nodes
            for i in range(len(cell) - 1):
                # Store the node ids
                new_nodes_ids.append([cell[i], cell[i + 1]])
                # Create the new node and add to list
                new_nodes.append(self._mid_point(cell[i], cell[i + 1]))

        # Convert to array
        new_nodes = np.array(new_nodes)
        # Make unique
        unique_nodes, inverse_indexes = np.unique(new_nodes, axis=0, return_inverse=True)
        # Split the two nodes list
        i, j = np.array(new_nodes_ids).T
        # Create a sparse matrix where M[1,2] returns the index of the new node between nodes 1 and 2
        N_Mat = scipy.sparse.coo_matrix((inverse_indexes + self.points.shape[0], (i, j)))
        N_Mat = N_Mat.tocsr()

        new_cells = []
        for cell in self.cells['triangle']:
            new_cell = [cell[0], N_Mat[cell[0], cell[1]],
                        cell[1], N_Mat[cell[1], cell[2]],
                        cell[2], N_Mat[cell[2], cell[0]]]
            new_cells.append(new_cell)
        self.cells['triangle'] = np.array(new_cells)
        self.points = np.vstack([self.points, unique_nodes])

    def _mid_point(self, id1, id2):
        p1 = self.points[id1]
        p2 = self.points[id2]
        x = (p1[0] + p2[0]) / 2
        y = (p1[1] + p2[1]) / 2
        return np.array([x, y, 0])

    def cell_to_point_data(self, which):
        """Converts cell data to point data by linear interpolation"""
        # Get all the cell values
        values = np.hstack([val for val in self.cell_data[which].values()])
        # Get the points at cell centers
        cell_points = np.hstack([pts for pts in self.cell_centers.values()])
        # Linear interpolation of the points
        data_linear = griddata(cell_points, values, self.points[:, :2], method='linear')
        # Nearest interpolation for the points on the boundaries
        data_nearest = griddata(cell_points, values, self.points[:, :2], method='nearest')
        # Use the nearest value when linear is Nan
        data = data_linear
        data[np.isnan(data)] = data_nearest[np.isnan(data)]
        if not (values < 0).any():
            data[data < 0] = values.min()
        # Add to the mesh point data
        self.point_data[which] = data

    def point_data_to_STL(self, filename, which, hb=1, hc=0, symmetric=False, scale=1, flip=False):
        """ Creates a 3D STL mesh where the z coordinate is the point data"""
        # Get two copies of the mesh nodes points
        points1 = self.points.copy()
        points2 = self.points.copy()

        if symmetric:
            # get the z values from the point data
            if hc > 0:
                # if there is a core
                z_top = hc * self.point_data[which] + 2 * hb + hc
                z_bot = -hc * (self.point_data[which] - 1)
            else:
                # if there is only a base
                z_top = hb * self.point_data[which] + hb
                z_bot = -hb * (self.point_data[which] - 1)
            # points at the top
            points1[:, 2] = z_top
            # points at the bottom
            points2[:, 2] = z_bot
            # Combine all the points
            points = np.vstack([points1, points2])
        else:
            # get the z values from the point data
            if hc > 0:
                # if there is a core
                z_top = hc * self.point_data[which] + hb
                z_bot = np.zeros((self.points.shape[0]))
            else:
                # if there is only a base
                z_top = hb * self.point_data[which]
                z_bot = np.ones((self.points.shape[0])) * self.point_data[which].min()

            # points at the top
            points1[:, 2] = z_top
            # points at the bottom
            points2[:, 2] = z_bot
            # Combine all the points
            points = np.vstack([points1, points2])

        # Get all the mesh cells converted to triangles
        tris1 = self.plot.all_tris.copy()
        tris2 = self.plot.all_tris.copy()
        # add the size to points index so they refer to new points
        tris2 += self.points.shape[0]

        # Flip the triangles underneath
        if flip:
            # flip the main triangles
            tris2[:, 1:] = np.flip(tris2[:, 1:], axis=1)
        else:
            # flip the triangles underneath
            tris1[:, 1:] = np.flip(tris1[:, 1:], axis=1)

        # Fill the boundaries with triangles
        npts = self.points.shape[0]
        # get the boundary groups
        boundary_groups = self.capture_boundary(separate_subsets=True)
        # add wall triangles for each group
        tris3 = []
        for group in boundary_groups:
            for i, node in enumerate(group[:-1]):
                tris3.append([node, node + npts, group[i + 1]])
                tris3.append([group[i + 1], node + npts, group[i + 1] + npts])
            tris3.append([group[-1], group[-1] + npts, group[0]])
            tris3.append([group[0], group[-1] + npts, group[0] + npts])

        tris3 = np.array(tris3)
        tris3[:, 1:] = np.flip(tris3[:, 1:], axis=1)

        tris = np.vstack([tris1, tris2, tris3])
        cells = [('triangle', tris)]
        points *= scale
        meshio_mesh = meshio.Mesh(points, cells)
        meshio_mesh.write(filename + '.stl')

    def height_with_area(self, which, target_area_fraction, precision=10e-4):
        """
        Compute the contour line height of a mesh variable for wich the
        circonspect area is equal to target_area_fraction.
        :param which: mesh point data variable key
        :param target_area_fraction: target area fraction [0, 1]
        :param precision: precision of the returned height
        :return: height of the contour line
        """
        h = 0
        h_max = self.point_data[which].max()
        increment = precision * h_max
        all_area = points_area(self.points)
        area = all_area
        while (area/all_area) > target_area_fraction:
            h += increment
            points = self.points[self.point_data['zc'] > h, :]
            area = points_area(points)
        return h


class MeshPlot(object):
    """
    General class for plotting FEM results on meshes
    """

    def __init__(self, mesh, backend='matplotlib'):
        self.mesh = mesh
        self.backend = backend
        self._make_all_cells_T3()

    def quality(self):
        """
        Mesh quality cell data plot
        """
        try:
            self.cell_data('quality', cmap='viridis_r')
        except KeyError:
            self.mesh.get_quality()
            self.cell_data('quality', cmap='viridis_r')

    def wrapped_2D(self, color='#FFBFC8'):
        """
        Plot the wrapped mesh in the X-Y plane with a nice color
        """
        if self.backend == 'matplotlib':
            ax = plt.gca()
            ax.set_title('Wrapped 2D mesh')
            ax.set_aspect('equal')
            ax.set_axis_off()

            for cell_type in self.mesh.contains:
                for cell in self.mesh.cells[cell_type]:
                    cell_points = self.mesh.wrapped_points_2D[cell]
                    x = cell_points.T[0]
                    y = cell_points.T[1]
                    ax.fill(x, y, color, edgecolor='k', zorder=-1)

    def point_data(self, which, wrapped=True, cmap=None):
        """
        Plots the displacement colormap on the wrapped mesh
        """
        try:
            if wrapped:
                self._empty_wrapped_2D()
                triangulation = tri.Triangulation(*self.mesh.wrapped_points_2D.T, self.mesh.plot.all_tris)
            else:
                self._empty_mesh_2D()
                triangulation = tri.Triangulation(*self.mesh.points.T[:2], self.all_tris)
            ax = plt.gca()
            ax.set_title('{which} point data'.format(which=which))
            ax.tricontourf(triangulation, self.mesh.point_data[which], cmap=cmap)

        except AttributeError:
            self.point_data(which, wrapped=False, cmap=cmap)

    def cell_data(self, which, cmap='Greys'):
        """
        Display the mesh cell data
        """
        if self.backend == 'matplotlib':
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_axis_off()
            cm = plt.cm.get_cmap(cmap)
            # normalize the cell data
            all_cell_data = self.mesh.cell_data[which]
            all_cell_data = np.hstack([all_cell_data[cell_type] for cell_type in self.mesh.contains])
            if all_cell_data.max() > all_cell_data.min():
                all_cell_data = (all_cell_data - all_cell_data.min()) / (all_cell_data.max() - all_cell_data.min())
            elif all_cell_data.max() != 0:
                all_cell_data = all_cell_data / all_cell_data.max()

            for cell_type in self.mesh.contains:
                current_cell_size = self.mesh.cells[cell_type].shape[0]
                current_cell_data = all_cell_data[:current_cell_size]
                for nodes, cell_data in zip(self.mesh.cells[cell_type], current_cell_data):
                    x, y = self.mesh.points[nodes].T[:2]
                    color = cm(cell_data)
                    plt.fill(x, y, color=color)

    def quad_tris(self):
        """
        Plots the mesh highlighting where the quads and tris are
        """
        # Quads are a nice blue and triangles a smooth rose-ish red
        cell_color_dict = {'triangle': '#FFAFCC', 'quad': '#D1E8FF'}

        if self.backend == 'matplotlib':
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_axis_off()
            for cell_type in self.mesh.contains:
                for nodes in self.mesh.cells[cell_type]:
                    x, y = self.mesh.points[nodes].T[:2]
                    plt.fill(x, y, color=cell_color_dict[cell_type], edgecolor='black', linewidth=0.3)

    def mode_2D(self, name, factor=1):
        """
        Plots the 2D mode corresponding to the specified mode name
        """
        self.mesh.unwrap()
        self.mesh.wrap(name, factor)
        self._empty_wrapped_2D()

    def mode(self, v, wrapped=False, N_dof=6, **kwargs):
        """
        Plots the eigenmode corresponding to vector v
        :param v: eigenvector
        :param wrapped: wrap the x-y coordinates of the mesh
        :return: None
        """
        self.mesh.add_mode('mode_plot', v, N_dof)
        self.point_data('mode_plot_Uz', wrapped=wrapped, **kwargs)

    def mode_Z(self, name):
        self.point_data(name + '_' + 'Uz', wrapped=False)

    def _empty_wrapped_2D(self):
        """
        Utility function to plot the wrapped mesh without fill
        """
        pass
        """
        Plots the empty wrapped mesh
        """
        if self.backend == 'matplotlib':
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_axis_off()
            for cell_type in self.mesh.contains:
                for nodes in self.mesh.cells[cell_type]:
                    x, y = self.mesh.wrapped_points_2D[nodes].T
                    plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.3)

    def _empty_mesh_2D(self):
        """
        Utility function to plot the mesh without fill
        """
        if self.backend == 'matplotlib':
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_axis_off()
            for cell_type in self.mesh.contains:
                for nodes in self.mesh.cells[cell_type]:
                    x, y = self.mesh.points[nodes].T[:2]
                    plt.fill(x, y, edgecolor='black', fill=False, linewidth=0.3)

    def _make_all_cells_T3(self):
        """
        Convert the quadrilateral elements to triangles
        """
        T3_cells_group = []
        if 'triangle' in self.mesh.contains:
            if self.mesh.cells['triangle'][0].shape[0] == 3:
                T3_cells_group.append(self.mesh.cells['triangle'])
            elif self.mesh.cells['triangle'][0].shape[0] == 6:
                T3_cells = []
                for cell in self.mesh.cells['triangle']:
                    T3_cells.append(cell[[0, 1, 3]])
                    T3_cells.append(cell[[1, 2, 3]])
                    T3_cells.append(cell[[3, 4, 5]])
                    T3_cells.append(cell[[0, 3, 5]])
                _ = T3_cells.pop(0)
                T3_cells = np.array(T3_cells)
                T3_cells_group.append(T3_cells)

        if 'quad' in self.mesh.contains:
            quads = self.mesh.cells['quad']
            T3_cells = []
            for quad in quads:
                T3_cells.append(quad[[0, 1, 2]])
                T3_cells.append(quad[[2, 3, 0]])
            # Make into array
            T3_cells = np.array(T3_cells)
            T3_cells_group.append(T3_cells)

        self.all_tris = np.concatenate(T3_cells_group)


"""
Mesh manipulation
"""


def load_vtk(file):
    """
    Load a vtk mesh with meshio and convert it to FEMOL mesh
    :param file: filepath
    :return: FEMOL mesh
    """
    meshio_mesh = meshio.read(file)
    femol_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict)
    femol_mesh.point_data = meshio_mesh.point_data
    cell_data = meshio_mesh.cell_data
    new_cell_data = {}
    for data in cell_data:
        new_data = {}
        for i, cell_type in enumerate(femol_mesh.contains):
            new_data[cell_type] = cell_data[data][i]
        new_cell_data[data] = new_data
    femol_mesh.cell_data = new_cell_data
    return femol_mesh


"""
Pre defined Meshes
"""


def rectangle_Q4(Lx, Ly, nelx, nely):
    """
    A function returning a structured rectangular 2D quadrilateral mesh
    """
    # Get the points and cells from meshzoo module
    points, cells = meshzoo.rectangle_quad((0, Lx), (0, Ly), (nelx, nely))
    cell_dict = {'quad': cells}
    mesh = Mesh(points, cell_dict, structured=True, quad_element=FEMOL.elements.Q4)
    return mesh


def rectangle_T3(Lx, Ly, nelx, nely):
    """
    A function returning an unstructured rectangular 2D triangular mesh
    """
    # Get the points and cells from meshzoo module
    points, cells = meshzoo.rectangle_tri((0, Lx), (0, Ly), (nelx, nely))
    cell_dict = {'triangle': cells}
    mesh = Mesh(points, cell_dict, tri_element=FEMOL.elements.T3)
    return mesh


def rectangle_T6(Lx, Ly, nelx, nely):
    """
    A function returning a unstructured rectangular 2D quadratic triangles mesh
    """
    # Get the points and cells from meshzoo module
    points, cells = meshzoo.rectangle_tri((0, Lx), (0, Ly), (nelx, nely))
    cell_dict = {'triangle': cells}
    mesh = Mesh(points, cell_dict, tri_element=FEMOL.elements.T6)
    return mesh


def circle_Q4(R, N_ele, **kwargs):
    """
    Function returning a circular mesh with quadrilaterals
    meshzoo : Ordered mesh from meshzoo
    pygmsh : Generated mesh from pygmsh
    """
    # Use meshzoo for simple meshes
    points, cells = meshzoo.disk_quad(N_ele)
    cells_dict = {'quad': cells}
    # Create a mesh with the Q4 elements (circle with quads is not a good quality mesh)
    mesh = Mesh(points * (R / (np.sqrt(2) / 2)), cells_dict, **kwargs)
    return mesh


def circle_T3(R, N_ele, order=7):
    """
    Function returning a circular mesh with quadrilaterals
    R: Circle radius
    N_ele: Number of elements on the radius
    order: Source polygon for the inflation
    Taken from: https://github.com/nschloe/meshzoo
    """
    # Use meshzoo for simple meshes
    points, cells = meshzoo.disk(order, N_ele)
    cells_dict = {'triangle': cells}
    # Create a mesh with the T3 elements
    mesh = Mesh(points, cells_dict, tri_element=FEMOL.elements.T3)
    # Scale the points
    mesh.points *= R
    return mesh


def circle_T6(R, N_ele, order=7):
    """
    Function returning a circular mesh with quadrilaterals
    R: Circle radius
    N_ele: Number of elements on the radius
    order: Source polygon for the inflation
    Taken from: https://github.com/nschloe/meshzoo
    """
    # Use meshzoo for simple meshes
    points, cells = meshzoo.disk(order, N_ele)
    cells_dict = {'triangle': cells}
    # Create a mesh with the T3 elements
    mesh = Mesh(points, cells_dict, tri_element=FEMOL.elements.T6)
    # Scale the points
    mesh.points *= R
    return mesh


def guitar(L=1, lcar=0.05, option='quad', algorithm=1):
    """
    2D mesh of a classical guitar soundboard
    """
    with pygmsh.geo.Geometry() as geom:
        # Points on the boundary
        p0 = (0, 0.38 * L)
        p2 = (0.25 * L, 0.76 * L)
        p5 = (0.8175 * L, 0.09 * L)
        p7 = (1 * L, 0.38 * L)
        # Ellipse 1
        elc1 = (0.25 * L, 0.38 * L)
        elh1 = 0.76 * L
        # Ellipse 2
        elc2 = (0.8175 * L, 0.38 * L)
        elh2 = 0.58 * L

        # Left top ellipse
        elsa1 = geom.add_point(p0, lcar)
        elce1 = geom.add_point(elc1, lcar)
        pt = (elc1[0], elc1[1] + elh1 / 2)
        elax1 = geom.add_point(pt, lcar)
        elso1 = geom.add_point(p2, lcar)
        ell1 = geom.add_ellipse_arc(elsa1, elce1, elax1, elso1)

        # Top guitar side 1
        p1 = FEMOL.domains.create_polynomial(0.25 * L, 0.76 * L, 0.625 * L, (0.71225 - 0.1645 / 2) * L, 0)
        x1 = np.linspace(0.25 * L, 0.625 * L, 10)
        y1 = p1[0] * x1 ** 3 + p1[1] * x1 ** 2 + p1[2] * x1 + p1[3]
        points = [(xi, yi) for xi, yi in zip(x1, y1)]
        spli1_points = [elso1]
        for pt in points[1:]:
            spli1_points.append(geom.add_point(pt, lcar))
        spli1 = geom.add_spline(spli1_points)

        # Top guitar side 2
        p2 = FEMOL.domains.create_polynomial(0.625 * L, (0.71225 - 0.1645 / 2) * L, 0.8175 * L, 0.67 * L, 0)
        x2 = np.linspace(0.625 * L, 0.8175 * L, 10)
        y2 = p2[0] * x2 ** 3 + p2[1] * x2 ** 2 + p2[2] * x2 + p2[3]
        points = [(xi, yi) for xi, yi in zip(x2, y2)]
        spli2_points = [spli1_points[-1]]
        for pt in points[1:]:
            spli2_points.append(geom.add_point(pt, lcar))
        spli2 = geom.add_spline(spli2_points)

        # Left top ellipse
        elce2 = geom.add_point(elc2, lcar)
        pt = (elc2[0], elc2[1] + elh2 / 2)
        elax2 = geom.add_point(pt, lcar)
        elso2 = geom.add_point(p7, lcar)
        ell2 = geom.add_ellipse_arc(spli2_points[-1], elce2, elax2, elso2)

        # Left bottom ellipse
        elso3 = geom.add_point(p5, lcar)
        ell3 = geom.add_ellipse_arc(elso2, elce2, elax2, elso3)

        # Bottom side 1
        p3 = FEMOL.domains.create_polynomial(0.625 * L, (0.04775 + 0.1645 / 2) * L, 0.8175 * L, 0.09 * L, 0)
        x3 = np.linspace(0.8175, 0.625, 10)
        y3 = p3[0] * x3 ** 3 + p3[1] * x3 ** 2 + p3[2] * x3 + p3[3]
        points = [(xi, yi) for xi, yi in zip(x3, y3)]
        spli3_points = [elso3]
        for pt in points[1:]:
            spli3_points.append(geom.add_point(pt, lcar))
        spli3 = geom.add_spline(spli3_points)

        # Bottom side 2
        p4 = FEMOL.domains.create_polynomial(0.25 * L, 0, 0.625 * L, (0.04775 + 0.1645 / 2) * L, 0)
        x4 = np.linspace(0.625 * L, 0.25 * L, 10)
        y4 = p4[0] * x4 ** 3 + p4[1] * x4 ** 2 + p4[2] * x4 + p4[3]
        points = [(xi, yi) for xi, yi in zip(x4, y4)]
        spli4_points = [spli3_points[-1]]
        for pt in points[1:]:
            spli4_points.append(geom.add_point(pt, lcar))
        spli4 = geom.add_spline(spli4_points)

        # Left bottom ellipse
        ell4 = geom.add_ellipse_arc(spli4_points[-1], elce1, elax1, elsa1)

        # Guitar outline curve loop
        loop1 = geom.add_curve_loop([ell1, spli1, spli2, ell2, ell3, spli3, spli4, ell4])

        # Soundhole
        hole = geom.add_circle([0.673 * L, 0.38 * L], 0.175 * L / 2, lcar, make_surface=False)
        loop2 = geom.add_curve_loop(hole.curve_loop.curves)

        s1 = geom.add_plane_surface(loop1, [loop2])

        if option == 'quad':
            geom.set_recombined_surfaces([s1])
        elif option == 'triangle':
            pass
        mesh = geom.generate_mesh(algorithm=algorithm)

    FEMOL_mesh = FEMOL.Mesh(mesh.points, mesh.cells_dict)
    return FEMOL_mesh


def prism_mesh(a, b, h):
    """
    Creates a 3D prismatic mesh
    :param a: width
    :param b: depth
    :param h: height
    :return: meshio 3D mesh suitable for STL write
    """
    # Create the first rectangle
    mesh = FEMOL.mesh.rectangle_T3(a, b, 1, 1)
    # Add a second rectangle to the points
    points = mesh.points.copy()
    mesh.points[:, 2] += h
    all_points = np.vstack([mesh.points, points])
    # Add the second rectangle to the cells
    cells = mesh.all_cells.copy()
    # Flip the triangles so they point downward
    cells[:, 1:] = np.flip(mesh.all_cells[:, 1:], axis=1)
    # Make the cells reference the new points
    cells += cells.max() + 1
    # Add wall cells surrounding the two rectangles
    wall_cells = np.array([[4, 5, 0],
                           [5, 1, 0],
                           [5, 7, 1],
                           [7, 3, 1],
                           [6, 3, 7],
                           [6, 2, 3],
                           [4, 2, 6],
                           [4, 0, 2]])
    all_cells = np.vstack([mesh.all_cells, cells, wall_cells])
    meshio_mesh = meshio.Mesh(all_points, [('triangle', all_cells.astype(int))])
    return meshio_mesh
