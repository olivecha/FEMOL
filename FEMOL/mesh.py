import numpy as np
import meshio
import meshzoo
import pygmsh
import gif
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import FEMOL.elements
# Configuration
gif.options.matplotlib["dpi"] = 150

class Mesh(object):
    """
    A class representing a generic 2D mesh
    """
    ElementClasses = {'triangle': FEMOL.elements.T3,
                      'quad': FEMOL.elements.Q4, }
    point_variables = ['Ux', 'Uy', 'Uz', 'Tx', 'Ty', 'Tz']

    def __init__(self, points, cell_dict, structured=False, quad_element=None, tri_element=None):
        """
        Constructor for the general mesh class
        """
        # Empty point and cell data dict
        self.point_data = {}
        self.cell_data = {}
        self.cell_centers = {}
        # Store the cells and points into the Mesh instance
        self.cells = cell_dict
        if points.shape[1] == 2:
            points = np.hstack([points, np.zeros((points.shape[0],1))])
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

        # Compute the total number of nodes
        self.N_nodes = self.points.shape[0]
        self.full_nodes = np.hstack([self.cells[cell_type].reshape(-1) for cell_type in self.contains])
        self.full_nodes = np.unique(self.full_nodes)
        self.empty_nodes = np.nonzero(~np.in1d(np.arange(self.N_nodes), self.full_nodes))[0]

        # Define the element classes
        if tri_element:
            self.ElementClasses['triangle'] = tri_element
        if quad_element:
            self.ElementClasses['quad'] = quad_element

        # Create the plotter instance
        self.plot = MeshPlot(self)

        # Compute all the cells in an array
        all_cells = [self.cells[cell_type] for cell_type in self.contains]
        all_cells = [nodes for node_list in all_cells for nodes in node_list]
        self.all_cells = np.array(all_cells, dtype=object)

    def display(self, backend='matplotlib', color='#D1E8FF'):
        """
        Plot the mesh using the specified backend
        Supported are :
        - 'matplotlib'
        - #TODO 'Pyvista'
        """
        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title('Mesh')
            ax.set_aspect('equal')
            ax.set_axis_off()

            for cell_type in self.contains:
                for cell in self.cells[cell_type]:
                    cell_points = self.points[cell]
                    x = cell_points.T[0]
                    y = cell_points.T[1]
                    ax.fill(x, y, color, edgecolor='k', zorder=-1)

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

        meshio_mesh = meshio.Mesh(
            self.points,
            self.cells,
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
        # Remove the nodes that are not in any cells
        nodes = nodes[np.in1d(nodes, self.full_nodes)]
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

        base_empty_node_range = np.tile(np.arange(N_dof), self.empty_nodes.shape[0])  # [0,...,N_dof]*N_empty_nodes
        empty_node_dof_range = N_dof*np.repeat(self.empty_nodes, N_dof)  # [node*N_dof, ..., node*N_dof] *N_empty_nodes
        empty_node_indexes = base_empty_node_range + empty_node_dof_range

        self.rows = np.append(np.hstack(rows), empty_node_indexes)
        self.cols = np.append(np.hstack(cols), empty_node_indexes)

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

    def optimization_to_gif(self):
        """
        Creates a gif from the optimization result
        """
        keys = self._get_Xi_keys()

        @gif.frame
        def plot_frame(key):
            self.plot.cell_data(key)

        frames = []
        for key in keys:
            frame = plot_frame(key)
            frames.append(frame)

        path = 'results/TOPOPT/gifs/'
        gif_name = path + 'topopt_' + FEMOL.utils.unique_time_string() + '.gif'
        fps = 20  # frames/s

        gif.save(frames, gif_name, duration=len(frames)/fps, unit="s", between="startend")

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


class MeshPlot(object):
    """
    General class for plotting FEM results on meshes
    """
    def __init__(self, mesh, backend='matplotlib'):
        self.mesh = mesh
        self.backend = backend
        self._quads_to_tris()

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
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.set_title('Wrapped 2D mesh')
            ax.set_aspect('equal')
            ax.set_axis_off()

            for cell_type in self.mesh.contains:
                for cell in self.mesh.cells[cell_type]:
                    cell_points = self.mesh.wrapped_points_2D[cell]
                    x = cell_points.T[0]
                    y = cell_points.T[1]
                    ax.fill(x, y, color, edgecolor='k', zorder=-1)

    def point_data(self, which, wrapped=True):
        """
        Plots the displacement colormap on the wrapped mesh
        """
        if wrapped:
            self._empty_wrapped_2D()
            triangulation = tri.Triangulation(*self.mesh.wrapped_points_2D.T, self.mesh.plot.all_tris)
        else:
            self._empty_mesh_2D()
            triangulation = tri.Triangulation(*self.mesh.points.T, self.mesh.plot.all_tris)
        ax = plt.gca()
        ax.set_title('{which} point data'.format(which=which))
        ax.tricontourf(triangulation, self.mesh.point_data[which])

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
                all_cell_data = (all_cell_data - all_cell_data.min())/(all_cell_data.max() - all_cell_data.min())
            elif all_cell_data.max() != 0:
                all_cell_data = all_cell_data/all_cell_data.max()

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

    def _quads_to_tris(self):
        """
        Convert the quadrilateral elements to triangles
        """
        if 'quad' in self.mesh.contains:
            quads = self.mesh.cells['quad']
            tris = []
            for quad in quads:
                tris.append(quad[[0, 1, 2]])
                tris.append(quad[[2, 3, 0]])

            tris = np.array(tris)
            if 'triangle' in self.mesh.contains:
                self.all_tris = np.concatenate([self.mesh.cells['triangle'], tris])
            else:
                self.all_tris = np.array(tris)
        else:
            self.all_tris = self.mesh.cells['triangle']

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
    A function returning a structured rectangular 2D triangular mesh
    """
    # Get the points and cells from meshzoo module
    points, cells = meshzoo.rectangle_tri((0, Lx), (0, Ly), (nelx, nely))
    cell_dict = {'triangle': cells}
    mesh = Mesh(points, cell_dict, tri_element=FEMOL.elements.T3)

    return mesh

def circle_Q4(R, N_ele, which='meshzoo'):
    """
    Function returning a circular mesh with quadrilaterals
    meshzoo : Ordered mesh from meshzoo
    pygmsh : Generated mesh from pygmsh
    """
    if which == 'meshzoo':
        # Use meshzoo for simple meshes
        points, cells = meshzoo.disk_quad(N_ele)
        cells_dict = {'quad': cells}
        # Create a mesh with the Q4 elements (circle with quads is not a good quality mesh)
        mesh = FEMOL.Mesh(points * (R / (np.sqrt(2) / 2)), cells_dict, quad_element=FEMOL.elements.Q4)
        return mesh

    elif which == 'pygmsh':
        mesh_size = R/(N_ele/2)
        with pygmsh.geo.Geometry() as geom:
            circle = geom.add_circle([0.0, 0.0], R, mesh_size=mesh_size, make_surface=False)
            loop = geom.add_curve_loop(circle.curve_loop.curves)
            surf = geom.add_surface(loop)
            geom.set_recombined_surfaces([surf])
            mesh_p = geom.generate_mesh(dim=2, algorithm=0)

        mesh = FEMOL.Mesh(mesh_p.points, mesh_p.cells_dict, quad_element=FEMOL.elements.Q4)
        return mesh
