import FEMOL.utils
import FEMOL.elements
from FEMOL.mesh import Mesh
from FEMOL.laminate import Layup
# Numpy
import numpy as np
# SciPy
import scipy.linalg
import scipy.stats
import scipy.sparse
import scipy.sparse.linalg
# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Ipython
from IPython.display import clear_output
# Python
import time

__all__ = ['FEM_Problem', 'Mesh', 'Layup',  'TOPOPT_Problem']


class FEM_Problem(object):
    """
    A Class used to define a general structural FEM Problem

    Supported physics are :
    - Displacement : Solving KU = F
        With supported models :
        - plane : two dof displacement with plane stress
        - plate : six dof displacement with Reissner-Mindlin plate
        - #TODO 3D

    - Modal : Solving det(K - lambda M) = 0
        With supported models :
        - plane : two dof displacement with plane stress
        - plate : six dof displacement with Reissner-Mindlin plate
        - #TODO 3D
    """
    dof_dict = {'plane': 2, 'plate': 6}

    def __init__(self, physics, model, mesh, ):
        # Store parameters
        self.mesh = mesh
        self.physics = physics
        self.model = model
        self.N_dof = self.dof_dict[model]
        self.coating = False

        # Compute the K matrix structure from the mesh
        self.mesh.global_matrix_indexes(self.N_dof)

        # Pre allocate attributes
        self.fixed_nodes = np.array([])
        self.fixed_domains = []
        self.fixed_ddls = []
        self.forces = []
        self.force_domains = []
        self.F = np.zeros(self.N_dof * self.mesh.N_nodes)
        self.fr = []

    """
    Defining the problem
    """

    def help(self):
        """
        Help function explaining how to define the current FEM Problem
        """
        if self.physics == 'displacement':
            print("""
            Plane-Stress displacement model : \n
            1. Define one material for the solid approach and two  \n
            \t for the coating approach. \n
            2. Define one stiffness tensors per materials \n
            \t (isotropic materials only require thickness and orthotropic require a \n
            \t Composite material layup, see help(FEMOL.laminate.Layup))\n
            3. Define external forces \n
            4. Define boundary conditions \n
            5. Assemble the K matrix \n
            6. Solve the problem
            """)

        elif self.physics == 'modal':
            print("""
                Plane-Stress displacement model : \n
                1. Define one material for the solid approach and two  \n
                \t for the coating approach. \n
                2. Define one stiffness tensors per materials \n
                \t (isotropic materials only require thickness and orthotropic require a \n
                \t Composite material layup, see help(FEMOL.laminate.Layup))\n
                3. Define external forces \n
                4. Define boundary conditions \n
                5. Assemble the K matrix \n
                6. Solve the problem
                """)

    def plot(self):
        """
        Plots the current FEM Problem, (boundary conditions and applied forces)
        """
        # TODO : plot forces for 6 dof
        fixed_dof_label_dict = {(0,): 'x displacement', (1,): 'y displacement', (2,): 'z displacement',
                                (3,): 'x rotation', (4,): 'y rotation', (0, 1): 'x-y displacement',
                                (2, 3): 'supported x', (2, 4): 'supported y', (0, 1, 2, 3, 4, 5): 'clamped', }

        fixed_dof_color_dict = {(0,): '#FF716A', (1,): '#FFBB6A', (2,): '#99FFA9',
                                (3,): '#A0FF6A', (4,): '#6AFFDF', (0, 1): '#6A75FF',
                                (2, 3): '#6A75FF', (2, 4): '#D26AFF', (0, 1, 2, 3, 4, 5): '#FF3A3A', }

        # Display the mesh
        self.mesh.display()

        plt.title('FEM problem')

        # Plot the fixed nodes and their degrees of freedom
        for domain, ddls in zip(self.fixed_domains, self.fixed_ddls):
            nodes = self.mesh.domain_nodes(domain)
            label = fixed_dof_label_dict[tuple(ddls)]
            color = fixed_dof_color_dict[tuple(ddls)]
            plt.scatter(*self.mesh.points[nodes].T[:2], label=label, color=color, zorder=1)

        for force, domain in zip(self.forces, self.force_domains):
            force_points = self.mesh.points[self.mesh.domain_nodes(domain)]
            plt.scatter(*force_points.T[:2], color='#FC1EFF', label='Force')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title='Problem BC and forces',
                   loc='upper left', bbox_to_anchor=(1, 0.9))

    def define_materials(self, *materials):
        """
        Define the materials associated to the current FEM Problem
        """
        self.materials = materials
        if len(materials) == 2:
            self.coating = True

    def define_tensors(self, *tensors):
        """
        Define the stiffness tensors used to build the element
        stiffness matrix
        __________________________________
        Solid problem :
        define_tensor(thickness or Layup)

        Coating problem :
        define_tensor(base thickness or base layup, coating thickness or coating layup)

        """

        # Empty tensor list
        C_A, C_D, C_G, ho = [], [], [], []

        # Loop over the user defined tensors
        for tensor, material in zip(tensors, self.materials):

            if material.kind == 'orthotropic':
                layup = tensor
                ho.append(layup.hA)
                C_A.append(layup.get_A_FEM())
                C_D.append(layup.D_mat)
                C_G.append(layup.G_mat)

            elif material.kind == 'isotropic':
                thickness = tensor
                ho.append(thickness)
                C_A.append(material.plane_tensor(thickness))
                C_D.append(material.bending_tensor(thickness))
                C_G.append(material.shear_tensor(thickness))

        # Define attributes according to kind
        self.C_A, self.C_D, self.C_G, self.ho = C_A[0], C_D[0], C_G[0], ho[0]

        if len(C_A) == 2:
            self.coat_C_A, self.coat_C_D, self.coat_C_G, self.coat_ho = C_A[1], C_D[1], C_G[1], ho[1]
            self.coating = True

    """
    Global Matrix Assembly
    """

    def assemble(self, which, X=None, p=None):
        """
        General assembly function for the FEM Problem

        Calls the corresponding assembly function to assemble the global
        matrix associated to either 'K' or 'M'
        This method should be called after the fixed boundaries are added

        Parameters
        ----------
        which : Global matrix kind 'K' or 'M'
        X : Element density vector
        p : Element density penalty
        """

        if which == 'M':
            self._assemble_M(X=X, p=p)

        elif which == 'K':
            self.assemble('F')
            self._assemble_K(X=X, p=p)

        elif which == 'F':
            self._assemble_F()

    """
    Solver
    """

    def solve(self, verbose=True, filtre=None):
        """
        solve the FEM from kind
        :return: FEM Result class associated to the Problem kind
        """

        if self.physics == 'displacement':
            return self._displacement_solve(verbose=verbose)

        elif self.physics == 'modal':
            return self._modal_solve(verbose=verbose, filtre=filtre)

    def _displacement_solve(self, verbose):

        # Assemble if not assembled
        if not hasattr(self, 'K'):
            self.assemble('K')

        if verbose:
            print('solving with scipy')
        U = self._scipy_displacement_solve()
        self.mesh.add_displacement(U, self.N_dof)

        return self.mesh

    def _modal_solve(self, verbose, filtre=0):
        """
        Modal solver for the eigen value problem
        """
        # TODO : Add eigen vectors into mesh

        if verbose:
            now = time.time()
            print('solving using scipy')

        for mat in ['M', 'K']:
            if not hasattr(self, mat):
                self.assemble(mat)

        w, v = self._scipy_modal_solve()
        # Transpose the eigen vectors
        v = v.T
        # Remove Nan values and 0 values
        v = v[~np.isnan(w)]
        w = w[~np.isnan(w)]
        v = v[~np.isclose(w, 0)]
        w = w[~np.isclose(w, 0)]

        if verbose:
            print('solved in : ', time.time() - now, ' s')

        # Defined eigen_filters
        eigen_filters = {0: self._filter_eigenvalues_0,
                         1: self._filter_eigenvalues_1,
                         2: self._filter_eigenvalues_2,}

        # Filter according to the chosen filter
        current_filter = eigen_filters[filtre]
        w, v = current_filter(w, v)

        return np.sqrt(w) / (2 * np.pi), v

    def _scipy_displacement_solve(self):
        U = scipy.sparse.linalg.spsolve(self.K, self.F)
        return U

    def _scipy_modal_solve(self):
        """
        Eigenvalue and eigenvectors solver using scipy
        """
        # TODO : Optimize eigenvalue solver (sparse + driver)
        # Solve the eigenvalue problem
        w, v = scipy.linalg.eigh(self.K.toarray(), self.M.toarray())
        return w, v

    def _filter_eigenvalues_0(self, w, v):
        """
        Null filter, place holder for the filter dictionary.
        """
        if self.N_dof>0:
            pass
        return w, v

    def _filter_eigenvalues_1(self, w, v):
        """
        Basic filter removing duplicate eigenvalues in the lowest frequencies
        and mesh related eigen vectors
        """
        w, v = self._filter_low_eigenvalues(w, v)
        w, v = self._filter_mesh_modes(w, v)

        return w, v

    def _filter_eigenvalues_2(self, w, v):
        """
        A static method to filter the duplicate eigenvalues and vectors obtained
        from `scipy.linalg.solve`

        Parameters:
        ___________________
        w : eigenvalues (array)
        v : eigenvectors (matrix)
        atol : Absolute tolerance for duplicate eigenvalues
        """
        atol = 32
        N_dof = self.N_dof
        # (rad/s)^2 -> Hz
        w = np.sqrt(w) / (2 * np.pi)

        w_out = []  # Output frequencies
        indexes = []  # Keep the indexes to filter the eigenvectors
        i = 0
        # Iterate over the frequency vector
        while i < w.shape[0] - 2:
            j = 1
            # Iterate while the frequencies are duplicates
            while np.isclose(w[i], w[i + j], atol=atol):
                j += 1
                if i+j == w.shape[0]:
                    break
            # If the frequency is relevant, add to output and keep index
            if w[i] > 1.1:
                # Check also if the eigen vector is not too skewed (removing mesh modes)
                if (np.abs(v[i].max()) / np.abs(v[i]).mean()) < 5:
                    w_out.append(w[i])
                    indexes.append(i)
            # Skip to the next non-duplicate frequency
            i += j

        return (np.array(w_out)*2*np.pi) ** 2, v[indexes]

    def _filter_mesh_modes(self, w, v):
        """
        Filters out the mesh modes by computing the eigenvector standard
        deviation in the component of interest
        :param w: eigen values (Hz)
        :param v: eigen vectors
        :return: w, v (filtered)
        """
        # Take the Z component for plate models
        if self.N_dof == 6:
            # Where the standard deviation of the Z component is not zero
            indexes = ~np.isclose([np.std(vi[2::6]) for vi in v], 0)
            return w[indexes], v[indexes]

    def _filter_low_eigenvalues(self, w, v):
        """
        Basic filter removing duplicate eigenvalues in the lowest frequencies
        """
        if self.N_dof > 1:
            pass
        # Convert to Hz
        w = np.sqrt(w) / (2 * np.pi)
        # Filter the low range duplicates
        idx = np.where(~np.isclose(w[:-1], w[1:], 10e-5))[0][1]
        # Convert to (rad/s)^2
        w = (w * (2 * np.pi)) ** 2

        return w[idx:], v[idx:]

    """
    Boundary Condition Functions
    """

    def add_fixed_domain(self, domain, ddls=None):
        """
        A a fixed domain to the FEM Problem
        :param domain: a domain function with the form domain(x, y) = True or False
        :param ddls: the dofs to be fixed
            if the problem is plane stress : ddls = [u_x, u_y]
            if the problem is complete plate : ddls = [u_x, u_y, u_z, theta_x, theta_y, theta_z]
        :return: None
        """
        if ddls is None:
            ddls = np.arange(self.N_dof)
            # N_ddls = 2 : [0, 1]
            # N_ddls = 6 : [0, 1, 2, 3, 4, 5]
        self.fixed_domains.append(domain)
        self.fixed_ddls.append(ddls)

    def add_forces(self, force, domain):
        """
        creates the force vector F to solve the form : KU = F
        :param force:
            if plane stress : force = [Fx, Fy]
            if complete plate : force = [Fx, Fy, Fz, Mx, My, Mz]
        :param domain:
            a domain function where the force is applied with the form `domain(x, y) = True or False`
        :return: None
        """
        N_ddls = self.N_dof
        F = np.zeros(N_ddls * self.mesh.N_nodes)
        self.forces.append(force)
        self.force_domains.append(domain)
        force_nodes = self.mesh.domain_nodes(domain)

        if len(force_nodes) > 0:
            force = np.array(force) / len(force_nodes)  # Distribute the force components over the domain nodes
            self.fr.append(force)
            for node in force_nodes:
                for ddl in np.arange(0, N_ddls):
                    F[N_ddls * node + ddl] = force[ddl]
        else:
            print('invalid force domain')

        self.F = self.F + F

    """
    Global Matrix Assembly Methods
    """

    def _assemble_K(self, X=None, p=None):
        """
        sparse matrix assembly
        Parameters
        ----------
        X Density values for the elements
        p penalty factor
        """
        # Compute the data vector of the sparse matrix
        if self.mesh.structured:
            if self.coating:
                # TODO : Test
                data = self._K_structured_mesh_data_coating(X, p)
            elif not self.coating:
                data = self._K_structured_mesh_data_base(X, p)

        elif not self.mesh.structured:
            if self.coating:
                pass  # TODO
            elif not self.coating:
                data = self._K_unstructured_mesh_data_base(X, p)

        data = self._apply_boundary_conditions_to_matrix_data(data)

        self.K = scipy.sparse.csr_matrix((data, (self.mesh.rows, self.mesh.cols)),
                                         shape=(self.N_dof * self.mesh.N_nodes, self.N_dof * self.mesh.N_nodes))
        self.K.sum_duplicates()

    def _assemble_M(self, X=None, p=None):
        """
        sparse mass matrix assembly
        Parameters
        ----------
        X Density values for the elements
        p penalty factor
        """

        # Compute the data vector of the sparse matrix
        if self.mesh.structured:
            if self.coating:
                pass
                # TODO : Coating problem mass structured mass matrix data

            elif not self.coating:
                data = self._M_structured_mesh_data_base(X, p)

        elif not self.mesh.structured:
            if self.coating:
                pass  # TODO : Coating problem mass unstructured mass matrix data
            elif not self.coating:
                data = self._M_unstructured_mesh_data_base(X, p)

        data = self._apply_boundary_conditions_to_matrix_data(data)

        self.M = scipy.sparse.csr_matrix((data, (self.mesh.rows, self.mesh.cols)),
                                         shape=(self.N_dof * self.mesh.N_nodes, self.N_dof * self.mesh.N_nodes))
        self.M.sum_duplicates()

    def _assemble_F(self):
        """
        Applies the boundary conditions to the Force vector
        """
        if self.fixed_ddls:
            for domain, ddls in zip(self.fixed_domains, self.fixed_ddls):
                fixed_nodes = self.mesh.domain_nodes(domain)
                for node in fixed_nodes:
                    self.F[np.full(len(ddls), node * self.N_dof) + np.array(ddls)] = 0

    """
    K Matrix Assembly Core Methods
    """

    def _K_structured_mesh_data_base(self, X=None, p=None):
        """
        Computes the global K matrix data vector for an structured mesh
        with only a base material

        Returns
        -------
        d : the data vector to build the global stiffness matrix
        """

        # Create an element from the mesh coordinates
        element = self.mesh.element

        if X is None:
            X = 1

        else:
            X = X[self.mesh.contains[0]]
            X = np.repeat(X, element.size ** 2) ** p

        # Element stiffness matrix data
        Ke = element.Ke(self.C_A, self.C_D, self.C_G)
        data = np.tile(Ke.reshape(-1), self.mesh.N_ele) * X

        return data

    def _K_structured_mesh_data_coating(self, X=None, p=None):
        """
        Computes the K data vector for an optimized coating material
        Parameters
        ----------
        X : density vector
        p : penalty exponent

        Returns
        -------
        data : the sparse matrix data vector
        """

        # Get the element instance from the mesh
        element = self.mesh.element

        # Adjust the X vector
        if X is None:
            X = 1
        else:
            X = X[self.mesh.contains[0]]
            X = np.repeat(X, element.size ** 2) ** p

        # Element stiffness matrix
        Ke_base = element.Ke(self.C_A, self.C_D, self.C_G)
        data_base = np.tile(Ke_base.reshape(-1), self.mesh.N_ele)
        Ke_coat = element.Ke(self.coat_C_A, self.coat_C_D, self.coat_C_G)
        data_coat = np.tile(Ke_coat.reshape(-1), self.mesh.N_ele)
        data = data_base + data_coat * X

        return data

    def _K_unstructured_mesh_data_base(self, X=None, p=None):
        """
        Computes the global K matrix data vector for an unstructured mesh
        for the base material case

        Returns
        -------
        d : the data vector to build the global stiffness matrix
        """
        # Empty data and element stiffness array
        self.element_Ke = {'triangle': [], 'quad': []}
        data = np.array([])

        # Case for a X vector
        if X:
            # Loop over the cell types in the mesh
            for cell_type in self.mesh.contains:
                # Loop over the cells
                for cell, Xe in zip(self.mesh.cells[cell_type], X[cell_type]):
                    # Create the element
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element stiffness matrix
                    Ke = element.Ke(self.C_A, self.C_D, self.C_G) * (Xe ** p)
                    self.element_Ke[cell_type].append(Ke)
                    data = np.append(data, Ke.reshape(-1))

        # Case for no X vector
        else:
            # Loop over the cell types in the mesh
            for cell_type in self.mesh.contains:
                # Loop over the cells
                for cell in self.mesh.cells[cell_type]:
                    # Create the element
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element stiffness matrix
                    Ke = element.Ke(self.C_A, self.C_D, self.C_G)
                    self.element_Ke[cell_type].append(Ke)
                    data = np.append(data, Ke.reshape(-1))

        data = np.append(data, np.ones(self.mesh.empty_nodes.shape[0] * self.N_dof))
        return data

    """
    M Matrix Assembly Core Methods
    """

    def _M_structured_mesh_data_base(self, X=None, p=None):
        """
        Creates the data vector to form the global mass matrix from
        a structured mesh where all the elements have the same dimensions
        """

        # Create an element from the mesh coordinates
        element = self.mesh.element

        if X is None:
            X = 1

        else:
            X = X[self.mesh.contains[0]]
            X = np.repeat(X, element.size ** 2) ** p

        # Element stiffness matrix data
        Me = element.Me(self.materials[0], self.ho)
        data = np.tile(Me.reshape(-1), self.mesh.N_ele) * X

        return data

    def _M_unstructured_mesh_data_base(self, X=None, p=None):
        """
        Method creating the global mass matrix data vector from
        the element mass matrices for an unstructured mesh
        """
        self.element_Me = {'triangle': [], 'quad': []}
        data = np.array([])

        # Case for a X vector
        if X:
            # Loop over the cell types in the mesh
            for cell_type in self.mesh.contains:
                # Loop over the cells
                for cell, Xe in zip(self.mesh.cells[cell_type], X[cell_type]):
                    # Create the element
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element stiffness matrix
                    Me = element.Me(self.materials[0], self.ho) * (Xe ** p)
                    self.element_Me[cell_type].append(Me)
                    data = np.append(data, Me.reshape(-1))

        else:
            # Loop over the cell types in the mesh
            for cell_type in self.mesh.contains:
                # Loop over the cells
                for cell in self.mesh.cells[cell_type]:
                    # Create the element
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element stiffness matrix
                    Me = element.Me(self.materials[0], self.ho)
                    self.element_Me[cell_type].append(Me)
                    data = np.append(data, Me.reshape(-1))

        data = np.append(data, np.ones(self.mesh.empty_nodes.shape[0] * self.N_dof))
        return data

    """
    Method for boundary conditions
    """

    def _apply_boundary_conditions_to_matrix_data(self, data):
        """
        Applies the boundary conditions to the data vector
        Parameters
        """
        if self.fixed_ddls:
            # Fixing the dofs
            diag_indexes = np.array([])
            for domain, ddls in zip(self.fixed_domains, self.fixed_ddls):
                fixed_nodes = self.mesh.domain_nodes(domain)
                self.fixed_nodes = np.append(self.fixed_nodes, fixed_nodes)
                for node in fixed_nodes:
                    for ddl in ddls:
                        fixed_rows = self.mesh.rows == node * self.N_dof + ddl
                        fixed_cols = self.mesh.cols == node * self.N_dof + ddl
                        fixed_diag = np.logical_and(fixed_rows, fixed_cols)
                        data[fixed_rows] = 0
                        data[fixed_cols] = 0
                        diag_indexes = np.append(diag_indexes, np.nonzero(fixed_diag))

            # use unique for the diagonal so the value == 1
            diag_indexes = np.hstack(diag_indexes).reshape(-1).astype(int)
            _, index1 = np.unique(self.mesh.cols[diag_indexes], return_index=True)
            _, index2 = np.unique(self.mesh.rows[diag_indexes], return_index=True)
            if (index1 == index2).all():
                diag_indexes = diag_indexes[index1]
                data[diag_indexes] = 1

        return data


class TOPOPT_Problem(object):
    """
    Supported Topology Optimisation problems:
    'simple'
    'coating

    Supported Topology Optimisation methods:
    'SIMP'

    Supported Topology optimisation kinds:
    'compliance
    """

    def __init__(self, Problem, volfrac=0.5, penal=3, rmin=1.5):
        """
        Constructor for the Topology Optimisation Problem
        """
        FEM_SOLVERS = {'displacement': self._SIMP_displacement_solver,
                       'modal': self._SIMP_modal_solver}
        DISPLACEMENT_OBJECTIVE_FUNS = {'solid': self._compliance_objective_function,
                                       'coating': self._coating_compliance_objective_function}
        VIBRATION_OBJECTIVE_FUNS = {'solid': self._max_eigs_objective_function()}
        OBJECTIVE_FUNCTIONS = {'displacement': DISPLACEMENT_OBJECTIVE_FUNS,
                               'vibration': VIBRATION_OBJECTIVE_FUNS}

        # store the problem parameters
        self.FEM = Problem
        self.mesh = Problem.mesh
        self.mesh.compute_element_centers()
        self.method = 'SIMP'
        self.FEM_solver = FEM_SOLVERS[self.FEM.physics]
        kind = 'coating' * self.FEM.coating + 'solid' * (not self.FEM.coating)
        self.objective_function = OBJECTIVE_FUNCTIONS[self.FEM.physics][kind]

        # define the TOPOPT parameters
        if self.method == 'SIMP':
            self.f = volfrac
            self.rmin = rmin
            self.p = penal
            self.X = {key: np.ones(self.mesh.cells[key].shape[0]) * volfrac for key in self.mesh.contains}

    """
    Plotting
    """

    def plot(self):
        self.FEM.plot()

    """
    Solver
    """

    def solve(self, converge=0.01, max_loops=100, plot=True, save=True):

        if self.method == 'SIMP':
            change = 1
            self.loop = 0

            while (change > converge) & (self.loop < max_loops):

                # Iterate
                self.loop += 1

                # Iteration
                X_old = self.X
                self.U = self.FEM_solver(X_old)
                self.c, self.dc = self.objective_function(X_old)
                self.filter_sensibility(X_old)
                self.X = self.get_new_x(X_old)
                X1 = np.array(list(self.X.values())).flatten()
                X2 = np.array(list(X_old.values())).flatten()
                change = np.max(np.abs(X1 - X2))

                # adding the results to the mesh
                X_key = 'X{}'.format(self.loop)
                self.mesh.cell_data[X_key] = self.X
                self.mesh.point_data['U{}'.format(self.loop)] = self.U

                # animation
                if plot:
                    clear_output(wait=True)
                    N = int(np.sqrt(self.mesh.N_ele))
                    try:
                        X_plot = self.mesh.cell_data[X_key]
                        X_plot = np.hstack([X_plot[cell_type] for cell_type in self.mesh.contains])
                        X_plot = X_plot.reshape(N, N)
                        plt.imshow(np.flip(X_plot, 0), cmap='Greys')
                    except ValueError:
                        self.mesh.plot.cell_data(X_key)
                    ax = plt.gca()
                    title = "Iteration : " + str(self.loop) + ', variation : ' + str(np.around(change * 100, 1))
                    ax.set_title(title)
                    plt.pause(0.1)
                else:
                    clear_output(wait=True)
                    title = "Iteration : " + str(self.loop) + ', variation : ' + str(np.around(change * 100, 1))
                    print(title)

            # Save the best X
            self.mesh.cell_data['X'] = self.X
            if save:
                # Try saving the file in results
                try:
                    filename = 'results/TOPOPT/arrays/topopt_' + FEMOL.utils.unique_time_string()
                    np.save(filename, np.array(self.X))
                # If it does not work save it here
                except FileNotFoundError:
                    filename = 'topopt_' + FEMOL.utils.unique_time_string()
                    np.save(filename, np.array(self.X))

            return self.mesh

    """
    FEM Methods
    """

    def _SIMP_displacement_solver(self, X):
        """
        Solved the FEM  displacement problem with the current element density values
        """
        self.FEM.assemble('K', X=X, p=self.p)
        U = self.FEM.solve(verbose=False).U
        return U

    def _SIMP_modal_solver(self, X, v_ref):
        """
        Solve the FEM modal problem with the current element density values
        """
        self.FEM.assemble('K', X=X, p=self.p)
        self.FEM.assemble('M', X=X, q=self.q)
        w, v = self.FEM.solve(verbose=False)

        for i, vi in enumerate(v):
            stats = FEMOL.utils.MAC(vi, v_ref)
            if np.isclose(stats, 1):
                wj = w[i]
                vj = vi
                break

        return wj**2, vj

    """
    Objective function methods
    """

    def _coating_compliance_objective_function(self, X):
        """
        Only works for structured mesh
        """

        if self.mesh.structured:
            c = 0
            dc = np.array([])
            Ke_base = self.mesh.element.Ke(self.FEM.C_A, self.FEM.C_D, self.FEM.C_G)
            Ke_coat = self.mesh.element.Ke(self.FEM.coat_C_A, self.FEM.coat_C_D, self.FEM.coat_C_G)

            # Loop over every element
            for ele, xe in zip(self.mesh.cells[self.mesh.contains[0]], X[self.mesh.contains[0]]):
                Ue = np.array([])

                # Get the displacement from the four nodes
                for node in ele:
                    Ue = np.append(Ue, self.U[self.FEM.N_dof * node:self.FEM.N_dof * node + self.FEM.N_dof])

                # Objective function
                Ke = Ke_base + Ke_coat * (xe ** self.p)
                c += Ue.transpose() @ Ke @ Ue

                # Sensibility to the Objective function
                dc = np.append(dc, -self.p * xe ** (self.p - 1) * Ue.T @ Ke_coat @ Ue)

            return c, dc

        # TODO : Fix
        elif not self.mesh.structured:
            # initiate the objective function values
            c = 0
            dc = np.array([])

            # Loop over every element nodes, density, element stiffness matrix
            for cell_type in self.mesh.contains:
                for ele, xe, Ke in zip(self.mesh.cells[cell_type], X[cell_type], self.FEM.element_Ke[cell_type]):
                    # Empty element displacement array
                    Ue = np.array([])

                    # Get the displacement from the element nodes
                    for node in ele:
                        I1 = int(self.FEM.N_dof * node)
                        I2 = int(self.FEM.N_dof * node + self.FEM.N_dof)
                        Ue = np.append(Ue, self.U[I1:I2])

                    # Objective function
                    c += xe ** self.p * Ue.transpose() @ Ke @ Ue

                    # Sensibility to the Objective function
                    dc = np.append(dc, -self.p * xe ** (self.p - 1) * Ue.transpose() @ Ke @ Ue)

            return c, dc

    def _compliance_objective_function(self, X):
        """
        Minimze compliance objective function for a solid part problem
        """
        if self.mesh.structured:
            c = 0
            dc = np.array([])
            Ke = self.mesh.element.Ke(self.FEM.C_A, self.FEM.C_D, self.FEM.C_G)

            # Loop over every element
            for ele, xe in zip(self.mesh.cells[self.mesh.contains[0]], X[self.mesh.contains[0]]):
                Ue = np.array([])

                # Get the displacement from the four nodes
                for node in ele:
                    Ue = np.append(Ue, self.U[self.FEM.N_dof * node:self.FEM.N_dof * node + self.FEM.N_dof])

                # Objective function
                c += xe ** self.p * Ue.transpose() @ Ke @ Ue

                # Sensibility to the Objective function
                dc = np.append(dc, -self.p * xe ** (self.p - 1) * Ue.transpose() @ Ke @ Ue)

            return c, dc

        elif not self.mesh.structured:
            # initiate the objective function values
            c = 0
            dc = np.array([])

            # Loop over every element nodes, density, element stiffness matrix
            for cell_type in self.mesh.contains:
                for ele, xe, Ke in zip(self.mesh.cells[cell_type], X[cell_type], self.FEM.element_Ke[cell_type]):
                    # Empty element displacement array
                    Ue = np.array([])

                    # Get the displacement from the element nodes
                    for node in ele:
                        I1 = int(self.FEM.N_dof * node)
                        I2 = int(self.FEM.N_dof * node + self.FEM.N_dof)
                        Ue = np.append(Ue, self.U[I1:I2])

                    # Objective function
                    c += xe ** self.p * Ue.transpose() @ Ke @ Ue

                    # Sensibility to the Objective function
                    dc = np.append(dc, -self.p * xe ** (self.p - 1) * Ue.transpose() @ Ke @ Ue)

            return c, dc

    def _max_eigs_objective_function(self, X):
        """
        Maximize eigenvalue objective function
        """

        # If the problem mesh is structured
        if self.mesh.structured:
            dlmbd = []
            # Constant element matrices
            Ke = self.mesh.element.Ke(self.FEM.C_A, self.FEM.C_D, self.FEM.C_G)
            Me = self.mesh.element.Me(self.FEM.materials[0], self.FEM.ho)

            # Loop over every element
            for ele, xe in zip(self.mesh.cells[self.mesh.contains[0]], X[self.mesh.contains[0]]):
                Ve = np.array([])

                # Get the displacement from the four nodes
                for node in ele:
                    Ve = np.append(Ve, self.v[self.FEM.N_dof * node:self.FEM.N_dof * node + self.FEM.N_dof])

                # Objective function

                # Sensibility to the Objective function
                dlmbd.append(Ve.T @ (self.p * xe ** (self.p- 1) * Ke
                                     - self.lmbd * self.q * xe ** (self.q-1) * Me) @ Ve)

            return self.lmbd, dlmbd

    """
    Filters
    """

    # TODO : Make faster (test)
    def filter_sensibility(self, X):
        """
        Returns the filtered sensitivity function for the density field X and the mesh
        :param X: Density vector
        :return: filtered sensibility vector
        """
        # empty vector for filtered dc
        dc_new = np.zeros(self.mesh.N_ele)
        # Search distance according to the average element size
        search_distance = self.rmin * self.mesh.element_size()

        all_X = np.hstack([X[cell_type] for cell_type in self.mesh.contains])

        # Iterate over every element
        for i, ele_e in enumerate(self.mesh.all_cells):

            # Create the sum of Hf weight variables
            sum_Hf = 0

            # Get the center of the element
            x_e, y_e = self.mesh.all_cell_centers[i]

            # Compute the distance between the center and all the other elements
            D = np.sqrt((self.mesh.all_cell_centers.T[0] - x_e) ** 2
                        + (self.mesh.all_cell_centers.T[1] - y_e) ** 2)

            # remove the current element
            D[i] = D.max()
            # Get the elements within the radius
            neighbouring_elements_numbers = np.nonzero(D <= search_distance)[0]

            for ele_f, j in zip(self.mesh.all_cells[neighbouring_elements_numbers], neighbouring_elements_numbers):
                # Get the center of the current neighbouring element
                x_f, y_f = self.mesh.all_cell_centers[j]

                # compute the distance between the element f and element e
                dist = ((x_f - x_e) ** 2 + (y_f - y_e) ** 2) ** 0.5

                # compute the weight Hf
                Hf = self.rmin - dist / self.mesh.element_size()

                # Add the weight Hf to the sum of the weights
                sum_Hf += np.max([0, Hf])

                # Add the left hand size summation term of equation (5) to the new sensibility
                dc_new[i] += np.max([0, Hf]) * all_X[j] * self.dc[j]

            dc_new[i] /= (all_X[i] * sum_Hf)

        self.dc = dc_new

    """
    Iterators
    """

    def get_new_x(self, X):
        l1 = 0
        l2 = 100000
        move = 0.5
        # Flatten the X array
        X = np.hstack([X[cell_type] for cell_type in self.mesh.contains])

        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l1 + l2)

            X1 = X + move
            X2 = X * (-self.dc / lmid) ** 0.3

            X_new = np.min([X1, X2], axis=0)
            X_new = np.min([np.ones(self.mesh.N_ele), X_new], axis=0)
            X_new = np.max([X - move, X_new], axis=0)
            # Remove the values lower than the threshold
            X_new = np.max([0.001 * np.ones(self.mesh.N_ele), X_new], axis=0)

            if hasattr(self, 'solid_domain'):
                X_new = self.solid(self.solid_domain, X_new)

            if hasattr(self, 'void_domain'):
                X_new = self.void(self.void_domain, X_new)

            if (np.sum(X_new) - self.f * self.mesh.N_ele) > 0:
                l1 = lmid
            else:
                l2 = lmid

            X_new = np.split(X_new, [self.mesh.cells[self.mesh.contains[0]].shape[0]])
            X_new = {key: Xi for (key, Xi) in zip(self.mesh.contains, X_new)}

        return X_new

    """
    Domain definition
    """

    def solid(self, domain, X):
        for i, element in enumerate(self.mesh.cells[self.mesh.contains[0]]):
            if np.array([domain(*coord) for coord in self.mesh.points[element]]).all():
                X[i] = 1
        return X

    def void(self, domain, X):
        for i, element in enumerate(self.mesh.cells[self.mesh.contains[0]]):
            if np.array([domain(*coord) for coord in self.mesh.points[element]]).all():
                X[i] = 0.001
        return X

    def define_solid_domain(self, domain):
        self.solid_domain = domain

    def define_void_domain(self, domain):
        self.void_domain = domain

