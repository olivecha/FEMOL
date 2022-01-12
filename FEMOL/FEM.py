# FEMOL Imports
from FEMOL.mesh import Mesh
from FEMOL.laminate import Layup
# Numpy import
import numpy as np
# SciPy
import scipy.linalg
import scipy.stats
import scipy.sparse
import scipy.sparse.linalg
# Matplotlib
import matplotlib.pyplot as plt
# Python
import time

__all__ = ['FEM_Problem', 'Mesh', 'Layup', ]

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

    def __init__(self, physics, model, mesh):
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
        self.layups = []
        self.F = np.zeros(self.N_dof * self.mesh.N_nodes)

    def plot(self):
        # TODO : plot forces for 6 dof
        """
        Plots the current FEM Problem, (boundary conditions and applied forces)
        """

        fixed_dof_label_dict = {(0,): 'x displacement', (1,): 'y displacement', (2,): 'z displacement',
                                (3,): 'x rotation', (4,): 'y rotation', (0, 1): 'x-y displacement',
                                (2, 3): 'supported x', (2, 4): 'supported y', (0, 1, 2, 3, 4, 5): 'clamped',}

        single_dof_dict = {0: 'X', 1:'Y', 2:'Z', 3:'Tx', 4:'Ty', 5:'Tz'}

        fixed_dof_color_dict = {(0,): '#FF716A', (1,): '#FFBB6A', (2,): '#99FFA9',
                                (3,): '#A0FF6A', (4,): '#6AFFDF', (0, 1): '#6A75FF',
                                (2, 3): '#6A75FF', (2, 4): '#D26AFF', (0, 1, 2, 3, 4, 5): '#FF3A3A', }

        single_dof_color = '#D8FF5E'

        # Display the mesh
        self.mesh.display()

        plt.title('FEM problem')

        # Plot the fixed nodes and their degrees of freedom
        for domain, ddls in zip(self.fixed_domains, self.fixed_ddls):
            nodes = self.mesh.domain_nodes(domain)
            try :
                label = fixed_dof_label_dict[tuple(ddls)]
                color = fixed_dof_color_dict[tuple(ddls)]
            except KeyError:
                label = 'DOFs : ' + ' '.join([single_dof_dict[d] for d in ddls])
                color = single_dof_color

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
                self.layups.append(layup)

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

    def assemble(self, which, X=None, p=None, q=None):
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
            self._assemble_M(X=X, q=q)

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
        # Remove Nan values and 0/negative values
        v = v[~np.isnan(w)]
        w = w[~np.isnan(w)]
        v = v[~np.isclose(w, 0)]
        w = w[~np.isclose(w, 0)]
        v = v[w > 0]
        w = w[w > 0]

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
        try:
            w, v = scipy.linalg.eigh(self.K.toarray(), self.M.toarray())
        except scipy.linalg.LinAlgError:
            w, v = scipy.linalg.eig(self.K.toarray(), self.M.toarray())
        return w, v

    def _filter_eigenvalues_0(self, w, v):
        """
        Null filter, place holder for the filter dictionary.
        """
        if self.N_dof > 0:
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
            #self.fr.append(force)
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
                data = self._K_structured_mesh_data_coating(X, p)

            elif not self.coating:
                data = self._K_structured_mesh_data_base(X, p)

        elif not self.mesh.structured:
            if self.coating:
                data = self._K_unstructured_mesh_data_coating(X, p)

            elif not self.coating:
                data = self._K_unstructured_mesh_data_base(X, p)

        data = self._apply_boundary_conditions_to_matrix_data(data)

        self.K = scipy.sparse.csr_matrix((data, (self.mesh.rows, self.mesh.cols)),
                                         shape=(self.N_dof * self.mesh.N_nodes, self.N_dof * self.mesh.N_nodes))
        self.K.sum_duplicates()

    def _assemble_M(self, X=None, q=None):
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
                data = self._M_structured_mesh_data_coat(X=X, q=q)

            elif not self.coating:
                data = self._M_structured_mesh_data_base(X=X, q=q)

        elif not self.mesh.structured:
            if self.coating:
                data = self._M_unstructured_mesh_data_coat(X=X, q=q)

            elif not self.coating:
                data = self._M_unstructured_mesh_data_base(X=X, q=q)

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

        #data = np.append(data, np.ones(self.mesh.empty_nodes.shape[0] * self.N_dof))
        return data

    def _K_unstructured_mesh_data_coating(self, X=None, p=None):
        """
               Computes the global K matrix data vector for an unstructured mesh
               for the base material case

               Returns
               -------
               d : the data vector to build the global stiffness matrix
               """
        # Empty data and element stiffness array
        self.element_Ke_base = {'triangle': [], 'quad': []}
        self.element_Ke_coat = {'triangle': [], 'quad': []}
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
                    Ke_base = element.Ke(self.C_A, self.C_D, self.C_G)
                    Ke_coat = element.Ke(self.coat_C_A, self.coat_C_D, self.coat_C_G) * (Xe ** p)
                    self.element_Ke_base[cell_type].append(Ke_base)
                    self.element_Ke_coat[cell_type].append(Ke_coat)
                    Ke = Ke_base.reshape(-1) + Ke_coat.reshape(-1)
                    data = np.append(data, Ke)

        # Case for no X vector
        else:
            # Loop over the cell types in the mesh
            for cell_type in self.mesh.contains:
                # Loop over the cells
                for cell in self.mesh.cells[cell_type]:
                    # Create the element
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element stiffness matrix
                    Ke_base = element.Ke(self.C_A, self.C_D, self.C_G)
                    Ke_coat = element.Ke(self.coat_C_A, self.coat_C_D, self.coat_C_G)
                    self.element_Ke_base[cell_type].append(Ke_base)
                    self.element_Ke_coat[cell_type].append(Ke_coat)
                    Ke = Ke_base.flatten() + Ke_coat.flatten()
                    data = np.append(data, Ke)

        return data

    """
    M Matrix Assembly Core Methods
    """

    def _M_structured_mesh_data_base(self, X=None, q=None):
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
            X = np.repeat(X, element.size ** 2) ** q

        # Element stiffness matrix data
        Me = element.Me(self.materials[0], self.ho)
        data = np.tile(Me.reshape(-1), self.mesh.N_ele) * X

        return data

    def _M_unstructured_mesh_data_base(self, X=None, q=None):
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
                    Me = element.Me(self.materials[0], self.ho) * (Xe ** q)
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

        #data = np.append(data, np.ones(self.mesh.empty_nodes.shape[0] * self.N_dof))
        return data

    def _M_structured_mesh_data_coat(self, X=None, q=None):
        """
        Method computing the M global matrix data vector for
        A structured mesh and a coating optimization problem

        :param X: Density vector
        :param q: Penalization exponent
        :return: None
        """
        # Create an element from the mesh coordinates
        element = self.mesh.element

        if X is None:
            X = 1

        else:
            X = X[self.mesh.contains[0]]
            X = np.repeat(X, element.size ** 2) ** q

        # Element stiffness matrix data
        Me_base = element.Me(self.materials[0], self.ho)
        Me_coat = element.Me(self.materials[1], self.coat_ho)
        Me = np.tile(Me_base.flatten(), self.mesh.N_ele) + np.tile(Me_coat.flatten(), self.mesh.N_ele) * X
        data = Me

        return data

    def _M_unstructured_mesh_data_coat(self, X=None, q=None):
        """
        Method computing the data vector of the sparse
        mass matrix corresponding to the coating topology
        optimization problem
        :param X: Density vector
        :param q: Penalization exponent
        """
        self.element_Me_base = {'triangle': [], 'quad': []}
        self.element_Me_coat = {'triangle': [], 'quad': []}
        data = np.array([])

        # If a density vector is input
        if X:
            # For each type of cell in the mesh (tri, quad)
            for cell_type in self.mesh.contains:
                # For each cell corresponding to a cell type
                for cell , Xe in zip(self.mesh.cells[cell_type], X[cell_type]):
                    # Create an element
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element mass matrices
                    Me_base = element.Me(self.materials[0], self.ho)
                    Me_coat = element.Me(self.materials[1], self.coat_ho)
                    # Store as class attributes
                    self.element_Me_base[cell_type].append(Me_base)
                    self.element_Me_coat[cell_type].append(Me_coat)
                    # Compute the actual element mass matrix and add to sparse matrix data
                    Me = (Me_base + Me_coat  * Xe ** q).flatten()
                    data = np.append(data, Me)
        else:
            # For each cell type in the mesh
            for cell_type in self.mesh.contains:
                # For each cell and Xe of a single cell type
                for cell in self.mesh.cells[cell_type]:
                    # Create an element instance
                    element = self.mesh.ElementClasses[cell_type](self.mesh.points[cell], self.N_dof)
                    # Compute the element mass matrices
                    Me_base = element.Me(self.materials[0], self.ho)
                    Me_coat = element.Me(self.materials[1], self.coat_ho)
                    # Store the in the attributes
                    self.element_Me_base[cell_type].append(Me_base)
                    self.element_Me_coat[cell_type].append(Me_coat)
                    # Compute the complete Me
                    Me = (Me_base + Me_coat).flatten()
                    # add to data
                    data = np.append(data, Me)

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
