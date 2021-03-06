import FEMOL.utils
import FEMOL.elements
# Numpy
import numpy as np
# Scipy
from scipy.interpolate import interp1d
# Matplotlib
import matplotlib.pyplot as plt
# Ipython
from IPython.display import clear_output

__all__ = ['SIMP_VIBE']


class SIMP_VIBE(object):
    """
    Solid Isotropic Material with Penalization method
    for topology optimization of the modal behavior
    _________________________________________________
    Supported objective functions:
    'max eig': Fundamental eigenvalue maximization
    """

    def load_reference_guitar_modes(self, lcar=None, sym=True):
        """
        Method to load the reference guitar modes into the current SIMP problem
        :param sym:
        :param lcar: mesh characteristic length
        :return: dict of reference vectors
        """
        root = 'Results/guitar_modes/'
        modes = ['T11', 'T21', 'T12', 'T31']
        fnames = [root+'guitar' + '_sym'*sym + f'_mode_{m}_lcar{str(lcar)[-2:]}.npy' for m in modes]
        vecs = [np.load(f) for f in fnames]
        return vecs

    def __init__(self, Problem, volfrac=0.5, p=3, q=1, rmin=1.5, FEM_solver_type='fast', saving_root=None):
        """
        Constructor for the Topology Optimisation Problem
        Problem: FEMOL FEM Problem instance
        volfrac: Allowed volume fraction
        p: Stiffness matrix penalization factor
        q: Mass matrix penalization factor
        rmin: Minimum sensibilty filter radius
        objective: Modal behavior objective function
        """
        MAX_EIG = {'solid': self._solid_max_eigs_objective_function,
                   'coating': self._coating_max_eigs_objective_function}

        # store the problem parameters
        self.FEM = Problem
        self.mesh = Problem.mesh
        self.mesh.compute_element_centers()
        if FEM_solver_type == 'fast':
            self.FEM_solver = self._SIMP_fast_modal_solver
        elif FEM_solver_type == 'guitar':
            self.guit_ref_vecs = self.load_reference_guitar_modes(self.mesh.lcar, self.mesh.sym)
            self.FEM_solver = self._SIMP_fast_modal_guitar_solver
            self.guit_vecs = []
            self.guit_freqs = []
        kind = 'coating' * self.FEM.coating + 'solid' * (not self.FEM.coating)
        self.objective_function = MAX_EIG[kind]

        # mesh element area to enforce density constraint
        self.element_areas = np.hstack([self.mesh.areas[cell_type] for cell_type in self.mesh.contains])
        self.mesh_area = np.sum(self.element_areas)

        # define the TOM parameters
        self.f = volfrac
        self.rmin = rmin
        self.p = p
        self.q = q
        self.X = {key: np.ones(self.mesh.cells[key].shape[0]) * volfrac for key in self.mesh.contains}
        self.lmbds = []
        self.all_lmbds = []
        self.eigen_vectors = []
        self.FEM_solver_used = []
        # Define the iteration variables
        self.v = 0
        self.lmbd = 0
        self.dlmbd = 0
        self.loop = 0
        self.change = None
        if saving_root is None:
            self.save_root = 'Results/_topopt_cache/'

    def solve(self, v_ref,  converge=0.01, min_iter=1, max_iter=100, plot=True,
              save=True, verbose=True, convergence_criteria='change', mesh_filename=None,
              eigvals_filename=None, eigvecs_filename=None):
        """
       SIMP Optimization solver
       :param eigvecs_filename:
       :param eigvals_filename:
       :param mesh_filename:
       :param convergence_criteria:
       :param verbose:
       :param converge: Convergence for density
       :param max_iter: Maximum
       :param plot: Plot the transient designs
       :param save: Save the result to mesh file
       :param v_ref: Reference eigenvector
       :return: mesh with density values
       """
        # Loop parameters
        self.change = 1
        self.loop = 0
        solved = False
        msh_file, eigs_file, vecs_file = self._get_saving_filenames(mesh_filename, eigvals_filename, eigvecs_filename)

        while not solved:
            # Iterate
            self.loop += 1
            # Iteration
            X_old = self.X
            self.lmbd, self.v = self.FEM_solver(X_old, v_ref=v_ref, verbose=(self.loop == 1))
            self.lmbds.append(self.lmbd)
            self.eigen_vectors.append(self.v)
            self.lmbd, self.dlmbd = self.objective_function(X_old)
            self._filter_sensibility(X_old)
            self.X = self._get_new_x(X_old)
            self.update_change(self.X, X_old)
            # Archive the previous X
            self.add_density_to_mesh()
            # Iteration information
            self.iteration_info(plot, verbose)
            # Check convergence
            solved = self.check_convergence(converge, min_iter, max_iter, convergence_criteria)
            # Save at each iteration
            if save:
                self.save_TOM_iteration(msh_file, eigs_file, vecs_file)

        return self.mesh

    def save_TOM_iteration(self, msh_file, eig_file, vec_file):
        """
        Saves the current TOM iteration
        :param msh_file: mesh file
        :param eig_file: eigenvalues file
        :param vec_file: eigenvectors file
        :return: None
        """
        # Add the core height transformation
        self.mesh = self.density_to_core_height()
        # Add the penalized density values
        self.mesh.cell_data['X_real'] = {'quad': self.mesh.cell_data['X']['quad'] ** self.p}
        # save for the current iteration
        self.mesh.save(msh_file)
        # if guitar solver
        if self.FEM_solver == self._SIMP_fast_modal_guitar_solver:
            np.save(eig_file, np.array(self.guit_freqs))
            np.save(vec_file, np.array(self.guit_vecs))
        else:
            np.save(eig_file, np.array(self.lmbds))

    def check_convergence(self, converge, min_iter, max_iter, convergence_criteria):
        """
        Method to check the convergence for the current iteration
        :param converge: convergence threshold
        :param min_iter: minimum number of iterations
        :param max_iter: maximum number of iterations
        :param convergence_criteria: convergence criteria to use ('change' or 'objective')
        :return: bool solved
        """
        solved = False
        if convergence_criteria == 'change':
            if (self.change < converge) and (self.loop > min_iter):
                solved = True
        elif convergence_criteria == 'objective':
            if np.abs(np.mean(self.lmbds[-3:-1]) - self.lmbds[-1]) < converge:
                solved = True
        if self.loop == max_iter:
            solved = True
        return solved

    def iteration_info(self, plot, verbose):
        """
        Method to output current iteration info
        :param plot: bool plot the density value
        :param verbose: bool print the iteration info
        :return: None
        """
        if plot:
            self._plot_iteration()
        if verbose:
            self.print_iteration()

    def add_density_to_mesh(self):
        """
        Method adding the current density result to the mesh
        :return: None
        """
        X_key = 'X{}'.format(self.loop - 1)
        if 'X' in self.mesh.cell_data.keys():
            self.mesh.cell_data[X_key] = self.mesh.cell_data['X']
        # Save the most recent as X
        self.mesh.cell_data['X'] = self.X
        # Save the displacement
        self.mesh.add_mode('m{}'.format(self.loop), self.v, self.FEM.N_dof)

    def update_change(self, X1, X2):
        """
        Method to update the change attribute according to two consecutive densities
        :param X1: density dict 1
        :param X2: density dict 2
        :return: None
        """
        X1 = np.array(list(X1.values())).flatten()
        X2 = np.array(list(X2.values())).flatten()
        self.change = np.max(np.abs(X1 - X2))

    def _get_saving_filenames(self, mesh_filename, eigvals_filename, eigvecs_filename):
        """ Method to get the files to save the data"""
        TOM_start_time = FEMOL.utils.unique_time_string()
        out_fnames = []
        fnames = [mesh_filename, eigvals_filename, eigvecs_filename]
        bases = ['TOM_', 'eigvals_', 'eigvecs_']
        for fname, base in zip(fnames, bases):
            if fname is None:
                fname = self.save_root + base + TOM_start_time
            else:
                fname = self.save_root + fname
            out_fnames.append(fname)
        return out_fnames

    def print_iteration(self):
        """
        Method to print the current iteration data to stdout
        :return:
        """
        info = f'Iteration : {self.loop}, Variation : {self.change}, EigenVal : {self.lmbd}'
        print(info)

    def _plot_iteration(self):
        """
        Plots the current iteration from the TOM solver
        """
        clear_output(wait=True)
        N = int(np.sqrt(self.mesh.N_ele))
        try:
            X_plot = self.mesh.cell_data['X']
            X_plot = np.hstack([X_plot[cell_type] for cell_type in self.mesh.contains])
            X_plot = X_plot.reshape(N, N)
            plt.imshow(np.flip(X_plot, 0), cmap='Greys')
        except ValueError:
            self.mesh.plot.cell_data('X')
        ax = plt.gca()
        title = "Iteration : " + str(self.loop) + ', variation : ' + str(np.around(self.change * 100, 1))
        ax.set_title(title)
        plt.pause(0.1)

    def _get_new_x(self, X):
        l1 = 0
        l2 = 100000
        move = 0.3
        # Flatten the X array
        X = np.hstack([X[cell_type] for cell_type in self.mesh.contains])
        # remove negative values
        self.dlmbd[self.dlmbd < 0] = 0

        if np.sum(self.dlmbd) < 1:
            self.dlmbd *= 1/self.dlmbd.max()
            print('Sensibility was rescaled')

    # Find the lagrange multiplier
        while (l2 - l1) > 1e-4:
            # Bijection algorithm
            lmid = 0.5 * (l1 + l2)
            # Move by an increment
            X1 = X + move
            # Multiply by the sensibility
            X2 = X * (self.dlmbd / lmid) ** 0.3
            # Take the min between move and sensibilities
            X_new = np.min([X1, X2], axis=0)
            # Remove value higher than one
            X_new = np.min([np.ones(self.mesh.N_ele), X_new], axis=0)
            # Do a negative move
            X_new = np.max([X - move, X_new], axis=0)
            # Remove the values lower than the threshold
            X_new = np.max([0.001 * np.ones(self.mesh.N_ele), X_new], axis=0)

            # Add matter where the domain is constrained to be solid
            if hasattr(self, 'solid_domain'):
                X_new = self._apply_solid_domain(self.solid_domain, X_new)
            # Remove matter where the domain is constrained to be void
            if hasattr(self, 'void_domain'):
                X_new = self._apply_void_domain(self.void_domain, X_new)
            # Do the bijection
            if self.mesh.structured:
                if (np.sum(X_new) - self.f * self.mesh.N_ele) > 0:
                    l1 = lmid
                else:
                    l2 = lmid
            else:
                if np.sum(X_new * self.element_areas)/self.mesh_area > self.f:
                    l1 = lmid
                else:
                    l2 = lmid

            # Reshape X into a cell dict
            X_new = np.split(X_new, [self.mesh.cells[self.mesh.contains[0]].shape[0]])
            X_new = {key: Xi for (key, Xi) in zip(self.mesh.contains, X_new)}

        return X_new

    def _filter_sensibility(self, X):
        """
        Returns the filtered sensitivity function for the density field X and the mesh
        :param X: Density vector
        :return: filtered sensibility vector
        """
        # empty vector for filtered dc
        dlmbd_new = np.zeros(self.mesh.N_ele)
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
                dlmbd_new[i] += np.max([0, Hf]) * all_X[j] * self.dlmbd[j]

            dlmbd_new[i] /= (all_X[i] * sum_Hf)

        self.dlmbd = dlmbd_new

    def _apply_solid_domain(self, domain, X):
        """
        Applies a constricted solid domain to the X vector
        """
        for i, element in enumerate(self.mesh.cells[self.mesh.contains[0]]):
            if np.array([domain(*coord[:2]) for coord in self.mesh.points[element]]).all():
                X[i] = 1
        return X

    def _apply_void_domain(self, domain, X):
        """
        Applies a constricted void domain to the X vector
        """
        for i, element in enumerate(self.mesh.cells[self.mesh.contains[0]]):
            if np.array([domain(*coord[:2]) for coord in self.mesh.points[element]]).all():
                X[i] = 0.001
        return X

    def _SIMP_fast_modal_solver(self, X, v_ref, verbose=True):
        """
        Solve the FEM modal problem with the current element density values
        Returns the eigenvectors uses scipy.linalg.eigh when possible
        """
        self.FEM.assemble('K', X=X, p=self.p)
        self.FEM.assemble('M', X=X, q=self.q)
        w, v = self.FEM.solve(verbose=verbose, filtre=0)
        self.all_lmbds.append(w)
        mac = [FEMOL.utils.MAC(vi, v_ref) for vi in v]
        print('Best mac match (fast solver) :', np.max(mac))
        i = np.argmax(mac)
        return w[i], v[i]

    def _SIMP_fast_modal_guitar_solver(self, X, v_ref, verbose=True):
        """
        Solve the FEM modal problem with the current element density values
        Returns the eigenvectors uses scipy.linalg.eigh when possible
        Stores the guitar modes T11, T21, T12, T31 in self.guit_vecs
        """
        self.FEM.assemble('K', X=X, p=self.p)
        self.FEM.assemble('M', X=X, q=self.q)
        w, v = self.FEM.solve(verbose=verbose, filtre=0)
        self.all_lmbds.append(w)
        # find and store all the guitar modes
        guit_idxs = [np.argmax([FEMOL.utils.MAC(vi, vref) for vi in v]) for vref in self.guit_ref_vecs]
        self.guit_vecs.append(v[guit_idxs])
        self.guit_freqs.append(w[guit_idxs])
        # find and return the reference vector
        mac = [FEMOL.utils.MAC(vi, v_ref) for vi in v]
        print('Best mac match (fast solver) :', np.max(mac))
        i = np.argmax(mac)
        return w[i], v[i]

    def _solid_max_eigs_objective_function(self, X):
        """
        Maximize eigenvalue objective function
        """
        # If the problem mesh is structured
        if self.mesh.structured:
            dlmbd = []
            # Constant element matrices
            Ke = self.mesh.element.Ke(*self.FEM.tensors)
            Me = self.mesh.element.Me(self.FEM.materials[0], self.FEM.ho)

            # Loop over every element
            for ele, xe in zip(self.mesh.cells[self.mesh.contains[0]], X[self.mesh.contains[0]]):
                Ve = np.array([])

                # Get the displacement from the four nodes
                for node in ele:
                    Ve = np.append(Ve, self.v[self.FEM.N_dof * node:self.FEM.N_dof * node + self.FEM.N_dof])

                # Objective function

                # Sensibility to the Objective function
                dlmbd.append(Ve.T @ (self.p * xe ** (self.p - 1) * Ke
                                     - self.lmbd * self.q * xe ** (self.q-1) * Me) @ Ve)

            return self.lmbd, np.array(dlmbd)
        
        # If the mesh is not structured
        if not self.mesh.structured:

            dlmbd = []
            for cell_type in self.mesh.contains:
                for ele, xe, Ke, Me in zip(self.mesh.cells[cell_type], X[cell_type],
                                           self.FEM.element_Ke[cell_type], self.FEM.element_Me[cell_type]):
                    # Empty localized eigenvector
                    Ve = np.array([])
                    # Add the component according to the element nodes
                    for node in ele:
                        Ve = np.append(Ve, self.v[self.FEM.N_dof*node:self.FEM.N_dof*node+self.FEM.N_dof])

                    dlmbd.append(Ve.T @ (self.p * xe ** (self.p - 1) * Ke
                                     - self.lmbd * self.q * xe ** (self.q-1) * Me) @ Ve)

            return self.lmbd, np.array(dlmbd)

    def _coating_max_eigs_objective_function(self, X):
        """
        Maximize eigenvalue objective function
        """
        # If the problem mesh is structured
        if self.mesh.structured:
            dlmbd = []
            # Constant element matrices
            Kc = self.mesh.element.Ke(*self.FEM.coat_tensors)
            Mc = self.mesh.element.Me(self.FEM.materials[1], self.FEM.coat_ho)

            # Loop over every element
            for ele, xe in zip(self.mesh.cells[self.mesh.contains[0]], X[self.mesh.contains[0]]):
                Ve = np.array([])

                # Get the displacement from the four nodes
                for node in ele:
                    Ve = np.append(Ve, self.v[self.FEM.N_dof * node:self.FEM.N_dof * node + self.FEM.N_dof])

                # Objective function

                # Sensibility to the Objective function
                dlmbd.append(Ve.T @ (self.p * xe ** (self.p - 1) * Kc
                                     - self.lmbd * self.q * xe ** (self.q - 1) * Mc) @ Ve)

            return self.lmbd, np.array(dlmbd)

        # If the mesh is not structured
        if not self.mesh.structured:

            dlmbd = []
            for cell_type in self.mesh.contains:
                for ele, xe, Kc, Mc in zip(self.mesh.cells[cell_type], X[cell_type],
                                           self.FEM.element_Ke_coat[cell_type], self.FEM.element_Me_coat[cell_type]):
                    # Empty localized eigenvector
                    Ve = np.array([])
                    # Add the component according to the element nodes
                    for node in ele:
                        Ve = np.append(Ve, self.v[self.FEM.N_dof * node:self.FEM.N_dof * node + self.FEM.N_dof])

                    dlmbd.append(Ve.T @ (self.p * xe ** (self.p - 1) * Kc
                                         - self.lmbd * self.q * xe ** (self.q - 1) * Mc) @ Ve)
            return self.lmbd, np.array(dlmbd)

    def density_to_core_height(self):
        """Export the density values results to core height values"""

        # Get the laminates from the FEM problem
        layup_b = self.FEM.layups[0]
        layup_c = self.FEM.layups[1]
        # Compute the maximum core height
        hc = -layup_b.zc - layup_b.hA/2 + layup_c.zc - layup_c.hA/2
        h_values = np.linspace(0, hc, 100)
        # List for the stiffness values
        D_list_1 = []
        D_list_2 = []
        for hi in h_values:
            layup_bn = FEMOL.Layup(plies=layup_b.angles, material=layup_b.mtr, symetric=False,
                                   z_core=-hi / 2 - layup_b.hA / 2)
            layup_cn = FEMOL.Layup(plies=layup_c.angles, material=layup_c.mtr, symetric=False,
                                   z_core=hi / 2 + layup_c.hA / 2)
            D_list_1.append(layup_bn.D_mat[0, 0] + layup_cn.D_mat[0, 0])
            D_list_2.append(layup_bn.D_mat[1, 1] + layup_cn.D_mat[1, 1])

        D_b_1 = FEMOL.Layup(plies=layup_b.angles, material=layup_b.mtr, symetric=False).D_mat[0, 0]
        D_b_2 = FEMOL.Layup(plies=layup_b.angles, material=layup_b.mtr, symetric=False).D_mat[1, 1]
        D_c_1 = D_list_1[-1] - D_b_1
        D_c_2 = D_list_2[-1] - D_b_2
        X1 = (np.array(D_list_1) - D_b_1) / D_c_1
        X2 = (np.array(D_list_2) - D_b_2) / D_c_2
        X_interp = np.mean([X1, X2], axis=0) ** (1 / self.p)
        # Append the first and last points
        X_interp = np.append(0, X_interp)
        X_interp = np.append(X_interp, 1)
        h_values = np.append(0, h_values)
        h_values = np.append(h_values, hc)

        # Create the interpolator
        core_height_interp = interp1d(X_interp, h_values)
        # Compute the core height values
        zcore = {}
        for key in self.mesh.cell_data['X']:
            zcore[key] = core_height_interp(self.mesh.cell_data['X'][key])
            zcore[key][zcore[key] < 0] = np.min(h_values)

        self.mesh.cell_data['zc'] = zcore

        return self.mesh
