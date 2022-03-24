import scipy.sparse.linalg

import FEMOL.utils
import FEMOL.elements
# Numpy
import numpy as np
# Scipy
from scipy.interpolate import interp1d
import scipy.sparse
# Matplotlib
import matplotlib.pyplot as plt
# Ipython
from IPython.display import clear_output
# Python
import sys
import time

__all__ = ['SIMP_COMP', 'SIMP_VIBE']


class SIMP_COMP(object):
    """
    Solid Isotropic Material with Penalization topology optimization
    for solid and coating formulations
    """
    def __init__(self, Problem, volfrac=0.5, penal=3, rmin=1.5):
        """
        Constructor for the Topology Optimisation Problem
        """
        COMPLIANCE_OBJECTIVE_FUNS = {'solid': self._solid_compliance_objective_function,
                                     'coating': self._coating_compliance_objective_function}

        # store the problem parameters
        self.FEM = Problem
        self.mesh = Problem.mesh
        self.mesh.compute_element_centers()
        self.FEM_solver = self._SIMP_displacement_solver
        kind = 'coating' * self.FEM.coating + 'solid' * (not self.FEM.coating)
        self.objective_function = COMPLIANCE_OBJECTIVE_FUNS[kind]

        # define the TOM parameters
        self.f = volfrac
        self.rmin = rmin
        self.p = penal
        self.X = {key: np.ones(self.mesh.cells[key].shape[0]) * volfrac for key in self.mesh.contains}

    def solve(self, converge=0.01, max_iter=100, plot=True, save=True):
        """
        SIMP Optimization solver
        :param converge: Convergence for density
        :param max_iter: Maximum
        :param plot: Plot the transient designs
        :param save: Save the result to mesh file
        :return: mesh with density values
        """
        # Loop parameters
        self.change = 1
        self.loop = 0
        start_time = FEMOL.utils.unique_time_string()

        while (self.change > converge) & (self.loop < max_iter):
            # Iterate
            self.loop += 1

            # Iteration
            X_old = self.X
            self.U = self.FEM_solver(X_old)
            self.c, self.dc = self.objective_function(X_old)
            self._filter_sensibility(X_old)
            self.X = self._get_new_x(X_old)
            X1 = np.array(list(self.X.values())).flatten()
            X2 = np.array(list(X_old.values())).flatten()
            self.change = np.max(np.abs(X1 - X2))

            # Archive the previous X
            X_key = 'X{}'.format(self.loop - 1)
            if 'X' in self.mesh.cell_data.keys():
                self.mesh.cell_data[X_key] = self.mesh.cell_data['X']
            # Save the most recent as X
            self.mesh.cell_data['X'] = self.X
            # Save the displacement
            self.mesh.add_mode('d{}'.format(self.loop), self.U, self.FEM.N_dof)

            # Iteration information
            if plot:
                self._plot_iteration()
            else:
                info = f"Iteration : {self.loop}, " \
                       f"variation: {np.around(self.change * 100, 1)}," \
                       f" objective : {np.abs(np.around(self.c, 3))}"
                print(info)

            if save:
                self._save_TOM_result(start_time)

        return self.mesh

    """
    Private methods
    """

    def _plot_iteration(self):
        """
        Plots the current iteration from the TOM solver
        """
        clear_output(wait=True)
        N = int(np.sqrt(self.mesh.N_ele))
        try:
            if self.mesh.structured:
                X_plot = self.mesh.cell_data['X']
                X_plot = np.hstack([X_plot[cell_type] for cell_type in self.mesh.contains])
                X_plot = np.rot90(X_plot.reshape(N, N))
                plt.imshow(np.flip(X_plot, 0), cmap='Greys')
            else:
                raise ValueError
        except ValueError:
            self.mesh.plot.cell_data('X')
        ax = plt.gca()
        title = "Iteration : " + str(self.loop) + ', variation : ' + str(np.around(self.change * 100, 1))
        ax.set_title(title)
        plt.pause(0.1)

    def _save_TOM_result(self, timestring):
        # Try saving the file in results
        try:
            filename = 'Results/_topopt_cache/TOM_' + timestring
            self.mesh.save(filename)
        # If it does not work save it here
        except FileNotFoundError:
            filename = 'TOM_' + FEMOL.utils.unique_time_string()
            self.mesh.save(filename)

    def _SIMP_displacement_solver(self, X):
        """
        Solved the FEM  displacement problem with the current element density values
        """
        self.FEM.assemble('K', X=X, p=self.p)
        U = self.FEM.solve(verbose=False).U
        return U

    def _solid_compliance_objective_function(self, X):
        """
        Minimize compliance objective function for a solid part problem
        """
        if self.mesh.structured:
            c = 0
            dc = np.array([])
            Ke = self.mesh.element.Ke(*self.FEM.tensors)

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

    def _coating_compliance_objective_function(self, X):
        """
        Only works for structured mesh
        """

        if self.mesh.structured:
            c = 0
            dc = np.array([])
            Ke_base = self.mesh.element.Ke(*self.FEM.tensors)
            Ke_coat = self.mesh.element.Ke(*self.FEM.coat_tensors)

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

        elif not self.mesh.structured:
            # initiate the objective function values
            c = 0
            dc = np.array([])

            # Loop over every element nodes, density, element stiffness matrix
            for cell_type in self.mesh.contains:
                for ele, xe, Ke_base, Ke_coat in zip(self.mesh.cells[cell_type], X[cell_type],
                                                     self.FEM.element_Ke_base[cell_type],
                                                     self.FEM.element_Ke_coat[cell_type]):
                    # Empty element displacement array
                    Ue = np.array([])
                    # Get the displacement from the element nodes
                    for node in ele:
                        I1 = int(self.FEM.N_dof * node)
                        I2 = int(self.FEM.N_dof * node + self.FEM.N_dof)
                        Ue = np.append(Ue, self.U[I1:I2])
                    # Objective function
                    Ke = Ke_base + Ke_coat * (xe ** self.p)
                    c += Ue.transpose() @ Ke @ Ue
                    # Sensibility to the Objective function
                    dc = np.append(dc, -self.p * xe ** (self.p - 1) * Ue.T @ Ke_coat @ Ue)
            return c, dc

    def _filter_sensibility(self, X):
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

    def _get_new_x(self, X):
        l1 = 0
        l2 = 100000
        move = 0.5
        # Flatten the X array
        X = np.hstack([X[cell_type] for cell_type in self.mesh.contains])

        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            X1 = X + move
            if np.any(self.dc > 0):
                X2 = X * (self.dc / lmid) ** 0.3
            else:
                X2 = X * (-self.dc / lmid) ** 0.3
            X_new = np.min([X1, X2], axis=0)
            X_new = np.min([np.ones(self.mesh.N_ele), X_new], axis=0)
            X_new = np.max([X - move, X_new], axis=0)
            # Remove the values lower than the threshold
            X_new = np.max([0.001 * np.ones(self.mesh.N_ele), X_new], axis=0)

            if hasattr(self, 'solid_domain'):
                X_new = self._apply_solid_domain(self.solid_domain, X_new)

            if hasattr(self, 'void_domain'):
                X_new = self._apply_void_domain(self.void_domain, X_new)

            if (np.sum(X_new) - self.f * self.mesh.N_ele) > 0:
                l1 = lmid
            else:
                l2 = lmid

        X_new = np.split(X_new, [self.mesh.cells[self.mesh.contains[0]].shape[0]])
        X_new = {key: Xi for (key, Xi) in zip(self.mesh.contains, X_new)}

        return X_new

    def _apply_solid_domain(self, domain, X):
        """
        Applies a constricted solid domain to the X vector
        """
        for i, element in enumerate(self.mesh.cells[self.mesh.contains[0]]):
            if np.array([domain(*coord) for coord in self.mesh.points[element]]).all():
                X[i] = 1
        return X

    def _apply_void_domain(self, domain, X):
        """
        Applies a constricted void domain to the X vector
        """
        for i, element in enumerate(self.mesh.cells[self.mesh.contains[0]]):
            if np.array([domain(*coord) for coord in self.mesh.points[element]]).all():
                X[i] = 0.001
        return X


class SIMP_VIBE(object):
    """
    Solid Isotropic Material with Penalization method
    for topology optimization of the modal behavior
    _________________________________________________
    Supported objective functions:
    'max eig': Fundamental eigenvalue maximization
    """
    def __init__(self, Problem, volfrac=0.5, p=3, q=1, rmin=1.5, objective='max eig', FEM_solver_type='fast'):
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
        MIN_EIG = {'solid': self._solid_min_eigs_objective_function}
        OBJECTIVE_FUNS = {'max eig': MAX_EIG,
                          'min eig': MIN_EIG}

        # store the problem parameters
        self.FEM = Problem
        self.mesh = Problem.mesh
        self.mesh.compute_element_centers()
        self.FEM_solver_type = FEM_solver_type
        if self.FEM_solver_type == 'fast':
            self.FEM_solver = self._SIMP_fast_modal_solver
        elif self.FEM_solver_type == 'dense':
            self.FEM_solver = self._SIMP_dense_modal_solver()
        elif self.FEM_solver_type == 'sparse':
            self.FEM_solver = self._SIMP_sparse_modal_solver
        kind = 'coating' * self.FEM.coating + 'solid' * (not self.FEM.coating)
        self.objective_function = OBJECTIVE_FUNS[objective][kind]

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
        self.loop = 0

    def solve(self, v_ref,  converge=0.01, min_iter=1, max_iter=100, plot=True,
              save=True, verbose=True, convergence_criteria='change', sigma=0):
        """
               SIMP Optimization solver
               :param logfile:
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
        TOM_start_time = FEMOL.utils.unique_time_string()

        while not solved:
            # Iterate
            self.loop += 1

            # Iteration
            X_old = self.X
            if self.FEM_solver_type == 'dense':
                if self.loop == 1:
                    self.lmbd, self.v = self.FEM_solver(X_old, v_ref=v_ref, verbose=True)
                else:
                    self.lmbd, self.v = self.FEM_solver(X_old, v_ref=v_ref, verbose=False)
            elif self.FEM_solver_type == 'sparse':
                self.lmbd, self.v = self.FEM_solver(X_old, v_ref=v_ref, sigma=sigma)

            self.lmbds.append(self.lmbd)
            self.eigen_vectors.append(self.v)
            self.lmbd, self.dlmbd = self.objective_function(X_old)
            self._filter_sensibility(X_old)
            self.X = self._get_new_x(X_old)
            X1 = np.array(list(self.X.values())).flatten()
            X2 = np.array(list(X_old.values())).flatten()
            self.change = np.max(np.abs(X1 - X2))

            # Archive the previous X
            X_key = 'X{}'.format(self.loop - 1)
            if 'X' in self.mesh.cell_data.keys():
                self.mesh.cell_data[X_key] = self.mesh.cell_data['X']
            # Save the most recent as X
            self.mesh.cell_data['X'] = self.X
            # Save the displacement
            self.mesh.add_mode('m{}'.format(self.loop), self.v, self.FEM.N_dof)

            # Iteration information
            if plot:
                self._plot_iteration()
            if verbose:
                info = 'Iteration : {it}, Variation : {va}, EigenVal : {eg}'.format(it=self.loop,
                                                                                    va=self.change,
                                                                                    eg=self.lmbd)
                print(info)

            if convergence_criteria == 'change':
                if (self.change < converge) and (self.loop > min_iter):
                    solved = True
            elif convergence_criteria == 'objective':
                if np.abs(np.mean(self.lmbds[-3:-1]) - self.lmbds[-1]) < converge:
                    solved = True
            if self.loop == max_iter:
                solved = True

            # Save at each iteration
            if save:
                # Add the core height transformation
                self.mesh = self.density_to_core_height()
                # Add the penalized density values
                self.mesh.cell_data['X_real'] = {'quad': self.mesh.cell_data['X']['quad'] ** 3}
                # save for the current iteration
                self._save_TOM_result(TOM_start_time)

        return self.mesh

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

    def _save_TOM_result(self, timestring):
        # Try saving the file in results
        try:
            filename = 'Results/_topopt_cache/TOM_' + timestring
            self.mesh.save(filename)
        # If it does not work save it here
        except FileNotFoundError:
            filename = 'TOM_' + timestring
            self.mesh.save(filename)

    def _get_new_x(self, X):
        l1 = 0
        l2 = 100000
        move = 0.3
        # Flatten the X array
        X = np.hstack([X[cell_type] for cell_type in self.mesh.contains])
        # remove negative values
        if len(self.dlmbd[self.dlmbd < 0]) > (len(self.dlmbd)//4):
            self.dlmbd[self.dlmbd <= 0] = -self.dlmbd[self.dlmbd <= 0]
        else:
            self.dlmbd[self.dlmbd < 0] = 0

        if np.sum(self.dlmbd) < 1:
            self.dlmbd *= 1/self.dlmbd.max()

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
            # TODO : use element area for unstructured meshes
            if (np.sum(X_new) - self.f * self.mesh.N_ele) > 0:
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

    def _SIMP_dense_modal_solver(self, X, v_ref, verbose=True):
        """
        Solve the FEM modal problem using scipy.linalg.eig
        More precise than the sparse or fast solvers but very slow
        """
        self.FEM.assemble('K', X=X, p=self.p)
        self.FEM.assemble('M', X=X, q=self.q)
        now = time.time()
        w, v = scipy.linalg.eig(self.FEM.K.toarray(), self.FEM.M.toarray())
        print(f'Solved in {time.time() - now} s')
        w = np.sqrt(np.real(w)) / 2*np.pi
        self.all_lmbds.append(w)
        mac = [FEMOL.utils.MAC(vi, v_ref) for vi in v.T]
        print('Best mac match (dense solver) :', np.max(mac))
        i = np.argmax(mac)
        return w[i], v.T[i]

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

    def _SIMP_sparse_modal_solver(self, X, v_ref, sigma=0, verbose=True, k=20):
        """
        Solve the FEM modal problem using scipy.sparse.linalg.eigsh
        """
        self.FEM.assemble('K', X=X, p=self.p)
        self.FEM.assemble('M', X=X, q=self.q)
        K, M = self.FEM.K, self.FEM.M
        w_sp, v_sp = scipy.sparse.linalg.eigsh(K, M=M, k=k, sigma=sigma)
        f_sp = np.sqrt(w_sp)/(2*np.pi)
        mac_res = [FEMOL.utils.MAC(vi, v_ref) for vi in v_sp.T]
        best_vi = np.max(mac_res)
        print('Best mac match (sparse solver) :', best_vi)
        if best_vi > 0.01:
            self.FEM_solver_used.append('sparse')
            i = np.argmax(mac_res)
            return f_sp[i], v_sp.T[i]
        else:
            print('No corresponding eigenvector found using the modal assurance criterion using sparse solve')
            print('Falling back on the dense solve')
            self.FEM_solver_used.append('dense')
            return self._SIMP_fast_modal_solver(X=X, v_ref=v_ref, verbose=verbose)

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

    def _solid_min_eigs_objective_function(self, X):
        raise NotImplementedError

    def density_to_core_height_old(self):
        """Converts the density values results to core height values"""
        # Create the height/stiffness vectors
        layup = self.FEM.layups[1]
        zmin = layup.hA / 2
        zmax = layup.zc - layup.hA / 2
        zcoords = np.linspace(zmin, zmax)
        D_list = []
        plies = layup.plies
        for z in zcoords:
            layup_t = FEMOL.Layup(plies=plies, material=layup.mtr, symetric=False, h_core=0, z_core=z)
            D_list.append(layup_t.D_mat[0, 0])

        # Create the X vector for interpolation
        X_interp = (np.array(D_list) / np.max(D_list))
        X_interp = np.append([0], X_interp)
        zcoords = np.append([0], zcoords)
        # Create the interpolator
        core_height_interp = interp1d(X_interp, zcoords)
        self.height_interp = core_height_interp
        # Compute the core height values
        zcore = {}
        for key in self.mesh.cell_data['X']:
            zcore[key] = core_height_interp(self.mesh.cell_data['X'][key])

        self.mesh.cell_data['zc'] = zcore

        return self.mesh

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
            zcore[key][zcore[key] < 0] = h_values.min()

        self.mesh.cell_data['zc'] = zcore

        return self.mesh
