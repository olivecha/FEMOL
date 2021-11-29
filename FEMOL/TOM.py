import FEMOL.utils
import FEMOL.elements
# Numpy
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
# Ipython
from IPython.display import clear_output

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
                clear_output(wait=True)
                info = "Iteration : " + str(self.loop) + ', variation : ' + str(np.around(self.change * 100, 1))
                print(info)

        if save:
            self._save_TOM_result()

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

    def _save_TOM_result(self):
        # Try saving the file in results
        try:
            filename = 'Results/_topopt_cache/TOM_' + FEMOL.utils.unique_time_string()
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

    def _solid_compliance_objective_function(self, X):
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
    'max eigs': Fundamental eigenvalue maximization
    """
    def __init__(self, Problem, volfrac=0.5, p=3, q=1, rmin=1.5, objective='max eigs'):
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
        OBJECTIVE_FUNS = {'max eigs': MAX_EIG, }

        # store the problem parameters
        self.FEM = Problem
        self.mesh = Problem.mesh
        self.mesh.compute_element_centers()
        self.FEM_solver = self._SIMP_modal_solver
        kind = 'coating' * self.FEM.coating + 'solid' * (not self.FEM.coating)
        self.objective_function = OBJECTIVE_FUNS[objective][kind]

        # define the TOM parameters
        self.f = volfrac
        self.rmin = rmin
        self.p = p
        self.q = q
        self.X = {key: np.ones(self.mesh.cells[key].shape[0]) * volfrac for key in self.mesh.contains}

    def solve(self, v_ref,  converge=0.01, max_iter=100, plot=True, save=True):
        """
               SIMP Optimization solver
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

        while (self.change > converge) & (self.loop < max_iter):
            # Iterate
            self.loop += 1

            # Iteration
            X_old = self.X
            self.lmbd, self.v = self.FEM_solver(X_old, v_ref=v_ref)
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
            else:
                clear_output(wait=True)
                info = "Iteration : " + str(self.loop) + ', variation : ' + str(np.around(self.change * 100, 1))
                print(info)

        if save:
            self._save_TOM_result()

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

    def _save_TOM_result(self):
        # Try saving the file in results
        try:
            filename = 'Results/_topopt_cache/TOM_' + FEMOL.utils.unique_time_string()
            self.mesh.save(filename)
        # If it does not work save it here
        except FileNotFoundError:
            filename = 'TOM_' + FEMOL.utils.unique_time_string()
            self.mesh.save(filename)

    def _get_new_x(self, X):
        l1 = 0
        l2 = 100000
        move = 0.5
        # Flatten the X array
        X = np.hstack([X[cell_type] for cell_type in self.mesh.contains])

        while (l2 - l1) > 1e-4:
            lmid = 0.5 * (l1 + l2)

            X1 = X + move
            X2 = X * (self.dlmbd / lmid) ** 0.3

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

    def _SIMP_modal_solver(self, X, v_ref):
        """
        Solve the FEM modal problem with the current element density values
        Returns the eigenvectors
        """
        self.FEM.assemble('K', X=X, p=self.p)
        self.FEM.assemble('M', X=X, q=self.q)
        w, v = self.FEM.solve(verbose=False, filtre=0)
        mac = [FEMOL.utils.MAC(vi, v_ref) for vi in v]
        i = np.argmax(mac)
        return w[i]**2, v[i]

    def _solid_max_eigs_objective_function(self, X):
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
                dlmbd.append(Ve.T @ (self.p * xe ** (self.p - 1) * Ke
                                     - self.lmbd * self.q * xe ** (self.q-1) * Me) @ Ve)

            return self.lmbd, np.array(dlmbd)

    def _coating_max_eigs_objective_function(self, X):
        raise NotImplementedError
