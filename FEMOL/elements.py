import numpy as np
import matplotlib.pyplot as plt


class Element(object):
    """
    Genereal FEMOL element class
    """
    # Number of nodes for the element
    N_nodes = 0
    # First order integration points and weights
    integration_points_1 = [(0, 0)]
    integration_weights_1 = [0]
    # Second order integration points and weights
    integration_points_2 = [(0, 0), (0, 0), (0, 0), (0, 0)]
    integration_weights_2 = [0, 0, 0, 0]
    # Third order integration points and weights
    integration_points_3 = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    integration_weights_3 = [0, 0, 0, 0, 0, 0]
    # Generalised coordinates shape function
    @staticmethod
    def shape(xi, eta):
        return None
    # Generalized coordinates shape function derivatives
    @staticmethod
    def dshape_dxi(xi, eta):
        return None
    @staticmethod
    def dshape_deta(xi, eta):
        return None

    def __init__(self, points, Ndof):
        """
        Element class constructor
        :param points: nodes coordinates of the element
        :param Ndof: nodes degrees of freedom of the element
        """
        self.x, self.y = points.transpose()[:2]  # 2D only
        self.N_dof = Ndof
        self.size = self.N_nodes * self.N_dof


class T3(object):
    """
    A class representing the T3 triangular element, equations are taken from
    Dhatt, G., & Touzot, G. (1981). Une Présentation de la méthode des éléments finis.
    Paris Québec : Maloine Presses de l’Université Laval (p.108).
    """
    # Number of nodes of the element
    N_nodes = 3
    # 1st order integration
    integration_points_1 = [(1/3, 1/3)]
    integration_weights_1 = [1]
    # 2nd order integration
    integration_points_2 = [(2/3, 1/6), (1/6, 1/6), (1/6, 2/3)]
    integration_weights_2 = [1/3, 1/3, 1/3]
    # 3rd order gauss integration points
    integration_points_3 = [(1/3, 1/3), (1/5, 1/5), (3/5, 1/5), (1/5, 3/5)]
    integration_weights_3 = [-27/96, 25/96, 25/96, 25/96]

    # shape functions and derivatives
    @staticmethod
    def shape(xi, eta):
        N = np.array([1 - xi - eta,
                      xi,
                      eta, ])
        return N

    @staticmethod
    def dshape_dxi():
        dN_dxi = np.array([-1, 1, 0])
        return dN_dxi

    @staticmethod
    def dshape_deta():
        dN_deta = np.array([-1, 0, 1])
        return dN_deta

    def __init__(self, points, N_dof=2):
        """
        Constructor for the T3 element
        Parameters
        ----------
        points : Triangle nodes points (in counter-clockwise order)
        N_dof : Number of degrees of freedom
        """
        self.x, self.y = points.transpose()[:2]  # 2D only
        self.N_dof = N_dof
        self.size = self.N_nodes * self.N_dof

    def area(self):
        """
        Computes the element area from the node coordinates
        """
        x, y = self.x, self.y
        A = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))

        return A

    def quality(self):
        """
        Measure the element quality by assessing the difference between its angles
        and the optimal value (60 degrees)
        """
        norm = np.linalg.norm
        v1 = [self.x[2] - self.x[0], self.y[2] - self.y[0]]
        v2 = [self.x[1] - self.x[0], self.y[1] - self.y[0]]
        T1 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = [self.x[0] - self.x[1], self.y[0] - self.y[1]]
        v2 = [self.x[2] - self.x[1], self.y[2] - self.y[1]]
        T2 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = [self.x[1] - self.x[2], self.y[1] - self.y[2]]
        v2 = [self.x[0] - self.x[2], self.y[0] - self.y[2]]
        T3 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        T = np.array([T1, T2, T3])
        Q = np.sum(np.abs(T - (np.pi/3)))
        return Q

    def plot(self):
        """
        Plots the element using matplotlib
        """
        ax = plt.gca()
        ax.scatter(self.x, self.y, color='k')
        ax.plot(self.x[[0, 1]], self.y[[0, 1]], color='k')
        ax.plot(self.x[[1, 2]], self.y[[1, 2]], color='k')
        ax.plot(self.x[[2, 0]], self.y[[2, 0]], color='k')

    def J(self):
        """
        Method computing the element jacobian matrix
        """
        x, y = self.x, self.y
        J = np.array([[x[1] - x[0], y[1] - y[0]],
                      [x[2] - x[0], y[2] - y[0]]])

        return J

    def inv_J(self):
        J = self.J()
        j = (1/self.det_J()) * np.array([[J[1, 1], -J[0, 1]],
                                         [-J[1, 0], J[0, 0]], ])
        return j

    def det_J(self):
        """
        Method computing the Jacobian determinant
        For a T3 element : det(J) = 2*A
        """
        return 2*self.area()

    def dshape_dx(self):
        """
        Method computing the derivatives of the shape functions according
        to the real coordinate x
        """
        j = self.inv_J()
        dN_dx = j[0, 0] * self.dshape_dxi() + j[0, 1] * self.dshape_deta()
        return dN_dx

    def dshape_dy(self):
        """
        Method computing the derivatives of the shape functions according
        to the real coordinate y
        """
        j = self.inv_J()
        dN_dy = j[1, 0] * self.dshape_dxi() + j[1, 1] * self.dshape_deta()
        return dN_dy

    def make_shape_xy(self):
        """
        Computes the element shape functions in respect to XY
        """
        xi, xj, xk = self.x[:3]
        yi, yj, yk = self.y[:3]

        def N1(x, y):
            return 1/(2 * self.area()) * ((yk - yj) * (xj - x) - (xk - xj) * (yj - y))

        def N2(x, y):
            return 1/(2 * self.area()) * ((yi - yk) * (xk - x) - (xi - xk) * (yk - y))

        def N3(x, y):
            return 1/(2 * self.area()) * ((yj - yi) * (xi - x) - (xj - xi) * (yi - y))

        def N(x, y):
            I = np.eye(self.N_dof)
            result = np.hstack([I * N1(x, y), I * N2(x, y), I * N3(x, y)])
            return result

        self.shape_xy = N

    def plane_B_matrix(self):
        """
        Computes the plane-stress strain matrix according to given generalized coordinates (xi, eta)

        returns : The strain matrix according to the number of degree of freedom and element size
        """
        # define shape according to degrees of freedom
        if self.N_dof == 2:
            b_i = 3

        b_j = self.size

        # Shape of the B matrix
        B_shape = (b_i, b_j)
        N_dof = self.N_dof
        B = np.zeros(B_shape)  # 3, 8 or 8, 24

        # Fill the strain matrix with shape function derivatives
        B[0, range(0, b_j - 1, N_dof)] = self.dshape_dx()
        B[1, range(1, b_j, N_dof)] = self.dshape_dy()
        B[2, range(0, b_j - 1, N_dof)] = self.dshape_dy()
        B[2, range(1, b_j, N_dof)] = self.dshape_dx()

        return B

    def Ke(self, *tensors):
        """
        Method to compute the element stiffness matrix of the T3 element
        Parameters
        ----------
        tensors :
        One 3x3 tensor for plane stress
        Three tensors (3x3, 3x3, 2x2) for plate bending

        Returns
        -------
        Ke : The element stiffness matrix
        """
        # Only works for plane stress
        if self.N_dof == 2:
            Ke = 0
            C = tensors[0]
            B = self.plane_B_matrix()
            # Stiffness matrix (all terms are constant)
            Ke += self.area() * (B.T @ C @ B)

            return Ke

    def Me(self, material, thickness):
        """
        Method returning the element mass matrix from the element material
        Parameters
        ----------
        material : A material class instance with a density attribute
        thickness : The element thickness
        Returns
        -------
        Me : The element mass matrix according to the nodes degrees of freedom
        """
        # Plane stress case
        if self.N_dof == 2:
            # Instantiate the node shape function
            self.make_shape_xy()
            # Get the integration points
            xis = np.array([pt[0] for pt in self.integration_points_3])
            etas = np.array([pt[1] for pt in self.integration_points_3])
            # Evaluate the shape function at the integration points
            shape = self.shape(xis, etas)

            # Compute the node coordinate integration points
            x_points = shape.T @ self.x
            y_points = shape.T @ self.y

            # Mass tensor
            V = np.identity(2) * material.rho * thickness

            # Integrate to find the element mass matrix
            Me = 0
            for x, y, wt in zip(x_points, y_points, self.integration_weights_3):
                N = self.shape_xy(x, y)
                Me += wt * N.T @ V @ N * self.det_J()
            return Me


class T6(object):

    N_nodes = 6
    # 1st order integration
    integration_points_1 = [(1/3, 1/3)]
    integration_weights_1 = [1]
    # 2nd order integration
    integration_points_2 = [(2/3, 1/6), (1/6, 1/6), (1/6, 2/3)]
    integration_weights_2 = [1/3, 1/3, 1/3]
    # 3rd order gauss integration points
    integration_points_3 = [(1/3, 1/3), (1/5, 1/5), (3/5, 1/5), (1/5, 3/5)]
    integration_weights_3 = [-27/96, 25/96, 25/96, 25/96]

    @staticmethod
    def shape(xi, eta):
        """
        Element shape function N(xi, eta) = [N1, ... , N6]
        """
        lm = 1 - xi - eta
        N = np.array([-lm*(1 - 2*lm),
                      4*xi*lm,
                     -xi*(1 - 2*xi),
                      4*xi*eta,
                     -eta*(1 - 2*eta),
                      4*eta*lm])
        return N

    @staticmethod
    def dshape_dxi(xi, eta):
        """
        Element shape function derivative in respect to xi
        """
        lm = 1 - xi - eta
        dN_dxi = np.array([1 - 4*lm,
                          4*(lm - xi),
                         -1 + 4*xi,
                          4*eta,
                          0,
                         -4*eta])
        return dN_dxi

    @staticmethod
    def dshape_deta(xi, eta):
        """
        Element shape function derivative in respect to eta
        """
        lm = 1 - xi - eta
        dN_deta = np.array([1 - 4*lm,
                          -4*xi,
                           0,
                           4*xi,
                          -1 + 4*eta,
                           4*(lm - eta)])
        return dN_deta

    def __init__(self, points, N_dof=2):
        """
        Constructor for the T6 quadratic triangle element
        :param points: node coordinates
        :param N_dof: degrees of freedom
        """
        self.x, self.y = points.transpose()[:2] # 2D only
        self.N_dof = N_dof  # Number of degrees of freedom
        self.size = self.N_nodes * N_dof  # Element size in the global matrix

    def make_shape_xy(self):
        """
        Method defining the shape functions of the element
        """
        x1, x2, x3 = self.x[:6:2]
        x4, x5, x6 = self.x[1:6:2]
        y1, y2, y3 = self.y[:6:2]
        y4, y5, y6 = self.y[1:6:2]

        def N1(x, y):
            a = (x2 - x3)*(y - y3) - (y2 - y3)*(x - x3)
            b = (x4 - x6)*(y - y6) - (y4 - y6)*(x - x6)
            c = (x2 - x3)*(y1 - y3) - (y2 - y3)*(x1 - x3)
            d = (x4 - x6)*(y1 - y6) - (y4 - y6)*(x1 - x6)
            return (a*b)/(c*d)

        def N2(x, y):
            a = (x3 - x1)*(y - y1) - (y3 - y1)*(x - x1)
            b = (x5 - x4)*(y - y4) - (y5 - y4)*(x - x4)
            c = (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)
            d = (x5 - x4)*(y2 - y4) - (y5 - y4)*(x2 - x4)
            return (a*b)/(c*d)

        def N3(x, y):
            a = (x2 - x1)*(y - y1) - (y2 -y1)*(x - x1)
            b = (x5 - x6)*(y - y6) - (y5 - y6)*(x - x6)
            c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
            d = (x5 - x6)*(y3 - y6) - (y5 - y6)*(x3 - x6)
            return (a*b)/(c*d)

        def N4(x, y):
            a = (x3 - x1)*(y - y1) - (y3 - y1)*(x - x1)
            b = (x2 - x3)*(y - y3) - (y2 - y3)*(x - x3)
            c = (x3 - x1)*(y4 - y1) - (y3 - y1)*(x4 - x1)
            d = (x2 - x3)*(y4 - y3) - (y2 - y3)*(x4 - x3)
            return (a*b)/(c*d)

        def N5(x, y):
            a = (x3 - x1)*(y - y1) - (y3 - y1)*(x - x1)
            b = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
            c = (x3 - x1)*(y5 - y1) - (y3 - y1)*(x5 - x1)
            d = (x2 - x1)*(y5 - y1) - (y2 - y1)*(x5 - x1)
            return (a*b)/(c*d)

        def N6(x, y):
            a = (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)
            b = (x2 - x3)*(y - y3) - (y2 - y3)*(x - x3)
            c = (x2 - x1)*(y6 - y1) - (y2 - y1)*(x6 - x1)
            d = (x2 - x3)*(y6 - y3) - (y2 - y3)*(x6 - x3)
            return (a*b)/(c*d)

        shape_functions = np.array([N1, N2, N3, N4, N5, N6])
        point_order = [0, 3, 1, 4, 2, 5]
        shape_functions = shape_functions[point_order]

        def N(x, y):
            I = np.eye(self.N_dof)
            shape = np.hstack([I*Ni(x,y) for Ni in shape_functions])
            return shape

        self.shape_xy = N

    def area(self):
        """
        Function computing the area of the element
        """
        x1, x2, x3 = self.x[[0, 2, 4]]
        y1, y2, y3 = self.y[[0, 2, 4]]
        A2 = x1*(y2 - y3) + x2*(y3-y1) + x3*(y1-y2)
        A = A2/2
        return A

    def plot(self):
        """
        Plots the element using matplotlib
        """
        ax = plt.gca()
        ax.scatter(self.x, self.y, color='k')
        ax.plot(self.x[[0, 2]], self.y[[0, 2]], color='k')
        ax.plot(self.x[[2, 4]], self.y[[2, 4]], color='k')
        ax.plot(self.x[[4, 0]], self.y[[4, 0]], color='k')

    def quality(self):
        """
        Measure the element quality by assessing the difference between its angles
        and the optimal value (60 degrees)
        """
        x1, x2, x3 = self.x[0, 2, 4]
        y1, y2, y3 = self.y[0, 2, 4]

        norm = np.linalg.norm
        v1 = [x3 - x1, y3 - y1]
        v2 = [x2 - x1, y2 - y1]
        T1 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = [x1 - x2, y1 - y2]
        v2 = [x3 - x2, y3 - y2]
        T2 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = [x2 - x3, y2 - y3]
        v2 = [x1 - x3, y1 - y3]
        T3 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        T = np.array([T1, T2, T3])
        Q = np.sum(np.abs(T - (np.pi/3)))
        return Q

    def J(self, xi, eta):
        """
        Fonction computing the Jacobian matrix for the T6 triangle element
        """
        S = np.vstack([self.x.T, self.y.T]).T
        dN_dxi = self.dshape_dxi(xi, eta)
        dN_deta = self.dshape_deta(xi, eta)
        L = np.vstack([dN_dxi, dN_deta])
        J = L @ S
        return J

    def det_J(self, xi, eta):
        """
        Function computing the jacobian matrix determinant at generalized
        coordinates xi and eta
        """
        S = np.vstack([self.x.T, self.y.T]).T
        dN_dxi = self.dshape_dxi(xi, eta)
        dN_deta = self.dshape_deta(xi, eta)
        L = np.vstack([dN_dxi, dN_deta])
        J = L @ S
        a, b, c, d = J.flatten()
        det = a*d - c*b
        return det

    def dshape_dx(self, xi, eta):
        """
        Element shape function derivative in respect to x
        """
        J_mat = self.J(xi, eta)
        d = J_mat[1, 1]
        b = J_mat[0, 1]
        J_det = self.det_J(xi, eta)
        dN_dx = (1/J_det) * (d*self.dshape_dxi(xi, eta) - b*self.dshape_deta(xi, eta))
        return dN_dx

    def dshape_dy(self, xi, eta):
        """
        Element shape function derivative in respect to y
        """
        J_mat = self.J(xi, eta)
        a = J_mat[0, 0]
        c = J_mat[1, 0]
        J_det = self.det_J(xi, eta)
        dN_dy = (1/J_det) * (-c*self.dshape_dxi(xi, eta) + a*self.dshape_deta(xi, eta))
        return dN_dy

    def plane_B_matrix(self, xi, eta):
        """
        Computes the plane-stress strain matrix according to given generalized coordinates (xi, eta)

        returns : The strain matrix according to the number of degree of freedom
        """
        # define shape according to degrees of freedom
        if self.N_dof == 2:
            b_i = 3
        elif self.N_dof == 6:
            b_i = 8

        b_j = self.size

        # Shape of the B matrix
        B_shape = (b_i, b_j)
        N_dof = self.N_dof
        B = np.zeros(B_shape)  # 3, 8 or 8, 24

        # Fill the strain matrix with shape function derivatives
        B[0, range(0, b_j - 1, N_dof)] = self.dshape_dx(xi, eta)
        B[1, range(1, b_j, N_dof)] = self.dshape_dy(xi, eta)
        B[2, range(0, b_j - 1, N_dof)] = self.dshape_dy(xi, eta)
        B[2, range(1, b_j, N_dof)] = self.dshape_dx(xi, eta)

        return B

    def bending_B_matrix(self, xi, eta):
        """
        Computes the bending strain matrix according to generalized coordinates

        returns : The bending strain matrix evaluated at xi and eta
        """
        # Get the strain matrix size
        b_i = 8
        b_j = self.size
        # Shape of the B matrix
        B_shape = (b_i, b_j)
        B = np.zeros(B_shape)
        # Plate bending
        B[3, range(3, b_j-1, self.N_dof)] = -self.dshape_dx(xi, eta)
        B[4, range(4, b_j-1, self.N_dof)] = -self.dshape_dy(xi, eta)
        B[5, range(4, b_j-1, self.N_dof)] = -self.dshape_dx(xi, eta)
        B[5, range(3, b_j-1, self.N_dof)] = -self.dshape_dy(xi, eta)
        return B

    def shear_B_matrix(self, xi, eta):
        """
        Computes the shear strain matrix according to generalized coordinates

        returns : The shear strain matrix evaluated at xi and eta
        """
        # Strain matrix
        # Get the strain matrix size
        b_i = 8
        b_j = self.size
        # Shape of the B matrix
        B_shape = (b_i, b_j)
        B = np.zeros(B_shape)
        # dw/dx + theta_y
        B[6, range(2, b_j-1, self.N_dof)] = self.dshape_dx(xi, eta)
        B[7, range(2, b_j-1, self.N_dof)] = self.dshape_dy(xi, eta)
        B[6, range(3, b_j-1, self.N_dof)] = -self.shape(xi, eta)
        B[7, range(4, b_j-1, self.N_dof)] = -self.shape(xi, eta)

        return B

    def Ke(self, *tensors):
        """
        Method returning the element stiffness matrix from the stiffness tensors
        Parameters
        ----------
        tensors :
        If the element is plane stress : C, a 3x3 tensor
        If the element is bending : C, D, G, 3x3, 3x3, 2x2 tensors

        Returns
        -------
        Ke : The element stiffness matrix according to the nodes degrees of freedom
        """

        # plane-stress case
        if self.N_dof == 2:
            Ke = np.zeros((self.size, self.size))
            C = tensors[0]
            # 2nd gauss integration
            for pt, w in zip(self.integration_points_3, self.integration_weights_3):
                # Strain matrix
                B = self.plane_B_matrix(*pt)
                # Stiffness matrix summation
                Ke += w*(B.T @ C @ B * self.det_J(*pt))
            return Ke

        # Plate bending case
        elif self.N_dof == 6:
            # Create 8x8 tensors
            T = np.zeros((8, 8))
            T[:3, :3] = tensors[0]
            C = T
            T = np.zeros((8, 8))
            T[3:6, 3:6] = tensors[1]
            D = T
            T = np.zeros((8, 8))
            T[6:, 6:] = tensors[2]
            G = T
            Kp, Kb, Ks = 0, 0, 0

            # 3nd order gauss integration for the plane and bending terms
            for pt, w in zip(self.integration_points_3, self.integration_weights_3):
                # Plane stress strain matrix
                Bp = self.plane_B_matrix(*pt)
                # Plate bending strain matrix
                Bb = self.bending_B_matrix(*pt)
                # Jacobian determinant
                det_J = self.det_J(*pt)
                # Stiffness matrix summation
                Kp += w*(Bp.T @ C @ Bp * det_J)
                Kb += w*(Bb.T @ D @ Bb * det_J)

            # 2nd order gauss integration for the shear term
            for pt, w in zip(self.integration_points_2, self.integration_weights_2):
                # Shear strain matrix
                Bs = self.shear_B_matrix(*pt)
                # Element stiffness summation
                Ks += w * (Bs.T @ G @ Bs * self.det_J(*pt))

            Ke = Kp + Kb + Ks
            Ke[np.arange(5, self.size, 6), np.arange(5, self.size, 6)] = 1
            return Ke

    def Me(self, material, thickness):
        """
        Method returning the element mass matrix from the element material
        Parameters
        ----------
        material : A material class instance with a density attribute
        thickness : The element thickness
        Returns
        -------
        Me : The element mass matrix according to the nodes degrees of freedom
        """
        # Plane stress case
        if self.N_dof == 2:
            # Instantiate the node shape function
            self.make_shape_xy()
            # Get the integration points
            xis = np.array([pt[0] for pt in self.integration_points_2])
            etas = np.array([pt[1] for pt in self.integration_points_2])
            # Evaluate the shape function at the integration points
            shape = self.shape(xis, etas)

            # Compute the node coordinate integration points
            x_points = shape.T @ self.x
            y_points = shape.T @ self.y

            # Mass tensor
            V = 0.45 * np.identity(2) * material.rho * thickness

            # Integrate to find the element mass matrix
            Me = 0
            for x, y, xi, eta, wt in zip(x_points, y_points, xis, etas, self.integration_weights_2):
                N = self.shape_xy(x, y)
                Me += wt * (N.T @ V @ N * self.det_J(xi ,eta))
            return Me

        # Plate bending case
        elif self.N_dof == 6:
            # Instantiate the node shape function
            self.make_shape_xy()
            # Get the integration points
            xis = np.array([pt[0] for pt in self.integration_points_2])
            etas = np.array([pt[1] for pt in self.integration_points_2])
            # Evaluate the shape function at the integration points
            shape = self.shape(xis, etas)

            # Compute the node coordinate integration points
            x_points = shape.T @ self.x
            y_points = shape.T @ self.y

            # Mass tensor
            V1 = 0.5*np.identity(3) * material.rho * thickness
            V2 = np.zeros((3, 3))
            V3 = np.identity(3) * material.rho * ((thickness) ** 3)/12
            V = np.vstack([np.hstack([V1, V2]), np.hstack([V2, V3])])

            # Integrate to find the element mass matrix
            Me = 0
            for x, y, xi, eta, wt in zip(x_points, y_points, xis, etas, self.integration_weights_2):
                N = self.shape_xy(x, y)
                Me += wt * (N.T @ V @ N * self.det_J(xi, eta))
            return Me


class Q4(object):
    """
    A class representing the Quadrilateral bi-linear finite
    element for structural analysis
    """
    # Number of node in the element
    N_nodes = 4

    # 1st order gauss integration points
    integration_points_1 = [(0, 0)]
    integration_weights_1 = [4]

    # 2nd order gauss integration points
    val = np.sqrt(3) / 3
    integration_points_2 = [(-val, -val), (val, -val), (val, val), (-val, val)]
    integration_weights_2 = [1, 1, 1, 1]

    # 3rd order gauss integration points
    val = np.sqrt(0.6)
    integration_points_3 = [(-val, -val), (-val, 0), (-val, val),
                            (0, -val), (0, 0), (0, val),
                            (val, -val), (val, 0), (val, val)]
    wt_1 = 5/9
    wt_2 = 8/9
    integration_weights_3 = [wt_1**2, wt_1*wt_2, wt_1**2,
                             wt_1*wt_2, wt_2**2, wt_1*wt_2,
                             wt_1**2, wt_1*wt_2, wt_1**2, ]

    @staticmethod
    def shape(xi, eta):
        """
        Method returning the element shape functions as a list where [N1, N2, ..., Nn]
        ----------
        xi, eta : Normalized coordinates in the element
        """
        N = 0.25 * np.array([(1 - xi) * (1 - eta),
                             (1 + xi) * (1 - eta),
                             (1 + xi) * (1 + eta),
                             (1 - xi) * (1 + eta), ])
        return N

    @staticmethod
    def dshape_dxi(eta):
        """
        Method returning the element shape functions derivatives according to xi
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        dN_dxi = 0.25 * np.array([(eta - 1), (1 - eta), (eta + 1), (-eta - 1)])  # vector
        return dN_dxi

    @staticmethod
    def dshape_deta(xi):
        """
        Method returning the element shape functions derivatives according to eta
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        dN_deta = 0.25 * np.array([(xi - 1), (-xi - 1), (xi + 1), (1 - xi)])  # vector
        return dN_deta

    @staticmethod
    def reference_plot():
        """
        Plots the quadrilateral reference element with fancy coordinates (figure style)
        """
        plt.figure(figsize=(8, 8))
        plt.scatter([-1, 1, 1, -1], [-1, -1, 1, 1], color='k')
        plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color='k')
        plt.axis("off")
        plt.annotate("(-1, -1)", (-1.37, -1))
        plt.annotate("(1, -1)", (1.1, -1))
        plt.annotate("(1, 1)", (1.1, 1))
        plt.annotate("(-1, 1)", (-1.37, 1))
        plt.arrow(-1.10, 0, 2.5, 0, head_width=0.08, color='k')
        plt.arrow(0, -1.10, 0, 2.5, head_width=0.08, color='k')
        plt.annotate(r"$\xi$", (-1.10 + 2.5, -0.2))
        plt.annotate(r"$\eta$", (-0.3, -1.10 + 2.5))
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])

    def __init__(self, points, N_dof=2):
        """
        Constructor for the Q4 element
        Parameters
        ----------
        points : nodes coordinates of the element as a list of tuples
        N_dof : degrees of freedom per node
        """
        self.x, self.y = points.transpose()[:2] # 2D only
        self.N_dof = N_dof  # Number of degrees of freedom
        self.size = self.N_nodes * N_dof  # Element size in the global matrix

    def make_shape_xy(self):
        """
        Element shape function in node coordinates
        """
        # Create the real element shape functions
        P_lines = []
        for xi, yi in zip(self.x, self.y):
            P_lines.append(np.array([1, xi, yi, xi * yi]))

        P = np.vstack(P_lines)
        P_inv = np.linalg.inv(P)

        def N1(x, y):
            coefs = P_inv[:, 0]
            return coefs[0] + coefs[1] * x + coefs[2] * y + coefs[3] * x * y

        def N2(x, y):
            coefs = P_inv[:, 1]
            return coefs[0] + coefs[1] * x + coefs[2] * y + coefs[3] * x * y

        def N3(x, y):
            coefs = P_inv[:, 2]
            return coefs[0] + coefs[1] * x + coefs[2] * y + coefs[3] * x * y

        def N4(x, y):
            coefs = P_inv[:, 3]
            return coefs[0] + coefs[1] * x + coefs[2] * y + coefs[3] * x * y

        def N(x, y):
            I = np.eye(self.N_dof)
            result = np.hstack([I * N1(x, y), I * N2(x, y), I * N3(x, y), I * N4(x, y)])
            return result

        self.shape_xy = N

    def plot(self):
        """
        Plots the element with nodes corresponding to x and y vectors
        :return: None
        """
        x, y = list(self.x.copy()), list(self.y.copy())
        plt.scatter(x, y, 20, color='k')
        i = 1
        for xi, yi in zip(x, y):
            plt.annotate(str(i), (xi + 0.3, yi + 0.3))
            i += 1
        x.append(x[0])
        y.append(y[0])
        ax = plt.gca()
        ax.plot(x, y, color='k')
        plt.arrow(min(x)-0.5, np.mean(y), max(x) + 0.5, 0, head_width=0.2, color='k')
        plt.arrow(np.mean(x), min(y) - 0.5, 0, max(y) + 1.6, head_width=0.2, color='k')
        plt.annotate(r"$x$", (min(x)-0.5 + max(x) + 0.5, np.mean(y) + 0.2))
        plt.annotate(r"$y$", (np.mean(x) + 0.2, min(y) - 0.5 + max(y) + 1.6))
        ax.set_aspect('equal')
        ax.set_xlim(min(x) - 1, max(x) + 1)
        ax.set_ylim(min(y) - 1, max(y) + 1)

    def quality(self):
        """
        Measure of the element quality by assessing the difference between its angles and the
        optimal value (90 degrees)
        """
        norm = np.linalg.norm
        v1 = np.array([self.x[3] - self.x[0], self.y[3] - self.y[0]])
        v2 = np.array([self.x[1] - self.x[0], self.y[1] - self.y[0]])
        T1 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = np.array([self.x[0] - self.x[1], self.y[0] - self.y[1]])
        v2 = np.array([self.x[2] - self.x[1], self.y[2] - self.y[1]])
        T2 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = np.array([self.x[1] - self.x[2], self.y[1] - self.y[2]])
        v2 = np.array([self.x[3] - self.x[2], self.y[3] - self.y[2]])
        T3 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        v1 = np.array([self.x[2] - self.x[3], self.y[2] - self.y[3]])
        v2 = np.array([self.x[0] - self.x[3], self.y[0] - self.y[3]])
        T4 = np.arccos(np.inner(v1, v2) / (norm(v1) * norm(v2)))
        T = np.array([T1, T2, T3, T4])
        Q = np.sum(np.abs(T - (np.pi/2)))
        return Q

    def det_J(self, xi, eta):
        x, y = self.x, self.y
        A0 = (1 / 8) * (x[0] * y[1] - x[0] * y[3] - x[1] * y[0] + x[1] * y[2]
                        - x[2] * y[1] + x[2] * y[3] + x[3] * y[0] - x[3] * y[2])
        A1 = (1 / 8) * (-x[0] * y[2] + x[0] * y[3] + x[1] * y[2] - x[1] * y[3]
                        + x[2] * y[0] - x[2] * y[1] - x[3] * y[0] + x[3] * y[1])
        A2 = (1 / 8) * (-x[0] * y[1] + x[0] * y[2] + x[1] * y[0] - x[1] * y[3]
                        - x[2] * y[0] + x[2] * y[3] + x[3] * y[1] - x[3] * y[2])
        det_J = A0 + A1 * xi + A2 * eta  # scalar

        return det_J

    def dshape_dx(self, xi, eta):
        """
        Method returning the element shape functions derivatives according to x
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        x, y = self.x, self.y
        d = 0.25 * (y[0] * (xi - 1) - y[1] * (xi + 1) + y[2] * (xi + 1) - y[3] * (xi - 1))
        b = 0.25 * (y[0] * (eta - 1) - y[1] * (eta - 1) + y[2] * (eta + 1) - y[3] * (eta + 1))
        dN_dxi = self.dshape_dxi(eta)
        dN_deta = self.dshape_deta(xi)

        dN_dx = (1/self.det_J(xi, eta))*(d * dN_dxi - b * dN_deta)
        return dN_dx

    def dshape_dy(self, xi, eta):
        """
        Method returning the element shape functions derivatives according to y
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        x, y = self.x, self.y
        a = 0.25 * (x[0] * (eta - 1) - x[1] * (eta - 1) + x[2] * (eta + 1) - x[3] * (eta + 1))
        c = 0.25 * (x[0] * (xi - 1) - x[1] * (xi + 1) + x[2] * (xi + 1) - x[3] * (xi - 1))
        dN_dxi = self.dshape_dxi(eta)
        dN_deta = self.dshape_deta(xi)

        dN_dy = (1/self.det_J(xi, eta))*(-c * dN_dxi + a * dN_deta)

        return dN_dy

    def plane_B_matrix(self, xi, eta):
        """
        Computes the plane-stress strain matrix according to given generalized coordinates (xi, eta)

        returns : The strain matrix according to the number of degree of freedom
        """
        # define shape acoording to degrees of freedom
        if self.N_dof == 2:
            b_i = 3
        elif self.N_dof == 6:
            b_i = 8
        else:
            print('Invalid N_dof')
            raise ValueError

        b_j = self.size

        # Shape of the B matrix
        B_shape = (b_i, b_j)
        N_dof = self.N_dof
        B = np.zeros(B_shape)  # 3, 8 or 8, 24

        # Fill the strain matrix with shape function derivatives
        B[0, range(0, b_j - 1, N_dof)] = self.dshape_dx(xi, eta)
        B[1, range(1, b_j, N_dof)] = self.dshape_dy(xi, eta)
        B[2, range(0, b_j - 1, N_dof)] = self.dshape_dy(xi, eta)
        B[2, range(1, b_j, N_dof)] = self.dshape_dx(xi, eta)

        return B

    def bending_B_matrix(self, xi, eta):
        """
        Computes the bending strain matrix according to generalized coordinates

        returns : The bending strain matrix evaluated at xi and eta
        """
        # Strain matrix
        B = np.zeros((8, 24))
        # Plate bending
        B[3, range(3, 23, 6)] = -self.dshape_dx(xi, eta)
        B[4, range(4, 23, 6)] = -self.dshape_dy(xi, eta)
        B[5, range(4, 23, 6)] = -self.dshape_dx(xi, eta)
        B[5, range(3, 23, 6)] = -self.dshape_dy(xi, eta)

        return B

    def shear_B_matrix(self, xi, eta):
        """
        Computes the shear strain matrix according to generalized coordinates

        returns : The shear strain matrix evaluated at xi and eta
        """
        # Strain matrix
        B = np.zeros((8, 24))
        # dw/dx + theta_y
        B[6, range(2, 23, 6)] = self.dshape_dx(xi, eta)
        B[7, range(2, 23, 6)] = self.dshape_dy(xi, eta)
        B[6, range(3, 23, 6)] = -self.shape(xi, eta)
        B[7, range(4, 23, 6)] = -self.shape(xi, eta)

        return B

    def center(self):
        """
        Returns the element center in global (x, y) coordinates
        """
        x_bar = np.mean(self.x)
        y_bar = np.mean(self.y)
        return x_bar, y_bar

    def Ke(self, *tensors):
        """
        Method returning the element stiffness matrix from the stiffness tensors
        Parameters
        ----------
        tensors :
        If the element is plane stress : C, a 3x3 tensor
        If the element is bending : C, D, G, 3x3, 3x3, 2x2 tensors

        Returns
        -------
        Ke : The element stiffness matrix according to the nodes degrees of freedom
        """

        # plane-stress case
        if self.N_dof == 2:
            Ke = np.zeros((8, 8))
            C = tensors[0]
            # 2nd gauss integration
            for pt, w in zip(self.integration_points_2, self.integration_weights_2):
                # Strain matrix
                B = self.plane_B_matrix(*pt)
                # Stiffness matrix summation
                Ke += B.T @ C @ B * self.det_J(*pt)

            return Ke

        # Plate bending case
        elif self.N_dof == 6:
            # Create 8x8 tensors
            T = np.zeros((8, 8))
            T[:3, :3] = tensors[0]
            A = T.copy()
            T = np.zeros((8, 8))
            T[3:6, 3:6] = tensors[1]
            D = T.copy()
            T = np.zeros((8, 8))
            T[6:, 6:] = tensors[2]
            G = T.copy()

            Ks = 0
            # 1st order gauss integration for the shear term
            for pt, w in zip(self.integration_points_1, self.integration_weights_1):
                # Shear strain matrix
                Bs = self.shear_B_matrix(*pt)
                # Element stiffness summation
                Ks += w * (Bs.T @ G @ Bs * self.det_J(*pt))

            # If there is a coupled bending tensor
            if len(tensors) > 3:
                T = np.zeros((8, 8))
                T[:3, 3:6] = tensors[3]
                T[3:6, :3] = tensors[3]
                B = T.copy()
                C = A + B + D
                K_coupled = 0
                # 2nd order gauss integration for the in-plane and bending
                for pt, w in zip(self.integration_points_2, self.integration_weights_2):
                    # Plane stress strain matrix
                    Bp = self.plane_B_matrix(*pt)
                    # Plate bending strain matrix
                    Bb = self.bending_B_matrix(*pt)
                    B = Bp + Bb
                    # Jacobian determinant
                    det_J = self.det_J(*pt)
                    # Stiffness matrix summation
                    K_coupled += w * (B.T @ C @ B * det_J)
                K_element = K_coupled + Ks

            else:
                Kp, Kb = 0, 0
                # 2nd order gauss integration for the plane and bending terms
                for pt, w in zip(self.integration_points_2, self.integration_weights_2):
                    # Plane stress strain matrix
                    Bp = self.plane_B_matrix(*pt)
                    # Plate bending strain matrix
                    Bb = self.bending_B_matrix(*pt)
                    # Jacobian determinant
                    det_J = self.det_J(*pt)
                    # Stiffness matrix summation
                    Kp += w * (Bp.T @ A @ Bp * det_J)
                    Kb += w * (Bb.T @ D @ Bb * det_J)
                K_element = Kp + Kb + Ks

            K_element[np.arange(5, 24, 6), np.arange(5, 24, 6)] = 1
            return K_element

    def Me(self, material=None, thickness=None, V=None):
        """
        Method returning the element mass matrix from the element material
        Parameters
        ----------
        material :
        A material class instance with a density attribute
        thickness :
        The element thickness

        Returns
        -------
        Me : The element mass matrix according to the nodes degrees of freedom
        """
        # Plane stress case
        if self.N_dof == 2:
            # Instantiate the node shape function
            self.make_shape_xy()
            # Get the integration points
            xis = np.array([pt[0] for pt in self.integration_points_2])
            etas = np.array([pt[1] for pt in self.integration_points_2])
            # Evaluate the shape function at the integration points
            shape = self.shape(xis, etas)

            # Compute the node coordinate integration points
            x_points = shape.T @ self.x
            y_points = shape.T @ self.y

            # Mass tensor
            V = np.identity(2) * material.rho * thickness

            # Integrate to find the element mass matrix
            Me = 0
            for x, y, xi, eta, wt in zip(x_points, y_points, xis, etas, self.integration_weights_2):
                N = self.shape_xy(x, y)
                Me += N.T @ V @ N * self.det_J(xi, eta)

            return Me

        # Plate bending case
        elif self.N_dof == 6:
            # Instantiate the node shape function
            self.make_shape_xy()
            # Get the integration points
            xis = np.array([pt[0] for pt in self.integration_points_2])
            etas = np.array([pt[1] for pt in self.integration_points_2])
            # Evaluate the shape function at the integration points
            shape = self.shape(xis, etas)

            # Compute the node coordinate integration points
            x_points = shape.T @ self.x
            y_points = shape.T @ self.y

            # Mass tensor
            if V is not None:
                pass
            else:
                V1 = np.identity(3) * material.rho * thickness
                V2 = np.zeros((3, 3))
                V3 = np.identity(3) * material.rho * (thickness ** 3)/12
                V = np.vstack([np.hstack([V1, V2]), np.hstack([V2, V3])])

            # Integrate to find the element mass matrix
            Me = 0
            for x, y, xi, eta, wt in zip(x_points, y_points, xis, etas, self.integration_weights_2):
                N = self.shape_xy(x, y)
                Me += wt * N.T @ V @ N * self.det_J(xi, eta)

            return Me

    def stress(self, u, T):
        """
        Computes the stress components at the element centroid
        :param u: Displacement vector
        :param T: Stiffness tensor
        :return: S, the stress components [Sx, Sy, Sxy]
        """
        S = T @ self.plane_B_matrix(0, 0) @ u
        return S


# TODO : Complete the Q8 element
class Q8(object):
    """
    A class representing the Quadrilateral quadratic finite
    element for structural analysis
    """
    # Number of node in the element
    N_nodes = 8

    # 1st order gauss integration points
    integration_points_1 = [(0, 0)]
    integration_weights_1 = [4]

    # 2nd order gauss integration points
    val = np.sqrt(3) / 3
    integration_points_2 = [(-val, -val), (val, -val), (val, val), (-val, val)]
    integration_weights_2 = [1, 1, 1, 1]

    # 3rd order gauss integration points
    val = np.sqrt(3 / 5)
    w1 = 8 / 9
    w2 = 5 / 9
    integration_points_3 = [(-val, -val), (0, -val), (val, -val),
              (-val, 0), (0, 0), (val, 0),
              (-val, val), (0, val), (val, val)]
    integration_weights_3 = [w2 * w2, w1 * w2, w2 * w2, w1 * w2, w1 * w1, w1 * w2, w2 * w2, w1 * w2, w2 * w2]

    @staticmethod
    def shape(xi, eta):
        """
        Method returning the element shape functions as a list where [N1, N2, ..., Nn]
        ----------
        xi, eta : Normalized coordinates in the element
        """
        N = np.array([0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),
                      0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1),
                      0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1),
                      0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1),
                      0.50 * (1 - eta) * (1 + xi) * (1 - xi),
                      0.50 * (1 + xi) * (1 + eta) * (1 - eta),
                      0.50 * (1 + eta) * (1 + xi) * (1 - xi),
                      0.50 * (1 - xi) * (1 + eta) * (1 - eta), ])
        return N

    @staticmethod
    def dshape_dxi(xi, eta):
        """
        Method returning the element shape functions derivatives according to xi
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        dN_dxi = np.array([-(0.25 - 0.25*xi)*(1 - eta) + (0.25*eta - 0.25)*(-eta - xi - 1),
                            (0.25 - 0.25*eta)*(-eta + xi - 1) + (1 - eta)*(0.25*xi + 0.25),
                            (0.25 * eta + 0.25) * (eta + xi - 1) + (eta + 1) * (0.25 * xi + 0.25),
                           -(0.25 - 0.25 * xi) * (eta + 1) + (-0.25 * eta - 0.25) * (eta - xi - 1),
                            (0.5 - 0.5 * eta) * (1 - xi) + (0.5 - 0.5 * eta) * (-xi - 1),
                             0.5 * (1 - eta) * (eta + 1),
                            (1 - xi) * (0.5 * eta + 0.5) + (0.5 * eta + 0.5) * (-xi - 1),
                            -0.5 * (1 - eta) * (eta + 1), ])  # vector
        return dN_dxi

    @staticmethod
    def dshape_deta(xi, eta):
        """
        Method returning the element shape functions derivatives according to eta
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        dN_deta = np.array([-(0.25 - 0.25*xi)*(1 - eta) + (0.25*xi - 0.25)*(-eta - xi - 1),
                            -(1 - eta) * (0.25 * xi + 0.25) + (-0.25 * xi - 0.25) * (-eta + xi - 1),
                             (eta + 1) * (0.25 * xi + 0.25) + (0.25 * xi + 0.25) * (eta + xi - 1),
                             (0.25 - 0.25 * xi) * (eta + 1) + (0.25 - 0.25 * xi) * (eta - xi - 1),
                             -0.5 * (1 - xi) * (xi + 1),
                             (1 - eta) * (0.5 * xi + 0.5) + (-eta - 1) * (0.5 * xi + 0.5),
                              0.5 * (1 - xi) * (xi + 1),
                             (0.5 - 0.5 * xi) * (1 - eta) + (0.5 - 0.5 * xi) * (-eta - 1), ])  # vector
        return dN_deta

    def __init__(self, points, N_dof=2):
        """
        Constructor for the Q4 element
        Parameters
        ----------
        points : nodes coordinates of the element as a list of tuples
        N_dof : degrees of freedom per node
        """
        self.x, self.y = points.transpose()
        self.N_dof = N_dof  # Number of degrees of freedom
        self.size = self.N_nodes * N_dof  # Element size in the global matrix

    def det_J(self, xi, eta):
        x, y = self.x, self.y
        A0 = (1 / 8) * (x[0] * y[1] - x[0] * y[3] - x[1] * y[0] + x[1] * y[2]
                        - x[2] * y[1] + x[2] * y[3] + x[3] * y[0] - x[3] * y[2])
        A1 = (1 / 8) * (-x[0] * y[2] + x[0] * y[3] + x[1] * y[2] - x[1] * y[3]
                        + x[2] * y[0] - x[2] * y[1] - x[3] * y[0] + x[3] * y[1])
        A2 = (1 / 8) * (-x[0] * y[1] + x[0] * y[2] + x[1] * y[0] - x[1] * y[3]
                        - x[2] * y[0] + x[2] * y[3] + x[3] * y[1] - x[3] * y[2])
        det_J = A0 + A1 * xi + A2 * eta  # scalar

        return det_J

    def dshape_dx(self, xi, eta):
        """
        Method returning the element shape functions derivatives according to x
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        x, y = self.x, self.y
        d = 0.25 * (y[0] * (xi - 1) - y[1] * (xi + 1) + y[2] * (xi + 1) - y[3] * (xi - 1))
        b = 0.25 * (y[0] * (eta - 1) - y[1] * (eta - 1) + y[2] * (eta + 1) - y[3] * (eta + 1))
        dN_dxi = self.dshape_dxi(eta)
        dN_deta = self.dshape_deta(xi)

        dN_dx = (1 / self.det_J(xi, eta)) * (d * dN_dxi - b * dN_deta)
        return dN_dx

    def dshape_dy(self, xi, eta):
        """
        Method returning the element shape functions derivatives according to y
        as a list where [N1, N2, ..., Nn] n is the node number
        ----------
        xi, eta : Normalized coordinates in the element
        """
        x, y = self.x, self.y
        a = 0.25 * (x[0] * (eta - 1) - x[1] * (eta - 1) + x[2] * (eta + 1) - x[3] * (eta + 1))
        c = 0.25 * (x[0] * (xi - 1) - x[1] * (xi + 1) + x[2] * (xi + 1) - x[3] * (xi - 1))
        dN_dxi = self.dshape_dxi(eta)
        dN_deta = self.dshape_deta(xi)

        dN_dy = (1 / self.det_J(xi, eta)) * (-c * dN_dxi + a * dN_deta)

        return dN_dy

    def plane_B_matrix(self, xi, eta):
        """
        Computes the plane-stress strain matrix according to given generalized coordinates (xi, eta)

        returns : The strain matrix according to the number of degree of freedom
        """
        # define shape acoording to degrees of freedom
        if self.N_dof == 2:
            b_i = 3
            b_j = 8
        elif self.N_dof == 6:
            b_i = 8
            b_j = 24

        # Shape of the B matrix
        B_shape = (b_i, b_j)
        N_dof = self.N_dof
        B = np.zeros(B_shape)  # 3, 8 or 8, 24

        # Fill the strain matrix with shape function derivatives
        B[0, range(0, b_j - 1, N_dof)] = self.dshape_dx(xi, eta)
        B[1, range(1, b_j, N_dof)] = self.dshape_dy(xi, eta)
        B[2, range(0, b_j - 1, N_dof)] = self.dshape_dy(xi, eta)
        B[2, range(1, b_j, N_dof)] = self.dshape_dx(xi, eta)

        return B

    def bending_B_matrix(self, xi, eta):
        """
        Computes the bending strain matrix according to generalized coordinates

        returns : The bending strain matrix evaluated at xi and eta
        """
        # Strain matrix
        B = np.zeros((8, 24))
        # Plate bending
        B[3, range(3, 23, 6)] = -self.dshape_dx(xi, eta)
        B[4, range(4, 23, 6)] = -self.dshape_dy(xi, eta)
        B[5, range(4, 23, 6)] = -self.dshape_dx(xi, eta)
        B[5, range(3, 23, 6)] = -self.dshape_dy(xi, eta)

        return B

    def shear_B_matrix(self, xi, eta):
        """
        Computes the shear strain matrix according to generalized coordinates

        returns : The shear strain matrix evaluated at xi and eta
        """
        # Strain matrix
        B = np.zeros((8, 24))
        # dw/dx + theta_y
        B[6, range(2, 23, 6)] = self.dshape_dx(xi, eta)
        B[7, range(2, 23, 6)] = self.dshape_dy(xi, eta)
        B[6, range(3, 23, 6)] = -self.shape(xi, eta)
        B[7, range(4, 23, 6)] = -self.shape(xi, eta)

        return B

    def Ke(self, *tensors):
        """
        Method returning the element stiffness matrix from the stiffness tensors
        Parameters
        ----------
        tensors :
        If the element is plane stress : C, a 3x3 tensor
        If the element is bending : C, D, G, 3x3, 3x3, 2x2 tensors

        Returns
        -------
        Ke : The element stiffness matrix according to the nodes degrees of freedom
        """

        # plane-stress case
        if self.N_dof == 2:
            Ke = 0
            C = tensors[0]
            # 2nd gauss integration
            for pt, w in zip(self.integration_points_2, self.integration_weights_2):
                # Strain matrix
                B = self.plane_B_matrix(*pt)
                # Stiffness matrix summation
                Ke += B.T @ C @ B * self.det_J(*pt)

            return Ke

        # Plate bending case
        elif self.N_dof == 6:
            C, D, G = tensors
            Kp, Kb, Ks = 0, 0, 0

            # 2nd order gauss integration for the plane and bending terms
            for pt, w in zip(self.integration_points_2, self.integration_weights_2):
                # Plane stress strain matrix
                Bp = self.plane_B_matrix(*pt)
                # Plate bending strain matrix
                Bb = self.bending_B_matrix(*pt)
                # Jacobian determinant
                det_J = self.det_J(*pt)
                # Stiffness matrix summation
                Kp += Bp.T @ C @ Bp * det_J
                Kb += Bb.T @ D @ Bb * det_J

            # 1st order gauss integration for the shear term
            for pt, w in zip(self.integration_points_1, self.integration_weights_1):
                # Shear strain matrix
                Bs = self.shear_B_matrix(*pt)
                # Element stiffness summation
                Ks += Bs.T @ G @ Bs * self.det_J(*pt)
