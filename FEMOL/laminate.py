import numpy as np

class Layup(object):
    """
    A class to manipulate layups according to laminate theory for the MECH530 class
    """

    def __init__(self, material, plies=None, symetric=False, z_core=0, h_core=0):
        """
        Constructor for the Layup class

        :param material: layup material, a OrthotropicMaterial class instance
        :param plies: a list of ply angles
        :param symetric: If true the plies are duplicated to form a symetric laminate
        :param core: total core thickness
        """
        # initiate strain variables
        self.curvature = 0
        self.eps_off = 0

        # Get variables into the class
        self.mtr = material
        self.plies = plies
        self.sym = symetric

        # save core thickness
        self.hc = h_core
        self.zc = z_core

        # get layup angles
        if self.sym:
            self.angles = np.hstack([np.array(plies), np.flip(np.array(plies))])
            self.angles_s = np.flip(np.array(plies))
        else:
            self.angles = np.array(plies)

        # Get layup thickness
        self.Nb_plies = self.angles.shape[0]
        self.hA = self.Nb_plies * self.mtr.hi
        self.h = self.hA + self.hc

        # Off-axis Q matrices
        self.Q_off = [self.mtr.Q_mat(angle) for angle in self.angles]

        # Off-axis S matrices
        self.S_off = [self.mtr.S_mat(angle) for angle in self.angles]

        # A matrix
        self.get_A()

        # D matrix
        self.get_D()

        # Out of plane shear matrix
        self.get_G()

        # Ply coordinates
        self.get_zcoord()

    def get_A(self):
        """
        A matrix computation for a given laminate
        """
        # Constant U values
        self.U_1 = (1 / 8) * (3 * self.mtr.Qxx + 3 * self.mtr.Qyy + 2 * self.mtr.Qxy + 4 * self.mtr.Qss)
        self.U_2 = (1 / 2) * (self.mtr.Qxx - self.mtr.Qyy)
        self.U_3 = (1 / 8) * (self.mtr.Qxx + self.mtr.Qyy - 2 * self.mtr.Qxy - 4 * self.mtr.Qss)
        self.U_4 = (1 / 8) * (self.mtr.Qxx + self.mtr.Qyy + 6 * self.mtr.Qxy - 4 * self.mtr.Qss)
        self.U_5 = (1 / 8) * (self.mtr.Qxx + self.mtr.Qyy - 2 * self.mtr.Qxy + 4 * self.mtr.Qss)
        self.U = [self.U_1, self.U_2, self.U_3, self.U_4, self.U_5]

        # Vi star Values
        # thickness of plies is constant (hi)
        V_1 = sum(np.cos(2 * np.radians(self.angles)) * self.mtr.hi / self.hA)
        V_2 = sum(np.cos(4 * np.radians(self.angles)) * self.mtr.hi / self.hA)
        V_3 = sum(np.sin(2 * np.radians(self.angles)) * self.mtr.hi / self.hA)
        V_4 = sum(np.sin(4 * np.radians(self.angles)) * self.mtr.hi / self.hA)

        # Vector for matrix product
        U_vector = [1, self.U_2, self.U_3]

        V_mat = np.array([[self.U_1, V_1, V_2],
                          [self.U_1, -V_1, V_2],
                          [self.U_4, 0, -V_2],
                          [self.U_5, 0, -V_2],
                          [0, 0.5 * V_3, V_4],
                          [0, 0.5 * V_3, -V_4], ])

        A = self.hA * (V_mat @ U_vector)

        self.A_mat = np.array([[A[0], A[2], A[4]],
                               [A[2], A[1], A[5]],
                               [A[4], A[5], A[3]]])

        self.a_mat = np.linalg.inv(self.A_mat)

    def get_D(self):
        """
        D matrix for stiffness bending computation
        """
        # bottom of the integral
        h_star_1 = (1/3) * ((-self.hc/2 + self.zc) ** 3 - (-self.h/2 + self.zc)**3)
        # top of the integral
        h_star_2 = (1/3) * ((self.h/2 + self.zc)**3 - (self.hc/2 + self.zc) ** 3)
        # total
        h_star = h_star_1 + h_star_2
        self.hstr = h_star

        # Case with a core
        if self.hc > 0:
            lower_z_values = np.arange(-self.h/2, -self.hc/2 + self.mtr.hi, self.mtr.hi) + self.zc
            upper_z_values = np.arange(self.hc/2, self.h/2 + self.mtr.hi, self.mtr.hi) + self.zc
            self.z_values = np.hstack([lower_z_values, upper_z_values])

            V = np.zeros(4)
            # Bottom of core
            for i, angle in enumerate(self.angles[:len(self.angles) // 2]):
                theta = np.radians(angle)
                z_i = lower_z_values[i + 1] ** 3 - lower_z_values[i] ** 3
                V += (1 / 3) * np.array(
                    [np.cos(2 * theta), np.cos(4 * theta), np.sin(2 * theta), np.sin(4 * theta)]) * z_i

            # Top of core
            for i, angle in enumerate(self.angles[len(self.angles) // 2:]):
                theta = np.radians(angle)
                z_i = upper_z_values[i + 1] ** 3 - upper_z_values[i] ** 3
                V += (1 / 3) * np.array(
                    [np.cos(2 * theta), np.cos(4 * theta), np.sin(2 * theta), np.sin(4 * theta)]) * z_i

        # case with no core
        elif self.hc == 0:
            self.z_values = np.arange(-self.h/2, self.h/2 + self.mtr.hi, self.mtr.hi) + self.zc

            V = np.zeros(4)
            for i, angle in enumerate(self.angles):
                theta = np.radians(angle)
                z_i = self.z_values[i + 1] ** 3 - self.z_values[i] ** 3
                V += (1 / 3) * np.array(
                    [np.cos(2 * theta), np.cos(4 * theta), np.sin(2 * theta), np.sin(4 * theta)]) * z_i

        self.V = V
        V1, V2, V3, V4 = V

        U_vec = np.array([h_star, self.U_2, self.U_3])

        U_mat = np.array([[self.U_1,  V1,  V2],
                          [self.U_1, -V1,  V2],
                          [self.U_4,   0, -V2],
                          [self.U_5,   0, -V2],
                          [0,     0.5*V3,  V4],
                          [0,     0.5*V3, -V4], ])

        D = U_mat @ U_vec

        self.D_mat = np.array([[D[0], D[2], D[4]],
                               [D[2], D[1], D[5]],
                               [D[4], D[5], D[3]]])

        self.d_mat = np.linalg.inv(self.D_mat)

    def get_G(self):
        """
        Method to compute the shear tensor for a laminate
        """
        self.G_mat = self.h * np.array([[self.mtr.Gxz, 0],
                                        [0, self.mtr.Gyz], ])

    def get_zcoord(self):
        """
        Computes the bottom, middle top for each ply
        """
        bottom_coords = np.arange(-self.h/2, -self.hc/2 + self.mtr.hi/2, self.mtr.hi/2)
        top_coords = np.arange(self.hc/2, self.h/2 + self.mtr.hi/2, self.mtr.hi/2)
        self.zcoord = np.hstack([bottom_coords, top_coords])

    def get_A_FEM(self):
        """
        Computes the A matrix for a FEM problem with corrected poisson ratios
        """
        A_FEM = self.A_mat
        A_FEM *= np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        return A_FEM

    def apply_load(self, load):
        """
        With load N = [N1, N2, N3]
        """
        # Off axis strain
        self.eps_off += self.a_mat @ load

    def apply_moment(self, moment):
        """
        With moment M = [M1, M2, M3]
        """
        # curvature
        self.curvature += self.d_mat @ moment

    def compute_offstrain_total(self):
        pass


class LayupGenerator(Layup):
    """
    A class to generate random layups for a single material
    """
    layup_angles = np.arange(-90, 91, 5)
    angle_len = layup_angles.shape[0]

    def __init__(self, material=None, symetric=False):
        layup_len = np.random.randint(5, self.angle_len // 2)
        layup_indexes = [np.random.randint(self.angle_len - 1) for _ in range(layup_len)]
        plies = self.layup_angles[np.array(layup_indexes)]

        super().__init__(material, plies=plies, symetric=symetric)

