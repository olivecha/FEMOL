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

        if (not self.sym) & (len(self.angles)%2 != 0) & (self.hc > 0):
            print('unable to build layup')
            raise ValueError

        # Get layup thickness
        self.Nb_plies = self.angles.shape[0]
        self.hA = self.Nb_plies * self.mtr.hi
        self.h = self.hA + self.hc

        # Compute the z values
        self.z_values = self.get_z_values()

        # Off-axis Q matrices
        self.Q_off = [self.mtr.Q_mat(angle) for angle in self.angles]

        # Off-axis S matrices
        self.S_off = [self.mtr.S_mat(angle) for angle in self.angles]

        # A matrix
        self.A_mat, self.a_mat = self.get_A(return_a=True)

        # B matrix
        self.B_mat = self.get_B()

        # D matrix
        self.D_mat, self.d_mat = self.get_D(return_d=True)

        # Out of plane shear matrix
        self.get_G()

    def get_A(self, return_a=False):
        """
        A matrix computation for a given laminate
        """
        A_mat = 0
        # Integral for the bottom of the laminate
        for j, theta in enumerate(self.angles[:len(self.angles)//2]):
            zi1 = self.z_values[j]
            zi2 = self.z_values[j+1]
            A_mat += self.mtr.Q_mat(theta) * (zi2 - zi1)

        # Integral for the top of the laminate
        for i, theta in enumerate(self.angles[len(self.angles)//2:]):
            i += len(self.angles)//2 + 1*(self.hc > 0)
            zi1 = self.z_values[i]
            zi2 = self.z_values[i+1]
            A_mat += self.mtr.Q_mat(theta) * (zi2 - zi1)

        if return_a:
            return A_mat, np.linalg.inv(A_mat)
        else:
            return A_mat

    def get_B(self, return_b=False):
        """
        B matrix for coupled bending formulation
        """
        B_mat = 0
        # Integral for the bottom of the laminate
        for j, theta in enumerate(self.angles[:len(self.angles)//2]):
            zi1 = self.z_values[j]
            zi2 = self.z_values[j+1]
            B_mat += self.mtr.Q_mat(theta) * (zi2**2 - zi1**2) * (1/2)

        # Integral for the top of the laminate
        for i, theta in enumerate(self.angles[len(self.angles)//2:]):
            i += len(self.angles)//2 + 1*(self.hc > 0)
            zi1 = self.z_values[i]
            zi2 = self.z_values[i+1]
            B_mat += self.mtr.Q_mat(theta) * (zi2**2 - zi1**2) * (1/2)

        if return_b:
            return B_mat, np.linalg.inv(B_mat)
        else:
            return B_mat

    def get_D(self, return_d=False):
        """
        Alternative computation for D matrix
        """
        D_mat = 0
        # Integral for the bottom of the laminate
        for j, theta in enumerate(self.angles[:len(self.angles)//2]):
            zi1 = self.z_values[j]
            zi2 = self.z_values[j+1]
            D_mat += self.mtr.Q_mat(theta) * (zi2**3 - zi1**3) * (1/3)

        # Integral for the top of the laminate
        for i, theta in enumerate(self.angles[len(self.angles)//2:]):
            i += len(self.angles)//2 + 1*(self.hc > 0)
            zi1 = self.z_values[i]
            zi2 = self.z_values[i+1]
            D_mat += self.mtr.Q_mat(theta) * (zi2**3 - zi1**3) * (1/3)

        if return_d:
            return D_mat, np.linalg.inv(D_mat)
        else:
            return D_mat

    def get_G(self):
        """
        Method to compute the shear tensor for a laminate
        """
        kappa = 5/6
        self.G_mat = kappa * self.h * np.array([[self.mtr.Gxz, 0],
                                               [0, self.mtr.Gyz], ])

    def get_z_values(self):
        """
        Compute the ply coordinates in the z direction
        """
        if self.hc == 0:
            z_values = np.arange(-self.h/2, self.h/2 + self.mtr.hi/2, self.mtr.hi)
            return z_values + self.zc
        elif self.hc > 0:
            z_values_b = np.arange(-self.h/2, -self.hc/2 + self.mtr.hi/2, self.mtr.hi)
            z_values_t = np.arange(self.hc/2, self.h/2 + self.mtr.hi/2, self.mtr.hi)
            z_values = np.hstack([z_values_b, z_values_t])
            return z_values + + self.zc

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


def porosity(h, w, t, m, phi, rho_f, rho_m):
    """
    Computes the porosity of a manufactured laminate using eq. 3 in:
    Monti, A., El Mahi, A., Jendli, Z., & Guillaumat, L. (2016).
    Mechanical behaviour and damage mechanisms analysis of a flax-fibre reinforced composite
    by acoustic emission. Composites Part A: Applied Science and Manufacturing, 90, 100‑110.
    https://doi.org/10.1016/j.compositesa.2016.07.002

    :param h: height of the part (mm)
    :param w: width of the part (mm)
    :param t: thickness of the part (mm)
    :param m: mass of the part (g)
    :param phi: theoretical fibre fraction
    :return: porosity
    """

    # Compute the volume of the part
    V = (h*w*t)/1000  # cm^3
    # Compute the density of the part
    rho_c = m/V  # g/cm^3
    # Compute the part porosity
    void = 1 - rho_c * (phi/rho_f + (1 - phi)/rho_m)
    return void

