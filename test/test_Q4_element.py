import unittest
import sys
import numpy as np
import scipy.io
sys.path.append('../')
import FEMOL
import FEMOL.test_utils

class MyTestCase(unittest.TestCase):
    """
    Test class for the Q4 element class
    """

    def test_element_center(self):
        """
        Test the element.center() method
        """
        # element coordinates
        element_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        element = FEMOL.elements.Q4(element_points, N_dof=2)
        self.assertTrue(element.center() == (0, 0))

    def test_Q4_stiffness_2dof(self):
        """
        Tests that the element stiffness matrix is equal to the reference one from
        """
        material = FEMOL.materials.IsotropicMaterial(E=210e6, mu=0.3, rho=1)
        thickness = 0.0250
        C = material.plane_tensor(thickness)
        element_points = np.array([[0, 0], [0.5, 0], [0.5, 0.25], [0, 0.25]])
        element = FEMOL.elements.Q4(element_points, N_dof=2)

        self.assertTrue((np.abs(np.sum(element.Ke(C) - FEMOL.test_utils.reference_2dof_stiffness_matrix())) < 1000))

    def test_Q4_stiffness_6dof_plane_stress(self):
        """
        Tests that the plane stress part of the bending Ke is equal to the 2 dof Ke
        """
        # Problem definition
        element_points = np.array([[-0.1, -0.1], [0.7, -0.2], [0.9, 0.3], [0, 0.5]])
        material = FEMOL.materials.IsotropicMaterial(2, 0.3, 1.1)
        thickness = 0.1

        # create element instances
        plane_element = FEMOL.elements.Q4(element_points, N_dof=2)
        plate_element = FEMOL.elements.Q4(element_points, N_dof=6)

        # define the plane stiffness tensor
        C = material.plane_tensor(thickness)
        D = material.bending_tensor(thickness)
        G = material.shear_tensor(thickness)

        Ke_plane = plane_element.Ke(C)
        Ke_plate = plate_element.Ke(C, D, G)

        Ke_plate = FEMOL.test_utils.reshape_Ke_into_plane_stress(Ke_plate)

        self.assertTrue(np.allclose(Ke_plane, Ke_plate))

    def test_Q4_stiffness_6dof_bending(self):
        """
        Compare the bending part of the stiffness element matrix to a reference from :
        KSSV (2021). Plate Bending (https://www.mathworks.com/matlabcentral/fileexchange/32029-plate-bending),
        MATLAB Central File Exchange. Retrieved September 22, 2021.
        """
        # Reference element stiffness matrix in bending
        KB_REF = scipy.io.loadmat('reference_matrices/element_bending.mat')['kb']

        # Recreate the reference problem
        element_points = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
        material = FEMOL.materials.IsotropicMaterial(10920, 0.3, 1)
        thickness = 0.1

        element = FEMOL.elements.Q4(element_points, N_dof=6)
        C = np.zeros((3, 3))
        D = material.bending_tensor(thickness)
        G = np.zeros((2, 2))
        Ke = element.Ke(C, D, G)

        self.assertTrue(np.allclose(KB_REF, FEMOL.test_utils.reshape_Ke_into_bending(Ke)))

    def test_Q4_stiffness_6dof_shear(self):
        """
        Compares the shear part of the element stiffness matrix with a benchmark problem from :
        KSSV (2021). Plate Bending (https://www.mathworks.com/matlabcentral/fileexchange/32029-plate-bending),
        MATLAB Central File Exchange. Retrieved September 22, 2021.
        """
        # reference matrix
        KS_REF = scipy.io.loadmat('reference_matrices/element_shear.mat')['ks']

        # Recreate the reference problem
        element_points = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
        material = FEMOL.materials.IsotropicMaterial(10920, 0.3, 1)
        thickness = 0.1

        element = FEMOL.elements.Q4(element_points, N_dof=6)
        C = np.zeros((3, 3))
        D = np.zeros((3, 3))
        G = material.shear_tensor(thickness)
        Ke = element.Ke(C, D, G)

        self.assertTrue(np.allclose(FEMOL.test_utils.reshape_Ke_into_bending(Ke), KS_REF))

if __name__ == '__main__':
    unittest.main()
