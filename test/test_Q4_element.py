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
    def test_Q4_element_center(self):
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

    def test_Q4_inplane_modal_disk_solution(self):
        """
        Test cas with reference solution of in plane vibration
        natural frequencies of a clamped circular plate

        Reference values taken from :
        Park, C. I. (2008). Frequency equation for the in-plane vibration of a clamped circular plate.
        Journal of Sound and Vibration, 313(1‑2), 325‑333. https://doi.org/10.1016/j.jsv.2007.11.034
        """
        # Reference eigenvalues (Hz) (Park, C. I. (2008)).
        REF_W = np.array([3363.6, 3836.4, 5217.5, 5380.5,
                          6624, 6749.3, 6929, 7019.3, 8093,
                          8476.5, 8530.6, 9258, 9328.1, 9887.7])

        # Circle mesh with R = 0.5 and ~25 ** 2 elements
        R = 0.5  # m
        N_ele = 25
        mesh = FEMOL.mesh.circle_Q4(R, N_ele)

        # Problem definition
        thickness = 0.005
        aluminium = FEMOL.materials.IsotropicMaterial(71e9, 0.33, 2700)

        # Create a FEM Problem from the mesh (compute displacement with a plate bending model)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
        problem.define_materials(aluminium)
        problem.define_tensors(thickness)

        circle_domain = FEMOL.domains.outside_circle(0, 0, R - 0.005)

        # Fix all the degrees of freedom
        problem.add_fixed_domain(circle_domain)

        # Solve the eigenvalue problem
        w, _ = problem.solve(verbose=False, filtre=2)

        i = REF_W.shape[0]
        relative_difference = 100 * np.abs(REF_W - w[:i]) / (REF_W + w[:i])/2

        # Assert the difference between the eigen values is lower than 1%
        self.assertTrue((np.array(relative_difference) < 1).all())

    def test_Q4_supported_square_modal_solution(self):
        """
        Test eigenvalues values for bending modal analysis
        of a simply supported  square plate

        Reference values taken from :
        NAFEMS Finite Element Methods & Standards, Abbassian, F., Dawswell, D. J., and Knowles, N. C.,
        Selected Benchmarks for Natural Frequency Analysis. Glasgow: NAFEMS, Nov., 1987. Test No. 13.
        """
        # Simply supported
        REF_W = np.array([2.377, 5.942, 5.942, 9.507, 11.884, 11.884, 15.449, 15.449])

        mesh = FEMOL.mesh.rectangle_Q4(10, 10, 15, 15)

        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')

        material = FEMOL.materials.IsotropicMaterial(200e9, 0.3, 8000)
        problem.define_materials(material)
        problem.define_tensors(0.05)

        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [[0, 10]]), ddls=[0, 1, 5])
        problem.add_fixed_domain(FEMOL.domains.inside_box([0, 10], [[0, 10]]), ddls=[2, 4])
        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [0, 10]), ddls=[2, 3])

        w, v = problem.solve(verbose=False, filtre=1)

        diff = 100 * np.abs(REF_W - w[:8]) / w[:8]

        self.assertTrue((diff < 5).all())

    def test_Q4_cantilever_square_modal_solution(self):
        """
        Test comparing the cantilever plate eigenvalues to reference
        values

        Reference values taken from :
        NAFEMS Finite Element Methods & Standards, Abbassian, F., Dawswell, D. J., and Knowles, N. C.,
        Selected Benchmarks for Natural Frequency Analysis. Glasgow: NAFEMS, Nov., 1987. Test No. 11a.
        """
        REF_W = np.array([0.421, 2.582, 3.306, 6.555, 7.381, 11.402])

        # Cantiever plate
        mesh = FEMOL.mesh.rectangle_Q4(10, 10, 15, 15)

        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')

        material = FEMOL.materials.IsotropicMaterial(200e9, 0.3, 8000)
        problem.define_materials(material)
        problem.define_tensors(0.05)

        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [[0, 10]]), ddls=[0, 1])
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, 10]]))

        w, v = problem.solve(verbose=False, filtre=1)

        # Take only the symetric modes
        FEM_W = w[[0, 2, 3, 5, 6, 9]]

        # Test that the error is below 4%
        self.assertTrue(((100 * np.abs(REF_W - FEM_W) / REF_W) < 4).all())

    def test_area_method(self):
        """
        Test that the area method returns the right value
        """
        points = np.array([[0, 0],
                           [1, 0],
                           [1, 1],
                           [0, 1]])
        element = FEMOL.elements.Q4(points)
        self.assertTrue(element.area() == 1)


if __name__ == '__main__':
    unittest.main()
