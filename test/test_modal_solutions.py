import unittest
import FEMOL
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_in_plane_disk_eigenvalues_quads(self):
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

    def test_in_plane_disk_eigenvalues_triangles(self):
        """
        Example with reference solution

        Reference values taken from :
        Park, C. I. (2008). Frequency equation for the in-plane vibration of a clamped circular plate.
        Journal of Sound and Vibration, 313(1‑2), 325‑333. https://doi.org/10.1016/j.jsv.2007.11.034
        """
        # Reference eigenvalues (Hz) (Park, C. I. (2008)).
        REF_W = np.array([3363.6, 3836.4, 5217.5, 5380.5,
                          6624, 6749.3, 6929, 7019.3, 8093,
                          8476.5, 8530.6, 9258, 9328.1, 9887.7])
        R = 0.5  # m
        N_ele = 10
        mesh = FEMOL.mesh.circle_T3(R, N_ele)
        thickness = 0.005
        aluminium = FEMOL.materials.IsotropicMaterial(71e9, 0.33, 2700)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
        problem.define_materials(aluminium)
        problem.define_tensors(thickness)
        circle_domain = FEMOL.domains.outside_circle(0, 0, R - 0.005)
        problem.add_fixed_domain(circle_domain)
        w, v = problem.solve(filtre=2)
        FEM_W = np.around(w, 1)
        DIFF = (FEM_W[:14] - REF_W)
        MEAN = 0.5 * ((FEM_W[:14] + REF_W))
        self.assertTrue(((100 * DIFF / MEAN)[:6] < 2).all())

    def test_supported_square_plate_eigenvalues(self):
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

    def test_cantilever_square_plate_eigenvalues(self):
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
        self.assertTrue(((100*np.abs(REF_W - FEM_W) / REF_W) < 4).all()
)

if __name__ == '__main__':
    unittest.main()
