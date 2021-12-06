import unittest
import sys
import numpy as np
sys.path.append('../')
import FEMOL.elements
import FEMOL.test_utils

class MyTestCase(unittest.TestCase):
    """
    Tests for the T3 element
    """
    def test_T3_2dof_stiffness(self):
        """
        Test the element stifness matrix computation for the T3 element
        Data taken from :
        Kattan, P. I. (2007). MATLAB guide to finite elements: an interactive approach (2nd ed).
        Berlin ; New York : Springer. Example 11.1.

        """
        # Define the element proprieties
        thickness = 0.025  # m
        E = 210e9  # Pa
        mu = 0.3
        material = FEMOL.materials.IsotropicMaterial(E=210e9, mu=0.3, rho=1)
        element_points = np.array([[0, 0, 0],
                                   [0.5, 0.25, 0],
                                   [0, 0.25, 0]])

        element = FEMOL.elements.T3(element_points, N_dof=2)
        Ke = element.Ke(material.plane_tensor(thickness))
        Ke = np.around(Ke/1e9, 4)
        Ke_ref = FEMOL.test_utils.reference_T3_2dof_stiffnes_matrix()

        self.assertTrue(np.allclose(Ke, Ke_ref, 1e-4))

    def test_T3_2dof_inplane_displacement_solution(self):
        """
        Tests the T3 element with a reference example taken from :
        Kattan, P. I. (2007). MATLAB guide to finite elements:
        an interactive approach (2nd ed). Berlin ; New York : Springer.
        Example 11.1
        """
        # Problem definition
        thickness = 0.025
        material = FEMOL.materials.IsotropicMaterial(E=210e9, mu=0.3, rho=1)
        Lx, Ly = 0.5, 0.25
        mesh = FEMOL.mesh.rectangle_T3(Lx, Ly, 1, 1)

        problem = FEMOL.FEM_Problem('displacement', 'plane', mesh)
        problem.define_materials(material)
        problem.define_tensors(thickness)

        # Fixed domain
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain)

        # Forces
        F = 9375 * 2
        force_domain = FEMOL.domains.inside_box([Lx], [[0, Ly]])
        problem.add_forces([F, 0], force_domain)

        # Assemble / Solve
        problem.assemble('K')
        mesh = problem.solve()

        # Validate solution
        self.assertTrue(np.allclose(mesh.U[mesh.U != 0] / 1e-5, [0.7111, 0.1115, 0.6531, 0.0045], 1e-2))

    def test_T3_in_plane_disk_modal_solution(self):
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
        w, v = problem.solve(filtre=0)
        w = w[w > 1]
        w = np.around(w, 0)
        w = np.unique(w)
        FEM_W = w
        DIFF = (FEM_W[:14] - REF_W)
        MEAN = 0.5 * ((FEM_W[:14] + REF_W))
        self.assertTrue(((100 * DIFF / MEAN)[:6] < 2).all())

if __name__ == '__main__':
    unittest.main()
