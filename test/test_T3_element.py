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

    def test_T3_2dof_reference_solution(self):
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

if __name__ == '__main__':
    unittest.main()
