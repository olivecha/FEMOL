import unittest
import FEMOL
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_shape_functions(self):
        """
        Test the element shape functions of the T6 element
        """
        element_points = np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5]])
        element = FEMOL.elements.T6(element_points)
        for i, point in enumerate(element_points):
            N = element.shape(*point)
            self.assertTrue(N[i] == 1)
            self.assertTrue(np.isclose(N[np.arange(N.shape[0]) != i], 0).all())

    def test_2dof_poisson_ration(self):
        """
        Test the element poisson ration in plane stress
        """
        # Plate problem
        thickness = 0.1
        Lx, Ly = 20, 10
        nelx, nely = 4, 4
        F = 0.5
        mesh = FEMOL.mesh.rectangle_T6(Lx, Ly, nelx, nely)
        problem = FEMOL.FEM_Problem('displacement', 'plane', mesh)
        material = FEMOL.materials.IsotropicMaterial(2, 0.3, 1)
        problem.define_materials(material)
        problem.define_tensors(thickness)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, Ly]]), ddls=[0])
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [Ly / 2]), ddls=[1])
        Fi = F / (2 * nely)
        force_domain_1 = FEMOL.domains.inside_box([Lx], [[0.5, Ly - 0.5]])
        problem.add_forces([Fi * (2 * nely - 1), 0, 0, 0, 0, 0], force_domain_1)
        problem.add_forces([Fi, 0, 0, 0, 0, 0], FEMOL.domains.inside_box([Lx], [0, Ly]))
        mesh = problem.solve(verbose=False)

        # compute problem strain
        FEM_eps_x = (mesh.point_data['Ux'].max() - mesh.point_data['Ux'].min()) / 20
        FEM_eps_y = -(mesh.point_data['Uy'].max() - mesh.point_data['Uy'].min()) / 10

        self.assertTrue(np.isclose(-FEM_eps_y/material.mu, FEM_eps_x, 1))

    def test_2dof_plane_stress_solution(self):
        """
        Test the element performance against a plane stress solution
        """
        # Plate problem
        thickness = 0.1
        Lx, Ly = 20, 10
        nelx, nely = 4, 3
        F = 0.5
        mesh = FEMOL.mesh.rectangle_T6(Lx, Ly, nelx, nely)
        problem = FEMOL.FEM_Problem('displacement', 'plane', mesh)
        material = FEMOL.materials.IsotropicMaterial(2, 0.3, 1)
        problem.define_materials(material)
        problem.define_tensors(thickness)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, Ly]]), ddls=[0])
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [Ly / 2]), ddls=[1])
        Fi = F / (2 * nely)
        force_domain_1 = FEMOL.domains.inside_box([Lx], [[0.5, Ly - 0.5]])
        problem.add_forces([Fi * (2 * nely - 1), 0, 0, 0, 0, 0], force_domain_1)
        problem.add_forces([Fi, 0, 0, 0, 0, 0], FEMOL.domains.inside_box([Lx], [0, Ly]))
        mesh = problem.solve(verbose=False)

        # compute problem strain
        FEM_eps_x = (mesh.point_data['Ux'].max() - mesh.point_data['Ux'].min()) / 20
        FEM_eps_y = -(mesh.point_data['Uy'].max() - mesh.point_data['Uy'].min()) / 10
        # True value for strain
        A = thickness * Ly
        sigma = np.sum(problem.F) / A
        eps_x = (sigma / material.E)
        eps_y = eps_x * -0.3
        self.assertTrue(np.isclose(eps_y, FEM_eps_y, 1))
        self.assertTrue(np.isclose(eps_x, FEM_eps_x, 1))

    def test_6dof_plane_stress_solution(self):
        """
        Test the element against the analytical solution to a plane stress
        problem with the 6 dof model
        """
        # Plate problem
        thickness = 0.1
        Lx, Ly = 10, 10
        nelx, nely = 1, 1
        F = 0.5
        mesh = FEMOL.mesh.rectangle_T6(Lx, Ly, nelx, nely)
        problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        material = FEMOL.materials.IsotropicMaterial(2, 0.3, 1)
        problem.define_materials(material)
        problem.define_tensors(thickness)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, Ly]]), ddls=[0])
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [Ly / 2]), ddls=[1])
        Fi = F/2
        force_domain_1 = FEMOL.domains.inside_box([Lx], [[0.5, Ly - 0.5]])
        problem.add_forces([Fi, 0, 0, 0, 0, 0], force_domain_1)
        problem.add_forces([Fi/2, 0, 0, 0, 0, 0], FEMOL.domains.inside_box([Lx], [0, Ly]))
        mesh = problem.solve(verbose=False)
        # compute problem strain
        FEM_eps_x = (mesh.point_data['Ux'].max() - mesh.point_data['Ux'].min()) / 20
        FEM_eps_y = -(mesh.point_data['Uy'].max() - mesh.point_data['Uy'].min()) / 10
        # True value for strain
        A = thickness * Ly
        sigma = np.sum(problem.F) / A
        eps_x = (sigma / material.E)
        eps_y = eps_x * -0.3
        print(np.isclose(eps_y, FEM_eps_y, 1))
        print(np.isclose(eps_x, FEM_eps_x, 1))

if __name__ == '__main__':
    unittest.main()
