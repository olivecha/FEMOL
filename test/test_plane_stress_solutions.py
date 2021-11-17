import unittest
import numpy as np
import sys
sys.path.append('../')
import FEMOL

class MyTestCase(unittest.TestCase):

    def test_2dof_isotropic_poisson_ratio(self):
        """
        Test the relation that y strain is equal to x strain * poisson ratio for an isotropic
        material subject to stress in one direction
        """
        n = 20
        Lx, Ly = 1, 1
        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, n, n)
        problem = FEMOL.core.FEM_Problem('displacement', 'plane', mesh)
        material = FEMOL.materials.general_isotropic()
        problem.define_materials(material)
        problem.define_tensors(1)

        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0])

        fixed_domain = FEMOL.domains.inside_box([0], [Ly / 2])
        problem.add_fixed_domain(fixed_domain, ddls=[1])

        force_domain = FEMOL.domains.inside_box([Lx], [[0.01, Ly - 0.01]])
        F = 0.05

        problem.add_forces([F, 0], force_domain)

        force_domain = FEMOL.domains.inside_box([Lx], [0, Ly])
        problem.add_forces([F / (n - 1), 0], force_domain)

        problem.assemble('K')

        mesh = problem.solve(verbose=False)

        # compute problem strain
        FEM_eps_x = (mesh.point_data['Ux'].max() - mesh.point_data['Ux'].min()) / 20
        FEM_eps_y = -(mesh.point_data['Uy'].max() - mesh.point_data['Uy'].min()) / 20

        self.assertTrue(np.isclose(-FEM_eps_y / material.mu, FEM_eps_x))

    def test_2dof_analytical_solution_tension(self):
        thickness = 0.1
        Lx, Ly = 20, 10
        F = 0.5

        mesh = FEMOL.mesh.rectangle_Q4(20, 10, 60, 30)
        problem = FEMOL.core.FEM_Problem('displacement', 'plane', mesh)
        material = FEMOL.materials.general_isotropic()
        problem.define_materials(material)
        problem.define_tensors(thickness)
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0])

        fixed_domain = FEMOL.domains.inside_box([0], [Ly / 2])
        problem.add_fixed_domain(fixed_domain, ddls=[1])

        force_domain_1 = FEMOL.domains.inside_box([Lx], [[0.01, Ly - 0.01]])
        problem.add_forces([F, 0], force_domain_1)
        force_domain_2 = FEMOL.domains.inside_box([Lx], [0, Ly])
        problem.add_forces([F / (30 - 1), 0], force_domain_2)

        problem.assemble('K')

        mesh = problem.solve(verbose=False)

        # compute problem strain
        FEM_eps_x = (mesh.point_data['Ux'].max() - mesh.point_data['Ux'].min()) / 20
        FEM_eps_y = -(mesh.point_data['Uy'].max() - mesh.point_data['Uy'].min()) / 10

        # True value for strain
        A = thickness * Ly
        sigma = np.sum(problem.F) / A
        eps_x = (sigma / material.E)
        eps_y = eps_x * -0.3

        self.assertTrue(np.isclose(FEM_eps_x, eps_x))
        self.assertTrue(np.isclose(FEM_eps_y, eps_y))

    def test_2dof_analytical_solution_bending(self):
        n = 100
        m = n // 4
        thickness = 2
        F = -0.5

        Lx, Ly = 40, 2
        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, n, m)
        problem = FEMOL.core.FEM_Problem('displacement', 'plane', mesh)
        material = FEMOL.materials.IsotropicMaterial(500, 0.3, 1)
        problem.define_materials(material)
        problem.define_tensors(thickness)

        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0, 1])

        fixed_domain = FEMOL.domains.inside_box([Lx], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0, 1])

        force_domain = FEMOL.domains.inside_box([Lx / 2], [Ly])
        problem.add_forces([0, F], force_domain)

        problem.assemble('K')

        mesh = problem.solve(verbose=False)

        # True value for deflexion
        I = thickness ** 4 / 12
        deflexion = F * (Lx / 2) ** 3 * (Lx / 2) ** 3 / (3 * Lx ** 3 * material.E * I)

        self.assertTrue(np.isclose(mesh.point_data['Uy'].min(), deflexion, atol=0.005))

    def test_2dof_analytical_solution_laminate(self):
        """
        Test concordance with laminate theory for 20 random layups
        """
        i = 0
        while i <= 10:
            i += 1
            L, n = 1, 2
            mesh = FEMOL.mesh.rectangle_Q4(L, L, n, n)

            material = FEMOL.materials.random_laminate_material()
            layup = FEMOL.laminate.Layup(material=material, plies=[0, -45, 90, 90, 45, 0])

            problem = FEMOL.core.FEM_Problem('displacement', 'plane', mesh)
            problem.define_materials(material)
            problem.define_tensors(layup)

            # Fix the left side
            fixed_domain = FEMOL.domains.inside_box([0], [[0, L]])
            problem.add_fixed_domain(fixed_domain, ddls=[0])
            fixed_domain = FEMOL.domains.inside_box([0], [L / 2])
            problem.add_fixed_domain(fixed_domain, ddls=[1])

            F = 10000000
            Fi = F / n  # n = number of nodes  - 1
            force_domain_1 = FEMOL.domains.inside_box([L], [[0.01, L - 0.01]])
            problem.add_forces([Fi * (n - 1), 0], force_domain_1)
            force_domain_2 = FEMOL.domains.inside_box([L], [0, L])
            problem.add_forces([Fi, 0], force_domain_2)

            problem.assemble('K')
            mesh = problem.solve(verbose=False)

            # compute problem strain
            FEM_eps_x = (mesh.point_data['Ux'].max() - mesh.point_data['Ux'].min())
            Uy = mesh.point_data['Uy'].reshape((n + 1, n + 1))
            FEM_eps_y = 2 * Uy[:, 0].min()

            a = layup.a_mat
            eps_real = a @ np.array([F / L, 0, 0])

            self.assertTrue(np.isclose(FEM_eps_y, eps_real[1]))
            self.assertTrue(np.isclose(FEM_eps_x, eps_real[0]))


if __name__ == '__main__':
    unittest.main()
