import unittest
import sys
import numpy as np
sys.path.append('../')
import FEMOL


class MyTestCase(unittest.TestCase):

    def test_6dof_analytical_solution_plane_stress_iso(self):
        """
        Compare the 6 dof bending model in plane stress to an analytical solution
        """
        # Plate problem
        thickness = 0.1
        Lx, Ly = 20, 10
        F = 0.5

        mesh = FEMOL.mesh.rectangle_Q4(20, 10, 60, 30)
        problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        material = FEMOL.materials.IsotropicMaterial(2, 0.3, 1)
        problem.define_materials(material)
        problem.define_tensors(thickness)
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0])
        fixed_domain = FEMOL.domains.inside_box([0], [Ly / 2])
        problem.add_fixed_domain(fixed_domain, ddls=[1])

        force_domain_1 = FEMOL.domains.inside_box([Lx], [[0.01, Ly - 0.01]])
        problem.add_forces([F, 0, 0, 0, 0, 0], force_domain_1)
        force_domain_2 = FEMOL.domains.inside_box([Lx], [0, Ly])
        problem.add_forces([F / (30 - 1), 0, 0, 0, 0, 0], force_domain_2)

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

        self.assertTrue(np.isclose(eps_x, FEM_eps_x))
        self.assertTrue(np.isclose(eps_y, FEM_eps_y))

    def test_6dof_analytical_solution_bending_iso(self):
        """
        Benchmark problem uniform pressure on unit square plate
        max defelxion = -0.004270
        """
        Lx, Ly, n, m = 1, 1, 20, 20

        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, n, m)
        material = FEMOL.materials.isotropic_bending_benchmark()
        t = 0.1

        problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)

        problem.define_materials(material)
        problem.define_tensors(t)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0, Lx], [[0, Ly]]), ddls=[2, 4])
        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, Lx]], [0, Ly]), ddls=[2, 3])

        Force = -1 * ((n - 1) ** 2 / n ** 2)
        print('Force : ', Force)

        problem.add_forces([0, 0, Force, 0, 0, 0],
                           FEMOL.domains.inside_box([[0 + 0.01, Lx - 0.01]], [[0 + 0.01, Ly - 0.01]]))
        problem.assemble('K')

        mesh = problem.solve(verbose=False)
        d_max = mesh.point_data['Uz'].min()
        self.assertTrue(np.isclose(d_max, -0.004270, 1e-4))

    def test_6dof_benchmark_solution_bending_laminate(self):
        """
        Comparative test for deflexion with reference solution from NAFEMS R0031/1
        National Agency for Finite Element Methods and Standards (U.K.):
        Test R0031/1 from NAFEMS publication R0031, “Composites Benchmarks,” February 1995.
        """
        Lx = 50 / 1000  # mm to m
        Ly = 10 / 1000  # mm to m

        ny = 12  # fixed value
        nx = ny * 5  # fixed ratio

        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, nx, ny)

        material = FEMOL.materials.abaqus_benchmark()
        plies = [0, 90, 0, 90, 90]
        layup = FEMOL.laminate.Layup(material=material, plies=plies, symetric=True)

        problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        problem.define_materials(material)
        problem.define_tensors(layup)

        eps = 0
        domain = FEMOL.domains.inside_box([[0.01 - eps, 0.01 + eps], [0.04 - eps, 0.04 + eps]], [[0, Ly]])
        problem.add_fixed_domain(domain, ddls=[2])

        force = 10 * 10  # N
        force_domain = FEMOL.domains.inside_box([[Lx / 2 - eps, Lx / 2 + eps]], [[0, Ly]])
        problem.add_forces([0, 0, -force, 0, 0, 0], force_domain)

        mesh = problem.solve(verbose=False)

        # Compare to reference solution = -1.06 mm
        self.assertTrue(np.isclose(mesh.point_data['Uz'].min() * 1000, -1.06, 1e-1))

    def test_6dof_analytical_solution_bending_laminate(self):
        """
        Test comparing FEMOL result with an analytical solution based on
        laminate theory, the problem is the assignment 4 from MECH530 class
        (Lessard, 2020)
        """
        # Problem data
        # Layup de travail
        P = -250 * 9.8  # N
        L = 550 / 1000  # m
        b = 100 / 1000  # m
        mesh = FEMOL.mesh.rectangle_Q4(L, b, 10, 50)
        plies = [0, 0, 0, 20, -20, 90]  # s

        # T300_N5208 material from MECH530 course
        material = FEMOL.materials.T300_N5208()
        # Symetric layup with 10 cm core
        layup = FEMOL.laminate.Layup(material=material, plies=plies, symetric=True, h_core=0.01)
        # analytical deflexion
        THE_deflexion = 1000 * (P * L ** 3) / (48 * b) * layup.d_mat[0, 0]

        # FEM solution
        problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        problem.define_materials(material)
        problem.define_tensors(layup)

        # Fixed domain
        domain = FEMOL.domains.inside_box([0, L], [[0, b]])
        problem.add_fixed_domain(domain, ddls=[2])
        # Force domain
        force_domain = FEMOL.domains.inside_box([[L / 2, L / 2]], [[0, b]])
        problem.add_forces([0, 0, P, 0, 0, 0], force_domain)
        problem.assemble('K')
        mesh = problem.solve(verbose=False)
        FEM_deflexion = mesh.point_data['Uz'].min() * 1000

        self.assertTrue(np.isclose(FEM_deflexion, THE_deflexion, 1e-2))


if __name__ == '__main__':
    unittest.main()
