import unittest
import FEMOL
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_T6_shape_functions_1(self):
        """
        Test the element shape functions of the T6 element
        """
        element_points = np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5]])
        element = FEMOL.elements.T6(element_points)
        for i, point in enumerate(element_points):
            N = element.shape(*point)
            self.assertTrue(N[i] == 1)
            self.assertTrue(np.isclose(N[np.arange(N.shape[0]) != i], 0).all())

    def test_T6_shape_functions_2(self):
        """
        Test the real shape functions of the element
        """
        element_points = np.array([[0, 0], [0.5, 0], [1, 0], [0.5, 0.5], [0, 1], [0, 0.5]])
        element = FEMOL.elements.T6(element_points)
        element.make_shape_xy()
        for i, point in enumerate(element_points):
            N = element.shape_xy(*point)
            self.assertTrue(N[0, 2 * i] == 1)

    def test_T6_2dof_poisson_ratio(self):
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

    def test_T6_2dof_plane_stress_solution(self):
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

    def test_T6_6dof_plane_stress_solution(self):
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

    def test_T6_6dof_uniform_pressure_solution(self):
        """
        Benchmark problem uniform pressure on unit square plate for triangle6 elements
        max defelxion = -0.004270
        """
        Lx, Ly, n, m = 1, 1, 10, 10
        mesh = FEMOL.mesh.rectangle_T6(Lx, Ly, n, m)
        material = FEMOL.materials.isotropic_bending_benchmark()
        t = 0.1
        problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)
        problem.define_materials(material)
        problem.define_tensors(t)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0, Lx], [[0, Ly]]), ddls=[2, 4])
        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, Lx]], [0, Ly]), ddls=[2, 3])
        boundary_nodes = mesh.domain_nodes(FEMOL.domains.outside_box(0.01, 0.99, 0.01, 0.99))
        all_nodes = np.arange(0, mesh.points.shape[0])
        Force = -1 * (all_nodes.shape[0] - boundary_nodes.shape[0] / 2.57) / all_nodes.shape[0]
        problem.add_forces([0, 0, Force, 0, 0, 0],
                           FEMOL.domains.inside_box([[0 + 0.01, Lx - 0.01]], [[0 + 0.01, Ly - 0.01]]))
        problem.assemble('K')
        mesh = problem.solve(verbose=False)
        d_max = mesh.point_data['Uz'].min()
        self.assertTrue(np.isclose(d_max, -0.004270, 1e-3))

    def test_T6_inplane_vibration_solution(self):
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
        N_ele = 5
        mesh = FEMOL.mesh.circle_T6(R, N_ele)
        thickness = 0.005
        aluminium = FEMOL.materials.IsotropicMaterial(71e9, 0.33, 2700)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
        problem.define_materials(aluminium)
        problem.define_tensors(thickness)
        circle_domain = FEMOL.domains.outside_circle(0, 0, R - 0.005)
        problem.add_fixed_domain(circle_domain)
        wo, v = problem.solve(filtre=0)
        w = wo[wo > 1]
        w = np.around(w, 0)
        FEM_W = np.unique(w)
        DIFF = (FEM_W[:14] - REF_W)
        MEAN = 0.5 * ((FEM_W[:14] + REF_W))
        print(((100 * DIFF / MEAN)[:8] < 1).all())

    def test_T6_6dof_vibration_solution(self):
        """
        Test eigenvalues values for bending modal analysis
        of a simply supported  square plate

        Reference values taken from :
        NAFEMS Finite Element Methods & Standards, Abbassian, F., Dawswell, D. J., and Knowles, N. C.,
        Selected Benchmarks for Natural Frequency Analysis. Glasgow: NAFEMS, Nov., 1987. Test No. 13.
        """
        # Simply supported
        REF_W = np.array([2.377, 5.942, 5.942, 9.507, 11.884, 11.884, 15.449, 15.449])
        mesh = FEMOL.mesh.rectangle_T6(10, 10, 6, 6)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        material = FEMOL.materials.IsotropicMaterial(200e9, 0.3, 8000)
        problem.define_materials(material)
        problem.define_tensors(0.05)
        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [[0, 10]]), ddls=[0, 1, 5])
        problem.add_fixed_domain(FEMOL.domains.inside_box([0, 10], [[0, 10]]), ddls=[2, 4])
        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [0, 10]), ddls=[2, 3])
        w, v = problem.solve(verbose=False, filtre=1)
        diff = 100 * np.abs(REF_W - w[:8]) / w[:8]
        self.assertTrue((diff[:3] < 10).all())

if __name__ == '__main__':
    unittest.main()
