import unittest
import FEMOL
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_2dof_ordered_mesh_SIMP_compliance(self):
        # Mesh
        Lx = 30
        Ly = 15

        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, Lx, Ly)

        # FEM Problem
        plate_FEM = FEMOL.FEM_Problem('displacement', 'plane', mesh)

        # Define the composite material layups
        material = FEMOL.materials.general_isotropic()

        plate_FEM.define_materials(material)
        plate_FEM.define_tensors(1)

        # Define the boundary conditions
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly / 4]])
        plate_FEM.add_fixed_domain(fixed_domain, ddls=[0, 1])
        fixed_domain = FEMOL.domains.inside_box([0], [[3 * Ly / 4, Ly]])
        plate_FEM.add_fixed_domain(fixed_domain, ddls=[0, 1])

        # Define the applied force
        force = [0, -0.1]
        force_domain = FEMOL.domains.inside_box([Lx], [[Ly / 2 - 1, Ly / 2 + 1]])
        plate_FEM.add_forces(force, force_domain)

        topo_problem = FEMOL.SIMP_COMP(plate_FEM, volfrac=0.4, penal=3)
        mesh = topo_problem.solve(converge=0.03, max_iter=1, plot=False, save=False)
        X = np.array(list(mesh.cell_data['X'].values())).flatten()
        self.assertTrue(np.isclose(np.sum(X), mesh.N_ele*0.4, 1e-2))

    def test_2dof_unordered_mesh_SIMP_compliance(self):
        """
        Test the SIMP Compliance minimization for an unordered mesh
        """
        mesh = FEMOL.mesh.circle_Q4(1, 10)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')
        composite_material = FEMOL.materials.general_carbon()
        plies = [0, 45, -45, -45, 45, 0]  # total plies
        layup = FEMOL.laminate.Layup(composite_material, plies=plies, symetric=False)

        # define the material and tensors the stiffness tensors are computed from the layup
        problem.define_materials(composite_material)
        problem.define_tensors(layup)

        # Fix all the degrees of freedom
        problem.add_fixed_domain(FEMOL.domains.outside_circle(0, 0, np.abs(mesh.points.max()) - 0.01))

        # Create a force domain at the center
        force_domain = FEMOL.domains.inside_box([[-0.09, 0.09]], [[-0.09, 0.09]])
        Fx = -1e3  # F = -10 kN
        F = [Fx, 0]  # F = [Fx, Fy]
        problem.add_forces(F, force_domain)

        # TOM Problem solve
        SIMP = FEMOL.SIMP_COMP(problem)
        mesh = SIMP.solve(converge=0.03, max_iter=3, plot=False, save=False)
        X = np.array(list(mesh.cell_data['X'].values())).flatten()
        self.assertTrue(np.isclose(np.sum(X), mesh.N_ele*0.5, 1))

    def test_2dof_ordered_mesh_SIMP_coating_compliance(self):
        # Mesh
        Lx = 30
        Ly = 15

        mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, Lx, Ly)

        # FEM Problem
        problem = FEMOL.FEM_Problem('displacement', 'plane', mesh)

        # Define the composite material layups
        carbon = FEMOL.materials.general_carbon()
        plies_base = [0, 90, 45, -45]
        plies_coat = [45, 45]
        layup_base = FEMOL.laminate.Layup(plies=plies_base, material=carbon, symetric=True)
        layup_coat = FEMOL.laminate.Layup(plies=plies_coat, material=carbon, symetric=True)

        problem.define_materials(carbon, carbon)
        problem.define_tensors(layup_base, layup_coat)

        # Define the boundary conditions
        fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly / 4]])
        problem.add_fixed_domain(fixed_domain, ddls=[0, 1])
        fixed_domain = FEMOL.domains.inside_box([0], [[3 * Ly / 4, Ly]])
        problem.add_fixed_domain(fixed_domain, ddls=[0, 1])

        # Define the applied force
        force = [0, -1e6]
        force_domain = FEMOL.domains.inside_box([Lx], [[Ly / 2 - 1, Ly / 2 + 1]])
        problem.add_forces(force, force_domain)

        topo_problem = FEMOL.SIMP_COMP(problem, volfrac=0.4, penal=3)
        mesh = topo_problem.solve(converge=0.01, max_iter=3, plot=False, save=False)
        X = np.array(list(mesh.cell_data['X'].values())).flatten()
        X = X.reshape(Lx, Ly)
        # Check for asymmetry
        self.assertFalse(np.isclose(np.sum(X[:, :Ly // 2]), np.sum(X[:, Ly // 2:]), atol=10))

    def test_2dof_unordered_mesh_SIMP_coating_compliance(self):
        """
        Test the coating formulation for an unstructured mesh
        """
        mesh = FEMOL.mesh.circle_Q4(1, 11)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')
        material = FEMOL.materials.IsotropicMaterial(E=3e6, mu=0.3, rho=5000)
        problem.define_materials(material, material)
        problem.define_tensors(0.1, 0.1)
        problem.add_fixed_domain(FEMOL.domains.outside_circle(0, 0, 0.99))
        problem.add_forces([1e5, 0], FEMOL.domains.inside_circle(0, 0, 0.25))
        SIMP = FEMOL.SIMP_COMP(problem, volfrac=0.2)
        mesh = SIMP.solve(max_iter=3, plot=False, save=False)
        self.assertTrue(np.isclose(np.sum(mesh.cell_data['X']['quad']), mesh.N_ele * 0.2))

    def test_2dof_unordered_mesh_SIMP_coating_compliance_2(self):
        """
        Test the 2 dof SIMP Optimization with a mesh with triangles
        """
        mesh = FEMOL.mesh.rectangle_T3(1, 1, 10, 10)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plane')
        material = FEMOL.materials.IsotropicMaterial(E=3e6, mu=0.3, rho=5000)
        problem.define_materials(material, material)
        problem.define_tensors(0.1, 0.1)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0], [[0, 1]]), [0, 1])
        problem.add_forces([0, -1e5], FEMOL.domains.inside_box([1], [0]))
        SIMP = FEMOL.SIMP_COMP(problem, volfrac=0.4)
        mesh = SIMP.solve(max_iter=3, plot=False, save=False)
        c1 = np.abs(np.sum(mesh.point_data['d1_Uy']))
        c2 = np.abs(np.sum(mesh.point_data['d3_Uy']))
        self.assertTrue(c2 < c1)

    def test_6dof_ordered_mesh_SIMP_compliance(self):
        """
        Ordered mesh with 6dof compliance minimization test
        """
        mesh = FEMOL.mesh.rectangle_Q4(1, 1, 10, 10)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
        material = FEMOL.materials.IsotropicMaterial(E=3e6, mu=0.3, rho=5000)
        problem.define_materials(material)
        problem.define_tensors(0.1)
        problem.add_fixed_domain(FEMOL.domains.outside_box(0.01, 0.99, 0.01, 0.99))
        problem.add_forces([0, 0, -1e5, 0, 0, 0], FEMOL.domains.inside_circle(0.5, 0.5, 0.11))
        SIMP = FEMOL.SIMP_COMP(problem, volfrac=0.4)
        mesh = SIMP.solve(max_iter=3, plot=False, save=False)
        c1 = np.abs(np.sum(mesh.point_data['d1_Uz']))
        c2 = np.abs(np.sum(mesh.point_data['d3_Uz']))
        self.assertTrue(c2 < c1)

    def test_6dof_unordered_mesh_SIMP_compliance(self):
        """
        Test for the compliance minimization of a 6dof unordered mesh
        """
        mesh = FEMOL.mesh.circle_Q4(1, 10)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
        material = FEMOL.materials.IsotropicMaterial(E=3e6, mu=0.3, rho=5000)
        problem.define_materials(material)
        problem.define_tensors(0.1)
        problem.add_fixed_domain(FEMOL.domains.outside_circle(0, 0, 0.9))
        problem.add_forces([0, 0, -1e5, 0, 0, 0], FEMOL.domains.inside_circle(0, 0, 0.25))
        SIMP = FEMOL.SIMP_COMP(problem, volfrac=0.4)
        mesh = SIMP.solve(max_iter=3, plot=False, save=False)
        c1 = np.abs(np.sum(mesh.point_data['d1_Uz']))
        c2 = np.abs(np.sum(mesh.point_data['d3_Uz']))
        self.assertTrue(c2 < c1)

    def test_6dof_ordered_mesh_SIMP_coating_compliance(self):
        """
        Test for the compliance minimization of an ordered 6dof mesh
        with the coating formulation
        """
        mesh = FEMOL.mesh.rectangle_Q4(1, 1, 10, 10)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
        flax = FEMOL.materials.general_flax()
        base_plies = [0, 90, 45, -45]
        coat_plies = [45, -45]
        base_layup = FEMOL.laminate.Layup(material=flax, plies=base_plies, symetric=True, h_core=1)
        coat_layup = FEMOL.laminate.Layup(material=flax, plies=coat_plies, h_core=2)
        problem.define_materials(flax, flax)
        problem.define_tensors(base_layup, coat_layup)
        problem.add_fixed_domain(FEMOL.domains.inside_box([0, 1], [[0, 1]]))
        problem.add_forces([0, 0, -1e6, 0, 0, 0], FEMOL.domains.inside_box([0.5], [[0, 1]]))
        SIMP = FEMOL.SIMP_COMP(problem, volfrac=0.4)
        mesh = SIMP.solve(max_iter=3, plot=False, save=False)
        c1 = np.abs(np.sum(mesh.point_data['d1_Uz']))
        c2 = np.abs(np.sum(mesh.point_data['d3_Uz']))
        self.assertTrue(c2 < c1)

    def test_6dof_unordered_mesh_SIMP_coating_compliance(self):
        """
        Test for the SIMP Compliance minimization for a 6 dof unordered mesh
        """
        mesh = FEMOL.mesh.circle_Q4(1, 10)
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='displacement', model='plate')
        material1 = FEMOL.materials.IsotropicMaterial(E=3e6, mu=0.3, rho=5000)
        material2 = FEMOL.materials.general_flax()
        layup_coat = FEMOL.laminate.Layup(material=material2, plies=[0, 0, 0, 0])
        problem.define_materials(material1, material2)
        problem.define_tensors(0.1, layup_coat)
        problem.add_fixed_domain(FEMOL.domains.outside_circle(0, 0, 0.9))
        problem.add_forces([0, 0, -1e5, 0, 0, 0], FEMOL.domains.inside_circle(0, 0, 0.25))
        SIMP = FEMOL.SIMP_COMP(problem, volfrac=0.4)
        mesh = SIMP.solve(max_iter=3, plot=False, save=False)
        c1 = np.abs(np.sum(mesh.point_data['d1_Uz']))
        c2 = np.abs(np.sum(mesh.point_data['d3_Uz']))
        self.assertTrue(c2 < c1)

if __name__ == '__main__':
    unittest.main()
