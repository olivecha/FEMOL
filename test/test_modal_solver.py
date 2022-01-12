import unittest
import FEMOL
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_eigenvectors_kronecker_delta(self):
        mesh = FEMOL.mesh.rectangle_Q4(20, 20, 10, 10)

        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        material1 = FEMOL.materials.isotropic_bending_benchmark()
        problem.define_materials(material1)
        problem.define_tensors(1)  # thick=1

        problem.add_fixed_domain(FEMOL.domains.inside_box([0, 20], [[0, 20]]), ddls=[2, 4])
        problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 20]], [0, 20]), ddls=[2, 3])

        problem.add_forces(force=[0, 0, -1, 0, 0, 0], domain=FEMOL.domains.inside_box([[9, 11]], [[9, 11]]))

        _, v = problem.solve(filtre=1)

        self.assertTrue(np.allclose([vj.T @ problem.M @ vj for vj in v], 1))

    def test_coating_mass_matrix_structured_assembly(self):
        # a structured mesh
        mesh = FEMOL.mesh.rectangle_Q4(1, 1, 10, 10)
        # Modal analysis problem (6 dof)
        problem1 = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        problem2 = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        # D = 1 isotropic material
        material = FEMOL.materials.isotropic_bending_benchmark()
        # Two layers
        problem1.define_materials(material, material)
        problem1.define_tensors(0.1, 0.1)
        # One layer
        problem2.define_materials(material)
        problem2.define_tensors(0.1)
        # Solve both
        w1, v = problem1.solve(filtre=1)
        w2, v = problem2.solve(filtre=1)
        # Compare eigen frequencies
        self.assertTrue(np.allclose(w1, w2))

    def test_coating_mass_matrix_unstructured_assembly(self):
        # Circle mesh with R = 0.5 and ~25 ** 2 elements
        R = 0.5  # m
        N_ele = 10
        mesh = FEMOL.mesh.circle_Q4(R, N_ele)

        # Problem definition
        thickness = 0.005
        aluminium = FEMOL.materials.IsotropicMaterial(71e9, 0.33, 2700)

        # Create a FEM Problem from the mesh (compute displacement with a plate bending model)
        problem1 = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
        problem2 = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')

        problem1.define_materials(aluminium, aluminium)
        problem1.define_tensors(thickness/2, thickness/2)

        problem2.define_materials(aluminium)
        problem2.define_tensors(thickness)

        circle_domain = FEMOL.domains.outside_circle(0, 0, R - 0.005)

        # Fix all the degrees of freedom
        problem1.add_fixed_domain(circle_domain)
        problem2.add_fixed_domain(circle_domain)

        # Solve the eigenvalue problem and store the frequencies
        w1, _ = problem1.solve(filtre=2)
        w2, _ = problem2.solve(filtre=2)

        self.assertTrue(np.allclose(w1[:10], w2[:10]))
        self.assertTrue(np.allclose(problem1.M.toarray(), problem2.M.toarray()))

    def test_T3_Q4_elements_comparison(self):
        """
        Test the same vibration problem for T3 and Q4 elements
        Compares the obtained eigenvalues
        """
        mesh1 = FEMOL.mesh.rectangle_T3(1, 1, 10, 10)
        mesh2 = FEMOL.mesh.rectangle_Q4(1, 1, 10, 10)

        w = []
        for mesh in [mesh1, mesh2]:
            problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
            problem.define_materials(FEMOL.materials.general_isotropic())
            problem.define_tensors(1)
            wi, v = problem.solve(filtre=0, verbose=False)
            w.append(wi)
        self.assertTrue(np.allclose(w[0][:15] - w[1][:15], 0, atol=1e-1))

    def test_eigenvalue_filter(self):
        """
        Test the eigenvalue filter of the modal solver
        """
        # Problem with questionable stability
        mesh = FEMOL.mesh.rectangle_Q4(1, 1, 15, 15)
        # laminates and materials
        plies1 = [0, 0, 0, 0]
        plies2 = [90, 90]
        flax = FEMOL.materials.general_flax()
        carbon = FEMOL.materials.general_carbon()
        layup1 = FEMOL.laminate.Layup(material=flax, plies=plies1, symetric=True)
        layup2 = FEMOL.laminate.Layup(material=carbon, plies=plies2, symetric=True, h_core=layup1.h / 2 + 0.010)
        # FEM problem definition
        problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
        problem.define_materials(flax, carbon)
        problem.define_tensors(layup1, layup2)  # thick=1
        # First modal solve
        w, v = problem.solve(filtre=0)
        self.assertFalse(np.isnan(w[0]))

if __name__ == '__main__':
    unittest.main()
