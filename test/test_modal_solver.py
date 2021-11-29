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
        mesh = FEMOL.mesh.rectangle_Q4(1, 1, 15, 15)
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
        N_ele = 25
        mesh = FEMOL.mesh.circle_Q4(R, N_ele)

        # Problem definition
        thickness = 0.005
        aluminium = FEMOL.materials.IsotropicMaterial(71e9, 0.33, 2700)

        # Create a FEM Problem from the mesh (compute displacement with a plate bending model)
        problem1 = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')
        problem2 = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plane')

        problem1.define_materials(aluminium, aluminium)
        problem1.define_tensors(thickness, thickness)

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
        self.assertFalse(np.allclose(problem1.M.toarray(), problem2.M.toarray()))

if __name__ == '__main__':
    unittest.main()
