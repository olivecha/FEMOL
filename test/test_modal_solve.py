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

        _, v = problem.solve(filtre=3)

        self.assertTrue(np.allclose([vj.T @ problem.M @ vj for vj in v], 1))


if __name__ == '__main__':
    unittest.main()
