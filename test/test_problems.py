import unittest
import FEMOL
import os
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_GuitarModal(self):
        os.chdir("..")
        problem = FEMOL.problems.GuitarModal(mesh_lcar=0.09, )
        w, v = problem.solve()
        w_ref = np.array([152.59688346, 375.9742514, 239.38978681, 731.83146082])
        self.assertTrue(np.allclose(w, w_ref))
        os.chdir('test')

    def test_GuitarSimpVibe(self):
        os.chdir("..")
        problem = FEMOL.problems.GuitarSimpVibe(mesh_lcar=0.09, mode='T11', volfrac=0.27)
        SIMP = problem.solve(plot=False, save=False, max_iter=1)
        self.assertTrue('X' in SIMP.mesh.cell_data.keys())
        os.chdir('test')


if __name__ == '__main__':
    unittest.main()

