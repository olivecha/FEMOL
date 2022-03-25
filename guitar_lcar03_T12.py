from FEMOL.problems import GuitarSimpVibe 
problem = GuitarSimpVibe(mode='T12', mesh_lcar=0.03, volfrac=0.27, p=1) 
problem.solve(max_iter=20, converge=0.02) 
