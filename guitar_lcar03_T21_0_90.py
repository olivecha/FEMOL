from FEMOL.problems import GuitarSimpVibe 
problem = GuitarSimpVibe(mode='T21', mesh_lcar=0.03, volfrac=0.27, p=1, plies_carbon=[0, 90]) 
problem.solve(max_iter=20, converge=0.03, mesh_filename ="TOM_lcar03_T21_0_90", eigvals_filename="eigvals_lcar03_T21_0_90") 
