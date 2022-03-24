import os
import FEMOL.utils

# current variables
LCAR = 0.05
MAXITER = 1
CONVERGENCE = 0.02
VOLFRAC = 0.27
SOLVE = True

def make_GuitarSimpVibe_script(mode='', lcar=LCAR, max_iter=MAXITER, conv=CONVERGENCE, volfrac=VOLFRAC):
    """function making a python script for a GuitarSimpVibe problem"""
    filename = f'batch_scripts/guitar_lcar{str(lcar)[-2:]}_{mode}.py'
    f = open(filename, 'w')
    f.write('from FEMOL.problems import GuitarSimpVibe \n')
    f.write(f"problem = GuitarSimpVibe(mode='{mode}', mesh_lcar={lcar}, volfrac={volfrac}) \n")
    f.write(f'problem.solve(max_iter={max_iter}, converge={conv}) \n')
    f.close()
    return filename


def make_GuitarSimpVibe_log(mode='', lcar=LCAR):
    """function making a log file for a GuitSimpVibe problem"""
    now = FEMOL.utils.unique_time_string()
    filename = f'batch_results/guitar_lcar{str(lcar)[-2:]}_{mode}_{now}.txt'
    f = open(filename, 'w')
    f.write(f'______ Run parameters ______\n')
    f.write(f'Mesh characteristic length : {LCAR} \n')
    f.write(f'Max number of iterations : {MAXITER} \n')
    f.write(f'Convergence criterion : {CONVERGENCE*100} % \n')
    f.write(f'Target volume fraction : {VOLFRAC*100} % \n')
    f.close()
    return filename


modes = ['T11', 'T21', 'T12', 'T31']
modes = ['T11']
for mode in modes:
    script = make_GuitarSimpVibe_script(mode=mode)
    log = make_GuitarSimpVibe_log(mode=mode)
    if SOLVE:
        os.system(f'python {script} >> {log}')
