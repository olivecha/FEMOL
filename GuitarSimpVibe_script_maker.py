import os
import FEMOL.utils

# current variables
LCAR = 0.05
MAXITER = 20
CONVERGENCE = 0.03
VOLFRAC = 0.27
PENALITY = 1
PLIES_CARBON = [0, 90]


def make_GuitarSimpVibe_script(mode='', lcar=LCAR, max_iter=MAXITER, conv=CONVERGENCE, volfrac=VOLFRAC, p=PENALITY,
                               plies_carbon=PLIES_CARBON):
    """function making a python script for a GuitarSimpVibe problem"""
    filename = f'guitar_lcar{str(lcar)[-2:]}_{mode}_{plies_carbon[0]}_{plies_carbon[1]}.py'
    meshfile = f'TOM_lcar{str(lcar)[-2:]}_{mode}_{plies_carbon[0]}_{plies_carbon[1]}'
    eigvalfile = f'eigvals_lcar{str(lcar)[-2:]}_{mode}_{plies_carbon[0]}_{plies_carbon[1]}'
    f = open(filename, 'w')
    f.write('from FEMOL.problems import GuitarSimpVibe \n')
    f.write(
        f"problem = GuitarSimpVibe(mode='{mode}', mesh_lcar={lcar}, volfrac={volfrac}, p={p}, plies_carbon={plies_carbon}) \n")
    f.write(
        f'problem.solve(max_iter={max_iter}, converge={conv}, mesh_filename ="{meshfile}", eigvals_filename="{eigvalfile}") \n')
    f.close()
    logfile = make_GuitarSimpVibe_log(mode=mode, lcar=lcar, plies_carbon=plies_carbon)
    return filename, logfile


def make_GuitarSimpVibe_log(mode='', lcar=LCAR, plies_carbon=None):
    """function making a log file for a GuitSimpVibe problem"""
    now = FEMOL.utils.unique_time_string()
    filename = f'guitar_lcar{str(lcar)[-2:]}_{mode}_{plies_carbon[0]}_{plies_carbon[1]}.txt'
    f = open(filename, 'w')
    f.write(f'______ Run parameters ______\n')
    f.write(f'Mesh characteristic length : {LCAR} \n')
    f.write(f'Max number of iterations : {MAXITER} \n')
    f.write(f'Convergence criterion : {CONVERGENCE * 100} % \n')
    f.write(f'Target volume fraction : {VOLFRAC * 100} % \n')
    f.write(f'Carbon layup plies : {plies_carbon} % \n')
    f.close()
    return filename


modes = ['T11', 'T21', 'T31']
for mode in modes:
    for plies in [[0, 90], [90, 90], [45, -45]]:
        script, log = make_GuitarSimpVibe_script(mode=mode, plies_carbon=plies)
