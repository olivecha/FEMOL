import FEMOL

# Mesh
nelx = 20
nely = 20
Lx = nelx
Ly = nely
mesh = FEMOL.mesh.rectangle_Q4(nelx, nely, Lx, Ly)

# FEM Problem
plate_FEM = FEMOL.FEM_Problem('displacement', 'plate', mesh)

# Define the composite material layups
flax_base = FEMOL.materials.general_flax()
layup_base = FEMOL.laminate.Layup(flax_base, plies=[0, 90, 0, 90])
carbon_coating = FEMOL.materials.general_carbon()
layup_coating = FEMOL.laminate.Layup(carbon_coating, plies=[0, -45, 90, 45], h_core=10)

plate_FEM.define_materials(flax_base, carbon_coating)
plate_FEM.define_tensors(layup_base, layup_coating)

# Define the boundary conditions
fixed_domain = FEMOL.domains.inside_box([Lx], [[0, Ly]])
plate_FEM.add_fixed_domain(fixed_domain, ddls=[2])

fixed_domain = FEMOL.domains.inside_box([[0, Lx]], [0])
plate_FEM.add_fixed_domain(fixed_domain, ddls=[2])

fixed_domain = FEMOL.domains.inside_box([[0, Lx]], [Ly])
plate_FEM.add_fixed_domain(fixed_domain, ddls = [1])

fixed_domain = FEMOL.domains.inside_box([ 0], [[0, Ly]])
plate_FEM.add_fixed_domain(fixed_domain, ddls = [0])

# Define the applied force
force = [0, 0, -500, 0, 0, 0]
force_domain = FEMOL.domains.inside_box([[0, Lx/10]], [[9*Ly/10, Ly]])
plate_FEM.add_forces(force, force_domain)

topo_problem = FEMOL.TOPOPT_Problem(plate_FEM, volfrac=0.4, penal=3, method='SIMP')
mesh = topo_problem.solve(plot=True)

mesh.plot.cell_data('X')
