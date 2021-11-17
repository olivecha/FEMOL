import FEMOL

# Mesh
nelx = 30
nely = 15
Lx = nelx
Ly = nely
mesh = FEMOL.mesh.rectangle_Q4(nelx, nely, Lx, Ly)

# FEM Problem
plate_FEM = FEMOL.FEM_Problem('displacement', 'plane', mesh)

# Define the composite material layups
material = FEMOL.materials.general_isotropic()

plate_FEM.define_materials(material)
plate_FEM.define_tensors(1)

# Define the boundary conditions
fixed_domain = FEMOL.domains.inside_box([ 0], [[0, Ly/4]])
plate_FEM.add_fixed_domain(fixed_domain, ddls = [0,1])
fixed_domain = FEMOL.domains.inside_box([ 0], [[3*Ly/4, Ly]])
plate_FEM.add_fixed_domain(fixed_domain, ddls = [0,1])

# Define the applied force
force = [0, -0.1]
force_domain = FEMOL.domains.inside_box([Lx], [[Ly/2-1, Ly/2+1]])
plate_FEM.add_forces(force, force_domain)

plate_FEM.plot()

mesh = plate_FEM.solve()

mesh.plot.point_data('Uy')

topo_problem = FEMOL.TOPOPT_Problem(plate_FEM, volfrac=0.4, penal=3, method='SIMP')
mesh = topo_problem.solve(converge=0.03, max_loops=20, plot=False, save=False)

mesh.plot.cell_data('X')
