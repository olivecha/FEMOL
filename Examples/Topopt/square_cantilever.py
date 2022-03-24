import FEMOL
# Create a square 1x1 mesh
mesh = FEMOL.mesh.rectangle_Q4(Lx=1, Ly=1, nelx=30, nely=30)
# The domain is 1x1x1
thickness = 1. 
# Create an isotropic material with E=100Pa, mu=0.3 and rho=1
material = FEMOL.materials.IsotropicMaterial(E=1e2, mu=0.3, rho=1)
# Create the FEM problem
FEM = FEMOL.FEM_Problem('displacement', 'plane', mesh)
# define the material and thickness
FEM.define_materials(material)
FEM.define_tensors(t)
# Apply a fixed domain on the left side
fixed_domain = FEMOL.domains.inside_box([0], [[0, Ly]]) 
FEM.add_fixed_domain(fixed_domain, ddls = [0,1])
# Apply a downward force on the lower right corner
force_domain = FEMOL.domains.inside_box([Lx], [0])
force = [0, -1] 
FEM.add_forces(force, force_domain)
# Plot the problem
FEM.plot()  
# Solve the problem and plot the displacement
mesh = FEM.solve()
mesh.plot.point_data('Uy')
# Define and solve the TOM problem and measure the time
TOM = FEMOL.SIMP_COMP(plate_FEM, volfrac=0.5, penal=3, rmin=1.5) 
now = time.time()
mesh = TOM.solve(converge=0.03, max_iter=100, plot=True, save=False)
print(time.time() - now, 'solve time (s)')
# plot the 8th iteration
mesh.plot.cell_data('X8')