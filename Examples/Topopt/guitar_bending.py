import FEMOL

# Mesh
nelx = 42 # elements in the x direction
nely = 32 # elements in the y direction
Lx = nelx
Ly = nely
mesh = FEMOL.mesh.rectangle_Q4(Lx, Ly, nelx, nely)

# FEM Problem
problem = FEMOL.FEM_Problem('displacement', 'plate', mesh)

# Define the composite material layups and materials
flax_base = FEMOL.materials.general_flax()
layup_base = FEMOL.laminate.Layup(flax_base, plies=[0, 90, 0, 90])
carbon_coating = FEMOL.materials.general_carbon()
layup_coating = FEMOL.laminate.Layup(carbon_coating, plies=[0, -45, 90, 45], h_core=10)
problem.define_materials(flax_base, carbon_coating)
problem.define_tensors(layup_base, layup_coating)

# Fix the guitar boundary
problem.add_fixed_domain(FEMOL.domains.guitar_domain(Lx=Lx, Ly=Ly), ddls=[0, 1, 2])

# define the string tension
force_domain = FEMOL.domains.inside_box([Lx/3], [[8*Ly/20, 12*Ly/20]])
force = [0, 0, 0, -1e4, 0, 0]
problem.add_forces(force, force_domain)

# Domain between the guitar side curves
def guitar_sides(Lx, Ly):
    angle = np.pi / 6
    p = angle / (np.pi / 2)
    x1 = 2 * Ly / 6 + 2 * Ly / 6 * np.sin(angle)
    y1 = 2 * Ly / 6 - 2 * Ly / 6 * np.cos(angle)
    x2 = Lx - Ly / 4 - Ly / 4 * np.sin(angle)
    y2 = 2 * Ly / 6 - Ly / 4 * np.cos(angle)
    a, b, c, d = FEMOL.domains.create_polynomial(x1, y1, x2, y2, p)

    def sides(x, y):
        Y_val = a * x ** 3 + b * x ** 2 + c * x + d
        return ~((x > x1) & (x < x2) & (y > Y_val) & (y < -Y_val + Ly))

    return sides

# Outside domain
circle1 = FEMOL.domains.outside_circle((2*Ly/6), (2*Ly/6), (2*Ly/6))
circle2 = FEMOL.domains.outside_circle((2*Ly/6), (4*Ly/6), (2*Ly/6))
circle3 = FEMOL.domains.outside_circle((Lx-Ly/4) , 2*Ly/6, Ly/4)
circle4 = FEMOL.domains.outside_circle((Lx-Ly/4) , 4*Ly/6, Ly/4)
box1 = FEMOL.domains.outside_box(0, Lx, 2*Ly/6, 4*Ly/6)
sides = guitar_sides(Lx, Ly)
sound_hole = FEMOL.domains.inside_circle(2*Lx/3, Ly/2, Ly/7)

def voided_guitar(x, y):
    """
    Parts where there is no material on the guitar
    """
    if np.array([circle1(x,y), circle2(x,y), circle3(x,y), circle4(x,y), box1(x,y), sides(x, y)]).all():
        return True
    elif sound_hole(x,y):
        return True
    else:
        return False 
    
topo_problem = FEMOL.TOPOPT_Problem(problem, volfrac=0.2, penal=3)
topo_problem.void_domain = voided_guitar
mesh = topo_problem.solve(plot=True)
