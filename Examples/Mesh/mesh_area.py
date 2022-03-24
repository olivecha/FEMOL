import FEMOL

# area of a guitar mesh
# get a unit guitar mesh
guitar = FEMOL.mesh.guitar()
# scale to the real guitar size
guitar.points *= (18.5*0.0254)
# compute the area
A = FEMOL.utils.points_area(guitar.points)
# Compute a volume for a 9 ply flax laminate
V = A*FEMOL.materials.general_flax().hi*9
# Compute a mass from a density
M = V * FEMOL.materials.general_flax().rho
print(f'mass : {int(1000*M)} g')
print(f'bracing mass : {0.3*int(1000*M)} g')