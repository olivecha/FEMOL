import FEMOL
import matplotlib.pyplot as plt

def free_BC_Problem(mesh):
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
    material1 = FEMOL.materials.isotropic_bending_benchmark()
    problem.define_materials(material1, material1)
    problem.define_tensors(0.5, 0.5)  # thick=1
    w, v = problem.solve(filtre=1, verbose=False)
    v_ref = v[1]
    return v_ref, problem

def simple_support_BC_Problem(mesh):
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
    material1 = FEMOL.materials.isotropic_bending_benchmark()
    problem.define_materials(material1, material1)
    problem.define_tensors(0.5, 0.5)  # thick=1
    problem.add_fixed_domain(FEMOL.domains.inside_box([0, 10], [[0, 10]]), ddls=[2, 4])
    problem.add_fixed_domain(FEMOL.domains.inside_box([[0, 10]], [0, 10]), ddls=[2, 3])
    w, v = problem.solve(filtre=1, verbose=False)
    v_ref = v [0]
    return v_ref, problem

def clamped_BC_Problem(mesh):
    problem = FEMOL.FEM_Problem(mesh=mesh, physics='modal', model='plate')
    material1 = FEMOL.materials.isotropic_bending_benchmark()
    problem.define_materials(material1, material1)
    problem.define_tensors(0.5, 0.5)  # thick=1
    problem.add_fixed_domain(FEMOL.domains.outside_box(0.1, 9.9, 0.1, 9.9))
    w, v = problem.solve(filtre=1, verbose=False)
    v_ref = v[0]
    return v_ref, problem

boundary = ['Free', 'Simply supported', 'Clamped']
meshes = [FEMOL.mesh.rectangle_Q4(10, 10, 15, 15) for i in range(3)]
problems = [free_BC_Problem, simple_support_BC_Problem, clamped_BC_Problem]
setups = [p(m) for p, m in zip(problems,meshes)]
meshes = [FEMOL.SIMP_VIBE(s[1], objective='max eig').solve(s[0], plot=False, save=False) for s in setups]

fig, axs = plt.subplots(2, 3, figsize=(8,8))
vectors = [s[0] for s in setups]
for mesh, b, v, i  in zip(meshes, boundary, vectors, range(3)):
    mesh.add_mode('m', v, 6)
    plt.sca(axs[0, i])
    mesh.plot.cell_data('X')
    plt.title(b)
    plt.sca(axs[1, i])
    mesh.plot.point_data('m_Uz')
plt.tight_layout()
plt.show()
