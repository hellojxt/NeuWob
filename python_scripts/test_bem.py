import sys
import bempp.api
import torch
import numpy as np
import time
import meshio
bempp.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "gpu"
bempp.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "gpu"


# mesh = meshio.read("data/mesh/cube.obj")
# vertices = mesh.points
# elements = mesh.cells_dict["triangle"]
# print("vertices: ", vertices.shape)
# print("elements: ", elements.shape)
# grid = bempp.api.Grid(vertices.T.astype(np.float64), 
#                         elements.T.astype(np.uint32))

grid = bempp.api.shapes.cube(length = 1, origin=(-0.5, -0.5, -0.5), h=0.05)

space = bempp.api.function_space(grid, "DP", 0)
wave_number = 10
slp = bempp.api.operators.boundary.helmholtz.single_layer(
    space, space, space, wave_number, device_interface="opencl", precision="single")
dlp = bempp.api.operators.boundary.helmholtz.double_layer(
    space, space, space, wave_number, device_interface="opencl", precision="single")
identity = bempp.api.operators.boundary.sparse.identity(
    space, space, space, device_interface="opencl", precision="single")

vertices = np.array(grid.vertices).T
elements = np.array(grid.elements).T

meshio.write("data/mesh/cube.obj", meshio.Mesh(vertices, {"triangle": elements}))

dirichlet_coeffs = np.zeros(len(elements), dtype=np.complex64)
for i in range(len(elements)):
    center = np.mean(vertices[elements[i]], axis=0)
    r_norm = np.linalg.norm(center)
    dirichlet_coeffs[i] = np.exp(1j * wave_number * r_norm) / (4 * np.pi * r_norm)

dirichlet_fun = bempp.api.GridFunction(space, coefficients=dirichlet_coeffs)

neumann_coeffs = np.zeros(len(elements), dtype=np.complex64)
for i in range(len(elements)):
    v1 = vertices[elements[i][0]]
    v2 = vertices[elements[i][1]]
    v3 = vertices[elements[i][2]]
    e1 = v2 - v1
    e2 = v3 - v1
    r = (v1 + v2 + v3) / 3 
    r_norm = np.linalg.norm(r)
    normal = np.cross(e1, e2)
    normal = normal / np.linalg.norm(normal)
    # if i == 1:
    #     print("r: ", r)
    #     print("r_norm: ", r_norm)
    #     print("normal: ", normal)
    #     print("np.exp(1j * wave_number * r_norm): ", np.exp(1j * wave_number * r_norm))
    #     print("np.dot(r, normal): ", np.dot(r, normal))
        
    neumann_coeffs[i] = -(1 - 1j * wave_number * r_norm) * np.exp(1j * wave_number * r_norm) / (4 * np.pi * r_norm**3) * np.dot(r, normal)

neumann_fun = bempp.api.GridFunction(space, coefficients=neumann_coeffs)


# check the correctness of the operators
print(vertices[elements[0, 0]], vertices[elements[0, 1]], vertices[elements[0, 2]])

right_arr = np.array((dlp * dirichlet_fun - slp * neumann_fun).coefficients)
left_arr = np.array(0.5*(identity * dirichlet_fun).coefficients)

dlp_mat = dlp.weak_form().A
slp_mat = slp.weak_form().A
print(right_arr[:10])
print(left_arr[:10])

mid_arr = dlp_mat[0,:] * dirichlet_coeffs - slp_mat[0,:] * neumann_coeffs
print(mid_arr.sum())
# mid_arr = mid_arr.real
# mid_arr_ = np.loadtxt("build/real.txt")

# error = mid_arr - mid_arr_
# mask = np.abs(error) > 1e-8
# print("error: ", error[mask])

# print(np.arange(len(error))[mask])



# check_idx = 2650
# print("check_idx: ", check_idx)
# print(mid_arr[check_idx])
# print(vertices[elements[check_idx, 0]], vertices[elements[check_idx, 1]], vertices[elements[check_idx, 2]])
# print(dlp_mat[0, check_idx], dirichlet_coeffs[check_idx], slp_mat[0, check_idx], neumann_coeffs[check_idx])
# print(dlp_mat[0, check_idx] * dirichlet_coeffs[check_idx])
# print(slp_mat[0, check_idx] * neumann_coeffs[check_idx])