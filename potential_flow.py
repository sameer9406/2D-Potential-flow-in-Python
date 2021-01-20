import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from time import time

# for sparse matrices
from scipy.sparse import csc_matrix  # for sparse matrix
from scipy.sparse.linalg import spsolve  # for sparse    matrix

from mesh_utils import rectangular_mesh  # import everything from custom module mesh_utils
import matplotlib.cm as cm

#################
# USER SETTINGS #
#################

ly = 31.5 #length of domain in y direction
lx = 46.5 #length of domain in x direction
ny = 57  #number of points in y direction
nx = 49  #number of points in x direction

u_out = 1  # velocity at outflow
u_in = u_out  # velocity at inflow

ix_phi = nx-2  # x-coordinate at which potential is provided
iy_phi = ny-1 # y-coordinate at which potential is provided
phi_default= 10


class potential_mesh(rectangular_mesh):

    def is_inout(self, ik, boundary):
        # this function returns true if boundary point is an inflow point

        inout = ''  # initialisation

        # inflow/outflow position data
        in_bottom = 0
        in_top = 0.1 * self.ly
        out_bottom = 0.9 * self.ly
        out_top = self.ly

        ix, iy = self.get_mat_pos(ik)  # get 2d indices of current point
        y_value = self.y[iy]  # get y position of current point
        x_value = self.x[ix]
       
        # inflow is west, between the given values of y
        if boundary == 'w' and y_value > in_bottom and y_value < in_top: #iy=y_value
            inout = 'in'

        # outflow is east, between the given values of y
        if boundary == 'e' and y_value > out_bottom and y_value < out_top:
            inout = 'out'

        return inout 

    def is_nonflow(self, ik):
        # this function returns true if the point is in the non flow area
        # position of the lower block where the flow is not present
       
        ix, iy = self.get_mat_pos(ik)
        y_value = self.y[iy]
        x_value = self.x[ix]

        # walls of the upper block
        y_top = 0.8 * self.ly 
        y_bot = 0.6 * self.ly 
        x_end = 0.5 * self.lx 
        if y_bot <= y_value <= y_top  and  x_end <= x_value :
            return True
        
        # walls of the lower block
        y_bot = 0.2 * self.ly
        y_top = 0.3 * self.ly
        x_start = 0.5 * self.lx
        if   y_bot <= y_value <= y_top  and x_start >= x_value :
            return True



m = potential_mesh(nx, lx, ny, ly) 

# allocation
A = np.zeros((m.nx * m.ny, m.nx * m.ny))
b = np.zeros(m.nx * m.ny)

for ii in range(m.nx * m.ny):  # cycle over rows of A

    north, south, west, east = m.get_compass(ii)  # get indices of neighbouring points
    boundary, nout, nin, dn = m.is_boundary(ii)  # assess whether you are on a boundary
    nonflow = m.is_nonflow(ii) 
    #nonflow_bound = m.is_nonflow_boundary(ii,A)
    
    
    if not boundary:
        if  not nonflow :

        # write coefficients corresponding to central differences
            A[ii, ii] = - 2 / (m.dx) ** 2 - 2 / (m.dy) ** 2
            A[ii, north] = 1 / (m.dy) ** 2
            A[ii, south] = 1 / (m.dy) ** 2
            A[ii, west] = 1 / (m.dx) ** 2
            A[ii, east] = 1 / (m.dx) ** 2
        else :
            A[ii, ii] = 1

    # known term is left untouched, since it is 0
    else:
        # in any case, A contains outgoing normal derivative
                A[ii, nin] = -1 / dn 
                A[ii, nout] = 1 / dn

    # check if point is inflow/outflow
    in_out_flow = m.is_inout(ii, boundary)
    if in_out_flow == 'in':  # if you are at an inflow

        # nonzero Neumann condition
        b[ii] = - u_in

    elif in_out_flow == 'out':  # if you are at an outflow

        # nonzero Neumann condition
        b[ii] = u_out

    # otherwise, b is left untouched: it is already zero

# get memory index of such point
imem = m.get_mem_pos(ix_phi, iy_phi)

# delete corresponding row of A
A[imem, :] = 0

# add 1 to diagonal
A[imem, imem] = 1

# correct known term
b[imem] = phi_default

A = csc_matrix(A)  # converts A to a sparse matrix
start = time()
# phi = solve(A,b) # for normal solution
phi = spsolve(A, b)  # for sparse matrix
end = time()
print('Matrix inversion took', end - start, 'seconds to complete.')

#removing the values present in the non_flow region
for ii in range(m.nx * m.ny):
     non_flow = m.is_nonflow(ii)

     if non_flow:
         phi[ii] = None

# treatment for mesh before plotting
phi = phi.reshape((m.nx, m.ny))
phi = phi.transpose()

#pcolormesh
fig, ax = plt.subplots()
ax.set_title('Velocity potential and isocontours')
pos = ax.pcolormesh(m.x, m.y, phi, shading='gouraud')
fig.colorbar(pos)
ax.contour(m.x, m.y, phi, 30, colors='white', linewidths=.7)
#
#
# calculate gradient
v, u = np.gradient(phi)  # , m.x, m.y)
u /= m.dx
v /= m.dy

#streamlines
x = ((np.linspace(0, lx, nx)))
y = ((np.linspace(0, ly, ny)))
fig, ax = plt.subplots()
ax.set_title('Streamlines')
ax.streamplot(x, y, u, v)

#vector plot
fig, ax = plt.subplots()
ax.set_title('Velocity field')
ax.quiver(m.x, m.y, u, v)
plt.show()
