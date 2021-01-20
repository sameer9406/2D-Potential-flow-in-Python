import numpy as np


class rectangular_mesh:

    # class constructor

    def __init__(self, nx, lx, ny, ly):

        self.nx = nx  # save nx into class
        self.lx = lx  # save lx into class
        self.ny = ny  # save ny into class
        self.ly = ly  # save nx into class

        # generate mesh
        self.x = np.rint((np.linspace(0, self.lx, self.nx)))
        self.y = np.rint((np.linspace(0, self.ly, self.ny)))
     
        # calculate mesh resolution
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

    #######################################################################################################

    # first off, let's define a function that converts 2D indices to 1D

    def get_mem_pos(self, ix, iy):
        # notice that input arguments ix, iy follow Python notation - which is, they start from 0

        ny = self.ny

        return ix * ny + iy

    #######################################################################################################

    # let's also define the inverse of previous function

    def get_mat_pos(self, ik):

        ny = self.ny

        ix = ik // ny
        iy = ik % ny

        return ix, iy

    #######################################################################################################

    # finally, we define a function that returns 1D indices of compass points

    def get_compass(self, curr_p, *args):  # args are optional arguments
        # optional arguments are here used so that user can either pass array or matrix notation
        # thus, user can either call this function as:
        #
        #     icomp = get_compass('N', ik),    where ik = get_arr_pos(ix, iy)
        #
        # or
        #
        #     icomp = get_compass('N', ix, iy)

        ny = self.ny

        # first off: if 2 indeces are passed, recover 1 index notation
        if args:  # if args is not empty! notice that if args is equivalent to False if it's empty
            curr_p = self.get_mem_pos(curr_p, args[0])

        # then, compass points are found
        north = curr_p + 1
        south = curr_p - 1
        west = curr_p - ny
        east = curr_p + ny

        return north, south, west, east

    #######################################################################################################

    # then, we define a funciton which tells whether a given 1D index corresponds to a boundary point;
    # it also returns the indices nout and nin needed to calculate the outgoing first order derivative
    # at the given point, as well as the spacing


    def is_boundary(self, ik):

        ny = self.ny
        nx = self.nx
        # print(nx, ny, ik, nx*ny)
        # initialise variables
        boundary = ''
        nout = ik
        nin = None
        dn = None

        # boundary will be empty if the point is not on the boundary;
        # otherwise, it will contain one or more letters specifying on which boundary it is located
        # preliminary operations
        ix, iy = self.get_mat_pos(ik)
        y_value = self.y[iy]
        x_value = self.x[ix]
        north, south, west, east = self.get_compass(ik)

        # assess: west or east
        if ik < ny:
            boundary += 'w'
            nin = east
            dn = self.dx
        elif ik >= ny * (nx - 1):
            boundary += 'e'
            nin = west
            dn = self.dx
        
        # assess: north or south
        if (ik % ny) == 0:
            boundary += 's'
            nin = north
            dn = self.dy
        elif (ik % ny) == (ny - 1):
            boundary += 'n'
            nin = south
            dn = self.dy

        # assess: inner walls of lower block        
        if  y_value == min(self.y, key=lambda x:abs(x-(0.2 * self.ly)))  and x_value <= self.lx / 2:
            boundary += 'r'
            nin = south
            dn = self.dy
        elif y_value == min(self.y, key=lambda x:abs(x-(0.3 * self.ly)))  and x_value <= self.lx / 2:
            boundary += 'r'
            nin = north
            dn = self.dy
        elif x_value == min(self.x, key=lambda x:abs(x-(0.5 * self.lx)))  and  0.2 * self.ly <= y_value <= 0.3 * self.ly :
            boundary += 'r'
            nin = east
            dn = self.dx
       
        # assess: inner walls of upper block 
        if  y_value == min(self.y, key=lambda x:abs(x-(0.8 * self.ly)))  and x_value >= self.lx / 2: 
            boundary += 'r'
            nin = north
            dn = self.dy
        elif  y_value == min(self.y, key=lambda x:abs(x-(0.6 * self.ly)))  and x_value >= self.lx / 2:
            boundary += 'r'
            nin = south
            dn = self.dy
        elif x_value == min(self.x, key=lambda x:abs(x-(0.5 * self.lx))) and  0.6 * self.ly <= y_value <= 0.8 * self.ly :
            boundary += 'r'
            nin = west
            dn = self.dx      
    
        return boundary, nout, nin, dn
