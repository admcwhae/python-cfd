import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm
import time 

starttime = time.time()

centred = 0

# number of cells 
nx = 25 
ny = 25

# reynolds number 
re = 400.0

# domain size 
xlen = 30.0 
ylen = 2.0

# relaxiation parameters 
relax1 = 1.0 
relax2 = 1.5

# convergence criteria
eps_omega = 1.0e-6
eps_psi = 1.0e-6

# find x and y cell size 
dx = xlen/nx 
dy = ylen/ny 

# viscosity 
visc = ylen/re

# simplification parameters
anu = 2.0 / dx / dx + 2.0/dy/dy
rex = visc * anu

# initalising matrices
psi = np.zeros((nx + 1, ny + 1))
w = np.zeros((nx + 1, ny + 1))
x = np.zeros((nx + 1, ny + 1))
y = np.zeros((nx + 1, ny + 1))
u = np.zeros((nx + 1, ny + 1))
v = np.zeros((nx + 1, ny + 1))
c = 0

# SET UP INITIAL CONDITIONS
for i in range(0, nx + 1):
    for j in range(0, ny + 1):
        x[i][j] = i * dx
        y[i][j] = j * dy
        if(i == 0):
            # set streamfunction and u velocity at inlet
            if(j > ny / 2):
                yy = y[i][j]
                psi[i][j] = -2 * yy**3 + 9 * yy**2 - 12 * yy + 5
                u[i][j] = 6 * (2 - yy) * (yy - 1)
        if(i == nx):
            # set streamfunction and u velocity at outlet
            yy = y[i][j]
            psi[i][j] = -0.25 * yy**3 + 0.75 * yy**2
            u[i][j] = -0.75 * yy**2 + 1.5 * yy
    psi[i][ny] = 1.0       

setuptime = time.time() - starttime
print('Setuptime took: {:.3f} microseconds'.format(setuptime * 1000))
# SOLVING LOOP

for k in range(1, 100000):
    adu = 0.0
    adv = 0.0

    psimin = 1e6
    psimax = -1e6

    #loop over all internal points 
    for i in range(1, nx):
        for j in range(1, nx):
            # modify vorticity if at boundary 

            # inlet side 
            if(i == 1):
                if(j <= ny / 2):
                    # vorticity off the boundary set to the following equation
                    w[i - 1][j] = -2.0 * (psi[i][j] - psi[i - 1][j])/dx/dx
                elif(j == (ny / 2 + 1)):
                    # step corner point, velocity gradiant du/dy is NOT defined at this point, use the average of the slopes at this point
                    w[i - 1][j] = -2.0 * (psi[i][j] - psi[i - 1][j])/dx/dx - 3
                else: 
                    # setting the vorticity for the inlet here
                    yy = y[i - 1][j]
                    w[i - 1][j] = -2.0 * (psi[i][j] - psi[i - 1][j])/dx/dx + 12 * yy - 18

            # outlet side 
            if(i == nx - 1):
                # free out flow vorticity assumed constant
                w[i + 1][j] = w[i][j]
            
            # bottom side 
            if(j == 1):
                w[i][j - 1] = -2.0 * (psi[i][j] - psi[i][j - 1])/dy/dy
                
            # top side
            if(j == ny - 1):
                w[i][j + 1] = -2.0 * (psi[i][j] - psi[i][j + 1])/dy/dy

            # gauss-seidel update 

            u[i][j] =  (psi[i ][j + 1] - psi[i][j - 1])/2.0/dy 
            v[i][j] = -1.0 * (psi[i + 1][j] - psi[i - 1][j])/2.0/dx
            
            # central difference equation for first two terms of vorticity equation
            if(centred == 1):
                w2 = u[i][j] * ((w[i + 1][j] - w[i - 1][j]) /2.0/dx) + v[i][j] * ((w[i][j + 1] - w[i][j - 1]) /2.0/dy)

            if(centred <= 1):
                if(u[i][j] >= 0.0):
                    if(v[i][j] >= 0.0):
                        fact = rex + u[i][j]/dx + v[i][j]/dy
                        wij = u[i][j] * w[i - 1][j]/dx + v[i][j] * w[i][j - 1]/dy + ((w[i + 1][j] + w[i - 1][j])/dx/dx + (w[i][j + 1] + w[i][j - 1])/dy/dy) * visc

                        if(centred == 1):
                            w1 = u[i][j] * (w[i][j] - w[i - 1][j])/dx + v[i][j] * (w[i][j] - w[i][j - 1])/dy
                    
                    else:
                        fact = rex + u[i][j]/dx - v[i][j]/dy
                        wij = u[i][j] * w[i - 1][j]/dx - v[i][j] * w[i][j + 1]/dy + ((w[i + 1][j] + w[i - 1][j])/dx/dx + (w[i][j + 1] + w[i][j - 1])/dy/dy) * visc

                        if(centred == 1):
                            w1 = u[i][j] * (w[i][j] - w[i - 1][j])/dx + v[i][j] * (w[i][j + 1] - w[i][j])/dy

                else:
                    if(v[i][j] >= 0.0):
                        fact = rex - u[i][j]/dx + v[i][j]/dy
                        wij = - u[i][j] * w[i + 1][j]/dx + v[i][j] * w[i][j - 1]/dy + ((w[i + 1][j] + w[i - 1][j])/dx/dx + (w[i][j + 1] + w[i][j - 1])/dy/dy) * visc

                        if(centred == 1):
                            w1 = u[i][j] * (w[i + 1][j] - w[i][j])/dx + v[i][j] * (w[i][j] - w[i][j - 1])/dy
                    
                    else:
                        fact = rex - u[i][j]/dx - v[i][j]/dy
                        wij = - u[i][j] * w[i + 1][j]/dx - v[i][j] * w[i][j + 1]/dy + ((w[i + 1][j] + w[i - 1][j])/dx/dx + (w[i][j + 1] + w[i][j - 1])/dy/dy) * visc

                        if(centred == 1):
                            w1 = u[i][j] * (w[i + 1][j] - w[i][j])/dx + v[i][j + 1] * (w[i][j] - w[i][j])/dy
            else: 
                wij = ((dx**2 * dy**2/visc) * (u[i][j] * (w[i + 1][j] - w[i - 1][j])/2/dx + v[i][j] * (w[i][j + 1] - w[i][j - 1])/2/dy) \
                    + dy**2 * (w[i + 1][j] + w[i - 1][j]) + dx**2 * (w[i][j + 1] + w[i][j - 1])) / 2/(dx**2 + dy**2)

            # determine the change from the previous value 
            if(centred == 1 ):
                wij = (wij + w1 - w2)/fact
            elif(centred == 0):
                wij = wij/fact
            else:
                wij = wij

            # difference in vorticity 
            dw = wij - w[i][j]

            # update value using relaxation factor
            w[i][j] = w[i][j] + relax1 * dw

            # updating stream function
            
            psiij = ((psi[i + 1][j] + psi[i - 1][j])/dx/dx + (psi[i][j + 1] + psi[i][j - 1])/dy/dy + w[i][j])/(2/dx**2 + 2/dy**2)
            dpsi = psiij - psi[i][j] 

            psi[i][j] = psi[i][j] + relax2 * dpsi 

            # find max and min stream functions
            if(psi[i][j] > psimax):
                psimax = psi[i][j]
            if(psi[i][j] < psimin):
                psimin = psi[i][j]
            
            # find maximum changes in psi and vort
            ddu = abs(dpsi)
            ddv = abs(dw)

            if(ddu > adu):
                adu = ddu
            if(ddv > adv):
                adv = ddv

    if(k%100 == 0):
        print('k = {} change in psi = {} change in omega = {}'.format(k, adu, adv))

    if(adu < eps_psi and adv < eps_omega):
        break

    # if(k == 1):
    #     break

print('Solution reached after {} iterations'.format(k))

timetaken = time.time() - starttime

print('Timetaken to get solution was {:.3f} seconds.'.format(timetaken))

plot.figure('Streamfunction')
plot.contourf(x, y, psi, 20, cmap=cm.inferno)
plot.colorbar()
plot.xlim(0, 10)
plot.title('Streamfunction')
# plot.contour(x, y, psi, cmap=cm.viridis)
# plot.quiver(x, y, u, v) 
plot.streamplot(x[:, 0], y[0, :], u, v)
plot.xlabel('x')
plot.ylabel('y')
plot.show()

# plot.figure('Streamfunction')
# plot.contourf(x, y, psi)
# plot.title('Streamfunction')
# plot.xlabel('x')
# plot.ylabel('y')
# plot.show()
