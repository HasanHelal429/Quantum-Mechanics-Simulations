# Import the nessecary libraries in order to run the program
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import Point as pt
import random as rand
from SpacialHashing import *

import matplotlib.animation as animation
from matplotlib.patches import Rectangle

start_time = time.time()

# This function uses linear interpolation to find the y value  between two points given the x value
def linInterpColl(z, x, y, vx, vy, j):
    f = y[j] + (z - x[j]) * ((y[j + 1] - y[j]) / (x[j + 1] - x[j]))
    return f

# This function calculates the Distance between any two points
def distanceBetwPts(point1, point2, j1):
    return np.sqrt((point1.x[j1] - point2.x[j1])**2 + (point1.y[j1] - point2.y[j1])**2)

# This function calculates the speed of any point
def speed(point1, i, j):
    return np.sqrt(point1[j].vx[i]**2 + point1[j].vy[i]**2)

def wallColl(point):
    return 0

def particleColl(pt, i, cellS):
    origCellX, origCellY = GetCell(pt[i], cellS)

    for i in range(9):
        hash = HashCell(origCellX + offsets[i][0], origCellY + offsets[i][1])
        key = KeyFromHash(hash, numP)
        #currIndex = 


# Constants:
collDamp = 0.90

timeRange = 15
dt = 0.05
tIteration = int(round(timeRange / dt))

#plt.style.use('dark_background')

fig, ax = plt.subplots(1,2)
t = np.linspace(0, timeRange, tIteration)

# Mass and Radius of the particles
mass1 = 10
r1 = 0.1

# Initial position and velocity values 
x1 = 0.01
y1 = 5

v1x = 5
v1y = 0

# Number of particles
numP = 1500

# Define the point objects for all particles
p1 = [pt.Point(x1, y1, v1x, v1y, mass1, r1)]
for j in range(numP - 1):
    p1.append(pt.Point(x1, y1, 0, 0, mass1 + 20, r1))

shoot = np.zeros(numP)
collided = np.array(shoot, dtype='bool')
distr = [0]*101
distrs = [np.zeros(100)]
for i in range(numP):
    shoot[i] = int(i*tIteration/numP)
for i in range(101):
    distr[i]=i

# Global acceleration values
a1x = 0
a1y = 0

# This for loop runs over every frame
for i in range(tIteration + 1):
    # This for loop runs over every point
    distrs.append(np.array(distrs[i]))
    coll_Happ = 0
    for j in range(numP):

        # Iterate the position based on velocity
        x1temp = p1[j].x[i] + dt * p1[j].vx[i] + dt**2 * a1x / 2
        y1temp = p1[j].y[i] + dt * p1[j].vy[i] + dt**2 * a1y / 2

        p1[j].x.append(x1temp)
        p1[j].y.append(y1temp)

        # Iterate the velocity based on position
        vx1temp = p1[j].vx[i] + dt * a1x
        vy1temp = p1[j].vy[i] + dt * a1y
        if(i==shoot[j]):
            vx1temp = v1x 
            vy1temp = (rand.gauss()) * 0.5

        p1[j].vx.append(vx1temp)
        p1[j].vy.append(vy1temp)

        
        
        # Collision detection between particle and right wall
        if p1[j].x[i + 1] > 10:
            yy = int((p1[j].y[i] - 3)*25)
            distrs[i + 1][yy] = 0.1 + distrs[i][yy]

            f1 = p1[j].y[i] + (10 - p1[j].x[i]) * (( p1[j].y[i + 1] - p1[j].y[i]) / (p1[j].x[i + 1] - p1[j].x[i]))
            p1[j].y[i + 1] = f1
            p1[j].x[i + 1] = 0

            dtInt = (p1[j].x[i + 1] - p1[j].x[i]) / p1[j].vx[i]

            t = np.insert(t, i + 1, t[i] + dtInt)
            
            p1[j].vx[i + 1] = -(p1[j].vx[i + 1] * (dtInt / dt) + p1[j].vx[i] * (1 - (dtInt / dt)))
            p1[j].vy[i + 1] = p1[j].vy[i]
            
            for b in t:
                if(b > i + 1):
                    t[b] += dtInt

        if ((4.9 < p1[j].x[i]) & (p1[j].x[i] < 5.1) 
            & (((p1[j].y[i] > 5.25) |  (p1[j].y[i] < 4.75)))):
            f1 = p1[j].y[i] + (4.9 - p1[j].x[i]) * (( p1[j].y[i + 1] - p1[j].y[i]) / (p1[j].x[i + 1] - p1[j].x[i]))
            p1[j].y[i + 1] = f1
            p1[j].x[i + 1] = 4.9

            dtInt = (p1[j].x[i + 1] - p1[j].x[i]) / p1[j].vx[i]

            t = np.insert(t, i + 1, t[i] + dtInt)
            
            p1[j].vx[i + 1] = -(p1[j].vx[i + 1] * (dtInt / dt) + p1[j].vx[i] * (1 - (dtInt / dt)))
            p1[j].vy[i + 1] = p1[j].vy[i]
            
            for b in t:
                if(b > i + 1):
                    t[b] += dtInt
        elif ((4.9 < p1[j].x[i]) & (p1[j].x[i] < 5.1) & 
              ~ (((p1[j].y[i] > 5.25) |  (p1[j].y[i] < 4.75))) & ~collided[j]):
            p1[j].vy[i + 1] += (rand.gauss() ) * 0.5
            collided[j] = True

# Define the scatter plot array
scat1 = [ax[0].scatter(p1[0].x[0], p1[0].y[0], c="r", s=5)]
for j in range(numP - 1):
    scat1.append(ax[0].scatter(p1[j + 1].x[0], p1[j + 1].y[0], c='r', s=5 
                            #, label=f'v1 = ({p1[j + 1].vx[0]}, {p1[j + 1].vy[0]}) m/s'
                            ))
       
hist = ax[1].bar(distr[:-1], distrs[0], width=np.diff(distr), color='skyblue', align = 'edge')
figure = [scat1, hist]


# Define the limits of the plot
ax[0].set(xlim=[0, 10], ylim=[0, 10], xlabel='X (m)', ylabel='Y (m)')
ax[1].set(ylim=[0, 10])
time_text = ax[0].text(0.02, 0.95, '', transform=ax[0].transAxes)

# We paint the walls of the double slit with rectangles.
slitcolor = "b" # Color of the rectangles.
slitalpha = 0.4 # Transparency of the rectangles.
wall_bottom = Rectangle((4.9,0), 0.2, 4.75,      color=slitcolor, zorder=50, alpha=slitalpha) # (x0, y0), width, height
wall_top    = Rectangle((4.9,5.25), 0.2, 4.75,      color=slitcolor, zorder=50, alpha=slitalpha)

# We add the rectangular patches to the plot.
ax[0].add_patch(wall_bottom)
ax[0].add_patch(wall_top)

# This function updates the frame to be displayed in the animation
def update(frame):
    time_text.set_text('Time = %0.2f' % t[frame])
    for j in range(numP):
        figure[0][j].set_offsets((p1[j].x[frame],p1[j].y[frame]))
    for i, b in enumerate(figure[1]):
        b.set_height(distrs[frame][i])

    return scat1,

plt.scatter
ani = animation.FuncAnimation(fig=fig, func=update, frames=tIteration, interval=10)

ani.save(filename="/Users/hasan/Python Animations/single_slit_particles.gif", writer="pillow", fps=50)

print("--- %s seconds ---" % (time.time() - start_time))

#plt.show()