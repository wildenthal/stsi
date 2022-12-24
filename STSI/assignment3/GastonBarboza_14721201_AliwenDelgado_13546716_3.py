import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi
import warnings

def main():
    # parameters
    N = 11
    iterations = 100
    T = np.e**2
    r = 0.01
    history = np.empty((iterations,N,2))
    
    
    # set initial positions of particles as circle of radius r
    pos = np.array([[r*np.cos(2*k*pi/N),r*np.sin(2*k*pi/N)] for k in range(N)])
    history[0] = pos
    
    # sample new positions
    run = 0
    while T > 1e-9:   
        for i in range(1, iterations):
            newstep = proposeForcedStep(pos)
            pos = decide(pos, newstep, T)
            history[i] = pos
        print('\r Run {}, temperature {:.2e}'.format(run,T), end="")
        run += 1
        T *= 0.9
        
    # plot movement
    #makePlots(history)
    movie = AnimatedScatter(history)
    movie.save()

def test():
    r = 1
    N = 11
    M = 10
    pos1 = np.array([[r*np.cos(2*k*pi/N),r*np.sin(2*k*pi/N)] for k in range(N)])
    pos2 = np.array([[r*np.cos(2*k*pi/M),r*np.sin(2*k*pi/M)] for k in range(M)])
    pos2 = np.append(pos2, np.array([[0,0]]),axis=0)
    history=[pos1,pos2]
    makePlots(history)
    print('The energy for the symmetric arrangement is {}'.format(energy(pos1)))
    print('The energy for the asymmetric arrangement is {}'.format(energy(pos2)))

def energy(pos):
    '''
        Takes in array of particle positions of size (N,2), calculates 
        pairwise energy as sum_ij=1^N (1/|r_ij|) and returns it.
        Suppresses divide by zero warning when calculating reciprocal
        (infinite autointeraction energy is later discarded).
    
    '''
    warnings.simplefilter("ignore")
    diff = pos[:,np.newaxis]-pos
    distances = np.linalg.norm(diff, axis = 2)
    energies = (distances**-1)[~np.tri(len(distances),k=0,dtype=bool)]
    return np.sum(energies)

def forces(pos):
    '''
        Takes in array of particle positions of size (N,2), calculates 
        force on particle i as sum_j=1^N (r_ij/|r_ij|**3) and returns N forces.
        Suppresses divide by zero warning when calculating reciprocal
        (infinite self-force is later discarded).
    
    '''
    warnings.simplefilter("ignore")
    diff = pos[:,np.newaxis]-pos
    distances = np.linalg.norm(diff, axis = 2)
    forces = (diff/distances[:,:,np.newaxis]**3)
    forces = np.nan_to_num(forces, posinf = 0.0, neginf = 0.0)
    return np.sum(forces,axis=1)

def proposeForcedStep(pos, stepsize = 0.05):
    '''
        Takes in an array of particle positions of size (N,2), generates 
        displacements according to parameter stepsize in the direction given
        by the force on the particle, another stochastic step, and projects 
        the new positions back into the circle if they lie outside, 
        and returns the new array of positions.
    
    '''
    weight = 1
    forceArray = forces(pos)
    forceArray /= np.linalg.norm(forceArray,axis=1)[:,np.newaxis]
    for ip in range(len(pos)):
        x = stepsize*forceArray[ip][0]*weight
        x += stepsize*np.random.uniform(low=-1.0,high=1.0)*(1-weight)
        y = stepsize*forceArray[ip][1]*weight
        y += stepsize*np.random.uniform(low=-1.0,high=1.0)*(1-weight)
        new = pos[ip] + [x,y]
        newnorm = np.linalg.norm(new)
        if newnorm > 1:
            new /= newnorm
        pos[ip] = new
    return pos

def proposeRandomStep(pos, stepsize = 0.05):
    '''
        Takes in an array of particle positions of size (N,2), generates 
        random displacements according to parameter stepsize, projects the new
        positions back into the circle if they lie outside, and returns the new
        array of positions.
    
    '''
    for ip in range(len(pos)):
        x = stepsize*np.random.uniform(low=-1.0,high=1.0)
        y = stepsize*np.random.uniform(low=-1.0,high=1.0)
        new = pos[ip] + [x,y]
        newnorm = np.linalg.norm(new)
        if newnorm > 1:
            new /= newnorm
        pos[ip] = new
    return pos

def decide(pos, newstep, T):
    '''
        Takes in an array of particle positions, a proposed new array, and a 
        temperature T, generates random float, checks if it below the
        threshhold corresponding to the temperature and position energies,
        and decides whether to step the array forward or stay still, and
        returns the updated array.
    
    '''
    
    alpha = np.min([np.exp(-energy(newstep)/T)/np.exp(-energy(pos)/T),1])
    u = np.random.uniform()
    pos = newstep if u <= alpha else pos
    return pos

def makePlots(history):
    fig = plt.figure()
    ax = fig.add_subplot()
    initial = history[0]
    final = history[-1]
    
    #for p in initial:
    #    plt.scatter(*p,c='b')
    for i,p in enumerate(final):
        plt.scatter(*p,c='r')
        plt.annotate('{}'.format(i+1), p)
    ax.set_aspect('equal', adjustable='box')
    arena = plt.Circle((0, 0), 1, color='k',alpha = 0.1)
    ax.add_patch(arena)
    plt.show()
    
class AnimatedScatter(object):
    def __init__(self, history):
        self.stream = self.dataStream(history)
        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(
            self.fig, self.update, init_func = self.setup_plot, 
            blit = True, interval = 500)

    
    def dataStream(self, history):
        i = -1
        while True:
            i += 1
            yield history[i]
            
    def setup_plot(self):
        particles = next(self.stream)
        self.ax.set_title('Frame 0')
        for number,charge in enumerate(particles):
            self.scat = self.ax.scatter(*charge)
            self.ax.annotate('{}'.format(number+1), charge)
        self.ax.axis([-1.1,1.1,-1.1,1.1])
        return self.scat,
        
    
    def update(self, frame):
        particles = next(self.stream)
        self.ax.cla()
        for number,charge in enumerate(particles):
            self.scat = self.ax.scatter(*charge)
            self.ax.annotate('{}'.format(number+1), charge)
        self.ax.set_title('Frame {}'.format(frame))
        self.ax.axis([-1.1,1.1,-1.1,1.1])
        return self.scat,
    
    def save(self):
        self.ani.save('animation.gif')
    
if __name__ == '__main__':
    main()
    #test()