import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi
import warnings
from tqdm import tqdm

def main():
    N = 11 # number of charges

    # sets an exponential cooling
    T0    = 1e2  #initial temperature
    T1    = 1e-13 #final temperature
    rate  = 0.99 #cooling rate
    steps = int(np.log(T1/T0)/np.log(rate)) #steps to reach final temoerature
    T = T0
    cool = lambda T: T*rate
    
    
    # set initial positions of particles as circle of radius r
    r = 0.01
    positions = [[r*np.cos(2*k*pi/N),r*np.sin(2*k*pi/N)] for k in range(N)]
    positions = np.array(positions)
    
    #sets up history array
    history = np.empty((steps,N,2), dtype='float64')
    history[0] = positions

    # sample new positions
    T = T0
    for step in tqdm(range(1,steps)):
        stepsize = 10/steps
        positions = takeForcedStep(positions, T, stepsize = stepsize)
        history[step] = positions
        T = cool(T)
    
    showParticles(history[-1])
    
    if False:
        #make movie
        frames = 50
        slicing = int(steps/frames) # array must be sliced to get specified frames
        sliced = history[0::slicing]
        movie = AnimatedScatter(sliced)
        movie.save()

def test():
    '''
        Compares minimum energies for N particles arranged symmetrically
        or with one charge placed in the center.
    '''
    r = 1
    N = 11
    M = N-1
    
    posSym = [[r*np.cos(2*k*pi/N),r*np.sin(2*k*pi/N)] for k in range(N)]
    posAsym = [[r*np.cos(2*k*pi/M),r*np.sin(2*k*pi/M)] for k in range(M)]
    
    posSym = np.array(posSym)
    posAsym = np.array(posAsym)
    posAsym = np.append(posAsym, np.array([[0,0]]),axis=0)
    history=[posSym,posAsym]
    makePlot(history)
    sym = 'The energy for the symmetric arrangement with {} particles is {}'
    print(sym.format(N,energy(posSym)))
    asym = 'The energy for the asymmetric arrangement with {} particles is {}'
    print(asym.format(N,energy(posAsym)))

def energy(positions):
    '''
        Takes in array of particle positions of size (N,2), calculates 
        pairwise energy as sum_ij=1^N (1/|r_ij|) and returns it.
        Suppresses divide by zero warning when calculating reciprocal
        (infinite autointeraction energy is later discarded).
    
    '''
    warnings.simplefilter("ignore")
    diff = positions[:,np.newaxis] - positions
    distances = np.linalg.norm(diff, axis = 2)
    energies = (distances**-1)[~np.tri(len(distances),k=0,dtype=bool)]
    return np.sum(energies)

def forces(positions):
    '''
        Takes in array of particle positions of size (N,2), calculates 
        force on particle i as sum_j=1^N (r_ij/|r_ij|**3) and returns N forces.
        Suppresses divide by zero warning when calculating reciprocal
        (infinite self-force is later discarded).
    
    '''
    warnings.simplefilter("ignore")
    diff = positions[:,np.newaxis] - positions
    distances = np.linalg.norm(diff, axis = 2)
    forces = (diff/distances[:,:,np.newaxis]**3)
    forces = np.nan_to_num(forces, posinf = 0.0, neginf = 0.0)
    return np.sum(forces,axis=1)

def takeForcedStep(positions, T, stepsize = 0.01):
    '''
        Takes in an array of particle positions of size (N,2), and for each 
        particle generates a displacement according to parameter stepsize and
        in the direction given by the force on the particle, but perturbed by
        an angle sampled from the random circular distribution. 
        If the new position lies outside of the circle, it is projected
        back inside.
        The displacement is passed to the decide function, and is accepted
        or rejected based on the temperature.
        The array of new particle positions is then returned.
    
    '''
    forceArray = forces(positions)
    forceArray /= np.linalg.norm(forceArray,axis=1)[:,np.newaxis]
    # loop over particles
    for index in range(len(positions)):
        proposal = positions
        
        theta = np.random.vonmises(0,1/T) 
        force = forceArray[index]*stepsize
        
        x = force[0]*np.cos(theta)  + force[1]*np.sin(theta)
        y = force[0]*np.sin(-theta) + force[1]*np.cos(theta)
        
        proposal[index] += [x,y]
        
        # normalize if it takes charge outside of the circle
        norm = np.linalg.norm(proposal[index])
        if norm > 1: proposal[index] /= norm
        
        #decide whether to step or not based on temperature
        positions = decide(positions, proposal, T)
        
    return positions


def takeRandomStep(positions, T, stepsize = 0.01):
    '''
        Takes in an array of particle positions of size (N,2), and for each
        particle generates a uniformly sampled random displacement according 
        to the parameter stepsize.
        If the new position lies outside of the circle, it is projected
        back inside.
        The displacement is passed to the decide function, and is accepted
        or rejected based on the temperature.
        The array of new particle positions is then returned.
    
    '''
    # loop over particles
    for index in range(len(positions)):
        
        proposal = positions
        # propose a step in a random direction
        theta = np.random.uniform(high = 2*pi)
        x = stepsize*np.sin(theta)
        y = stepsize*np.cos(theta)
        
        proposal[index] += [x,y]
        
        # normalize if it takes charge outside of the circle
        norm = np.linalg.norm(proposal)
        if norm > 1: proposal /= norm
        
        #decide whether to step or not based on temperature
        positions = decide(positions, proposal, T)
        
    return positions

def decide(positions, proposal, T):
    '''
        Takes in an array of particle positions, a proposed new array, and a 
        temperature T.
        Generates random float, checks if it below the
        threshhold corresponding to the temperature and position energies,
        and decides whether to step the array forward or stay still.
        Returns the updated array.
    
    '''
    
    alpha = np.exp(-energy(proposal)/T)/np.exp(-energy(positions)/T)
    alpha = np.min([alpha,1])
    u = np.random.uniform()
    if u<= alpha: positions = proposal
    return positions

def showParticles(positions):
    '''
        Takes in an array of particle positions and plots them
        onto a circular arena.
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot()

    for i,p in enumerate(positions):
        plt.scatter(*p,c='r')
        plt.annotate('{}'.format(i+1), p)
    ax.set_aspect('equal', adjustable='box')
    arena = plt.Circle((0, 0), 1, color='k',alpha = 0.1)
    ax.add_patch(arena)
    plt.show()
    
def plotEnergy(history):
    pass

class AnimatedScatter(object):
    '''
        Is initialized with a vector containing arrays of particle positions.
        Creates an animation displaying the movement of the particles.
    
    '''
    
    def __init__(self, history):
        self.stream = self.dataStream(history)
        self.fig, self.ax = plt.subplots()
        self.frame = 0
        self.length  = len(history)
        self.ani = animation.FuncAnimation(
            self.fig, self.update, init_func = self.setup_plot, 
            blit = True, interval = 10, save_count=self.length-1)

    
    def dataStream(self, history):
        i = 0
        yield history[0]
        yield history[0]
        while i < len(history):
            yield history[i]
            i += 1
            
    def setup_plot(self):
        particles = next(self.stream)
        for number,charge in enumerate(particles):
            self.scat = self.ax.scatter(*charge)
            self.ax.annotate('{}'.format(number+1), charge)
        self.ax.axis([-1.1,1.1,-1.1,1.1])
        self.ax.set_aspect('equal', adjustable='box')
        return self.scat,
        
    
    def update(self, frame):
        self.frame += 1
        particles = next(self.stream)
        print('\r plotting frame {} of {}'.format(self.frame,
                                                  self.length-1), end="")
        self.ax.cla()
        for number,charge in enumerate(particles):
            self.scat = self.ax.scatter(*charge, color='k')
            self.ax.annotate('{}'.format(number+1), charge)
        self.ax.axis([-1.1,1.1,-1.1,1.1])
        self.ax.set_aspect('equal', adjustable='box')
        return self.scat,
    
    def save(self):
        self.ani.save('animation.gif')
    
if __name__ == '__main__':
    main()
    #test()