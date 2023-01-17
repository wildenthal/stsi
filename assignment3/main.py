import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi
import warnings
from tqdm import tqdm

def main():
    
    N = int(input('Enter number of charges: '))
    T0 = float(input('Enter initial temperature: '))
    T1 = float(input('Enter final temperature: '))
    m = 'Enter cooling method: exponential, logarithmical, linear, quadratic: '
    method = input(m)
    steps, cool = cooling(T0, T1, method)
    
    # set initial positions of particles as circle of radius r
    T = T0
    r = 0.01
    positions = [[r*np.cos(2*k*pi/N),r*np.sin(2*k*pi/N)] for k in range(N)]
    positions = np.array(positions)
    
    #sets up history array
    history = np.empty((steps,N,2), dtype='float64')
    history[0] = positions
    historyT = np.empty((steps))
    historyT[0] = T

    # sample new positions
    for step in tqdm(range(1,steps)):
        stepsize = 10/steps
        positions = takeForcedStep(positions, T, stepsize = stepsize)
        history[step] = positions
        historyT[step] = T
        T = cool(step)
    
    showParticles(history[-1])
    plt.savefig('charges.jpg', dpi=300)
    plotEnergy(historyT, history)
    plt.savefig('energy.jpg', dpi=300)
    
    np.save('{}charges{}{}{}.npy'.format(N,T0,method,T1),history)
    
    if input('Make movie? y/n: ')=='y':
        frames = 50
        slicing = int(steps/frames) # must slice array to get specified frames
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
        
        theta = np.random.vonmises(0,0.6) 
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

def cooling(T0, T1, method):
    if method == "exponential":
        prompt = 'Enter rate for exponential cooling: '
        rate = float(input(prompt))
        steps = int(np.log(T1/T0)/np.log(rate))
        cool = lambda step: T0*rate**step
    if method == "logarithmical":
        prompt = 'Enter cooling speed-up for logarithmical cooling: '
        alpha = float(input(prompt))
        steps = int(1 + np.exp((T0+T1)/(alpha*T1)))
        cool = lambda step: T0/(1+np.log(1+step))
    if method == "linear":
        prompt = 'Enter factor for linear cooling: '
        alpha = float(input(prompt))
        steps = int(T0-T1/(alpha*T1))
        cool = lambda step: T0/(1+alpha*step)
    if method == "quadratic":
        prompt = 'Enter factor for quadratic cooling: '
        alpha = float(input(prompt))
        steps = int(np.sqrt(T0-T1/(alpha*T1)))
        cool = lambda step: T0/(1+alpha*step**2)
    return steps, cool

''' 
######################################
      Below are plotting tools
######################################
'''

def showParticles(positions):
    '''
        Takes in an array of particle positions and plots them
        onto a circular arena.
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot()

    for i,p in enumerate(positions):
        plt.scatter(*p,c='k')
        plt.annotate('{}'.format(i+1), p)
    ax.set_aspect('equal', adjustable='box')
    arena = plt.Circle((0, 0), 1, color='k',alpha = 0.1)
    ax.add_patch(arena)
    
def plotEnergy(historyT, history): 
    '''
        Takes in an array of particle positions and the log10 of their energies
        as a function of the log10 of the temperature at that point 
        in the simulation.
    
    '''
    energies = np.array([np.log10(energy(positions)) for positions in history])
    fig, ax = plt.subplots()
    ax.plot(np.log10(historyT), energies)
    ax.invert_xaxis()
    ax.set_xlabel("log10 of temperature")
    ax.set_ylabel("log10 of energy")

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