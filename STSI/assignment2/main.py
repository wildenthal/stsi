import simpy
import generators as gen
import numpy as np
import datetime

def main():
    
    for priority in ["FIFO", "SJF"]:
        print("Processing MMn with priority {}".format(priority))
        MGn("M",priority)
        print('\n\n\n')
        
    print("Processing MDn")
    MGn("D", "FIFO")
    print('\n\n\n')
    
    
    print("Processing long tail")
    MGn("LT", "FIFO")
    print('\n\n\n')

def MGn(sampling, priority):
    baseLambda = 1
    rhos = [0.8,0.9,0.95]
    
    serverNumbers = [1,2,4]
    runtime = 25*baseLambda
    e = 0.005
    
    for rho in rhos: 
        for n in serverNumbers:
                        
            env = simpy.Environment()
            start = datetime.datetime.now()
            
            mu = baseLambda/rho
            lambd = n*baseLambda
            
            print("The number of servers is now {}, load is {:.2f}".format(n,lambd/(n*mu)))
            
            server = simpy.PriorityResource(env, capacity = n)
            
            gen.queueTimes = []
            env.process(gen.arrival(env, lambd, mu, server, sampling, priority))
            env.run(until=runtime)
            mean = np.mean(gen.queueTimes)
            
            print("Will process approximately {} arrivals".format(len(gen.queueTimes)))
            
            means = np.array([mean])
            
            std2 = 0
            j = 0
            
            std = 1
            
            while j < 99 or std > e:
                
                j+= 1
                
                elapsed = datetime.datetime.now()-start
                
                print('\r Processed {} out of at least 100, std is {:.3f}, elapsed time is {}'.format(j+1,std,elapsed),end=' ')
                
                env = simpy.Environment()
                
                server = simpy.PriorityResource(env, capacity = n)
                gen.queueTimes = []
                env.process(gen.arrival(env, lambd, mu, server, sampling, priority))
                env.run(until=runtime)
                times = np.mean(gen.queueTimes)
                
                newMean = mean + (times - mean)/(j+1)
                std2 = (1-1/j)*std2 + (j+1)*(newMean - mean)**2
                mean = newMean
                std = np.sqrt(std2/j)
                
                means = np.append(means, mean)
            
            print("\n The mean is {:.3f} +- {:.3f}\n".format(mean,std*1.96))
            if runtime != 1:
                np.save("data/{}p{}s{}std{:.2f}rho{:.2f}mu{}rt{}smpl.npy".format(priority,n,e,rho,mu,runtime,sampling),means)

# set up environment and resources


# set up parameter values


# list to store queueing times


# start arrivals generator




if __name__ == "__main__":
    main()