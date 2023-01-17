'''
    Module that defines a generator for arriving customers and a generator
    for servers. Note that it writes queueTimes to a global variable, so when
    importing, an empty list generators.queueTimes must be initialized.

'''

import random
import simpy



def arrival(env, lambd, mu, server, sampling, priority):
    '''
        Creates an instance of an activity to be processed and passes it to 
        the simpy environment, then samples a Poisson time, times out, and
        generates a new activity.
    
    '''
    arrivalId = -1
    
    while True:
        
        arrivalId += 1
        act = activity(env, mu, server, arrivalId, sampling, priority)
        env.process(act)
        t = random.expovariate(lambd)
        yield env.timeout(t)
        

def activity(env, mu, server, arrivalId, sampling, priority):
    '''
        An activity requests a server. When a server appears, the patient
        is considered processed. The time spent waiting is logged to a global
        variable queueTimes which must be created after importing 
        generators.py. A Poisson time is sampled, and the server is held busy
        until this time elapses.
    
    '''
    global queueTimes
    entered = env.now
    
    sampleTime = getTime(sampling, mu)
    
    translatePrio = {"FIFO":None, "SJF": sampleTime}
    priority = translatePrio[priority]
    #print("Patient {} entered queue at {:.2f}".format(arrivalId,entered))
    
    with server.request(priority = priority) as req:
        
        # freeze until request is met
        yield req
        
        # log time waited
        served = env.now
        diff = served-entered
        queueTimes.append(diff)
        message = "Patient {} waited until {:.2f}, wait time {:.2f} minutes"
        #print(message.format(arrivalId,served,diff))
        
        # sample time with nurse
        
        
        yield env.timeout(sampleTime)
        
def getTime(sampling, mu):
    if sampling == "M":
        return random.expovariate(mu)
    if sampling == "D":
        return 1/mu
    if sampling == "LT":
        prob = random.random()
        time = random.expovariate(mu/5) if prob < 0.25 else random.expovariate(mu)
        return time