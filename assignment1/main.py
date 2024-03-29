import numpy as np
from numpy import random
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

''' 
The indicator() function returns 1 if the complex number c is in the 
Mandelbrot set, and otherwise returns 0.
c is inside the Mandelbrot set if, for f(z) = z^2 + c, the sequence 
z_n = f(f(...(f(0)))) remains bounded, where f is applied n times.
c is in the Mandelbrot set if and only if |z_n|^2 <= 4.

We calculate the sequence for a specified amount of iterations. 
If |z_n|^2 <= 4 for those iterations, we conclude that c is in the desired set.
This set is NOT the Mandelbrot set, but it tends to it as the amount of
iterations goes to infinity.

This set is actually larger than the Mandelbrot set, as we are including points
that may actually diverge as n goes to infinity, but grow slow enough that
they do not diverge for n < iterations.
'''

def indicator(c:complex, iterations:int) -> int:
    thresh = 4
    z      = c
    steps  = 0

    while steps < iterations and (z*z.conjugate()).real < thresh:
        z      = z**2 + c
        steps += 1
    
    return steps == iterations

'''
The Monte Carlo method for calculating the area of a set gives us a collection
of random variables depending on the number of point samples used in the method.
As we increase the number of point samples, the mean of the random variable
tends to the area of the set.
The calculateArea() function generates a sample from this distribution for the
set determined by $iterations.
(As commented in calculateArea(), this set is NOT the Mandelbrot set, but it 
tends to it as $iterations goes to infinity).

It takes a parameter $points, which is the number of point samples used to 
calculate the area.
It takes a parameter $iterations, which indicates the maximum amount of steps 
to calculate the Mandelbrot sequence before we conclude the point belongs to 
the set.
It takes a parameter sampling, which indicates how the points should be sampled.
'''


def calculateArea(points:int, iterations:int, sampling):
  xMin, xLen, yMin, yLen = -2, 2.47, -1.12, 2.24
  domainArea = xLen*yLen

  if sampling == 'uniform':
    pointsInside = 0

    for k in range(points):
      x = xMin + random.uniform()*xLen
      y = yMin + random.uniform()*yLen
      c = complex(x,y)
      pointsInside+= indicator(c,iterations)
    mandelArea = domainArea*pointsInside/points
  
  elif sampling == 'latin cube':
    p = random.permutation([i for i in range(points)])
    q = random.permutation(p)
    pointsInside = 0
    for k in range(points):
      x = xMin + xLen * (random.uniform() + p[k])/len(p)
      y = yMin + yLen * (random.uniform() + q[k])/len(q)
      c = complex(x,y)
      pointsInside+= indicator(c,iterations)
    mandelArea = domainArea*pointsInside/points
  
  elif sampling == 'orthogonal':
    gridSquares = 5
    xSquareLen  = xLen    / gridSquares
    ySquareLen  = yLen    / gridSquares
    gridPoints  = points // gridSquares**2
    extraPoints = points  % gridSquares**2
    pointsInside = 0

    for xSquare in range(gridSquares):
      xOffset = xMin + xSquare*xSquareLen

      for ySquare in range(gridSquares):
        yOffset = yMin + ySquare*ySquareLen

        p = random.permutation([i for i in range(gridPoints)])
        q = random.permutation(p)

        for k in range(gridPoints):
          x = xOffset + xSquareLen * (random.uniform() + p[k])/len(p)
          y = yOffset + ySquareLen * (random.uniform() + q[k])/len(q)
          c = complex(x,y)
          pointsInside += indicator(c,iterations)
          
    
    for _ in range(extraPoints):
      x = random.uniform(xMin,xMin+xLen)
      y = random.uniform(yMin,yMin+yLen)
      c = complex(x,y)
      pointsInside += indicator(c,iterations)
    
    mandelArea = domainArea*pointsInside/points

  else:
    raise ValueError("sampling method not defined")
  
  return mandelArea

'''
If we draw N independent samples of the distribution generated from the 
calculateArea() function, the average of their values estimates the area of the
set determined by max_steps (Ross, chapter 8).
The corresponding variance S is s/sqrt(N), where s is the variance of the 
distribution generated through calculateArea().

The estimateArea() function takes as parameters the variables $samples and 
$iterations, which are passed to calculateArea.
It generates area samples until S < 0.01, which ensures that the probability 
that the sample mean differs from the calculateArea() mean by more than 
0.0196 is 5%.
It returns the sample mean and the amount of area samples required to reach 
the specified variance.

As the number of point samples and iterations go to infinity, the 
area sample mean should tend to the Mandelbrot area.
'''

def estimateArea(samples:int, iterations:int, sampling):
  mean = calculateArea(samples,iterations, sampling)
  std2 = 0
  j = 1

  while j < 100 or np.sqrt(std2/j) > 0.005:
    area = calculateArea(samples,iterations,sampling)
    newMean = mean + (area - mean)/(j+1)
    std2 = (1-1/j)*std2 + (j+1)*(newMean - mean)**2
    mean = newMean
    j += 1

  return mean, j

'''

'''
def sweepParameters(samples,iterations, sampling):
  means = np.zeros((len(samples),len(iterations)))
  convergence = np.zeros((len(samples),len(iterations)))

  sIndex = 0
  for s in tqdm(samples):
    iIndex = 0
    for i in tqdm(iterations):
      means[sIndex,iIndex], convergence[sIndex,iIndex] = estimateArea(s,i,sampling)
      iIndex += 1
    sIndex += 1
  return means, convergence

# minimum and maximum amount of samples and the step increment
sMin, sMax, sStp = (100,1050,50)
# minimum and maximum amount of iterations and the step increment
iMin, iMax, iStp = (10,300,10)

samples = [s for s in range(sMin, sMax+sStp, sStp)]
iterations = [i for i in range(iMin, iMax+iStp, iStp)]

meansU, convergenceU = sweepParameters(samples,iterations,'uniform')
meansL, convergenceL = sweepParameters(samples,iterations,'latin cube')
meansO, convergenceO = sweepParameters(samples,iterations,'orthogonal')

np.save('./data/meansU',meansU)
np.save('./data/convergenceU',convergenceU)
np.save('./data/meansL',meansL)
np.save('./data/convergenceL',convergenceL)
np.save('./data/meansO',meansO)
np.save('./data/convergenceO',convergenceO)