'''
Temperature and humidity data for each day and every sensor type (S1, S2, S3, S4) should obey a certain function type as below:
    S1 --> Exponential (exp)
    S2 --> Linear (linear)
    S3 --> Logarithmic (log_n)
    S4 --> Trigonometric (trig)

*** All the functions have a range of 2.5 to 5 degrees of variation in the range for temperature and 5 to 10 percent variation for humidity.
The range that these functions designed for are based on integers of 0 to 11. This is because the hours in am and pm are ranging in [0:12]
usage:
    exp(np.arange(12), 'Temperature', seed=20)
'''

import numpy as np
import random
# remember to include 'random.seed(42)' inside the main program to make data reproducable


def exp(x, c, seed=None):
    rand = np.random.RandomState(seed)
    min_value, max_value, range_value = c_value(c)
    a = (range_value/(np.exp(11)-1))*rand.choice(np.linspace(0.5,1, num=10), p=[0.1, 0.05, 0.03, 0.11, 0.02, 0.14, 0.2, 0.15, 0.16, 0.04])
    b_lower = min_value - a
    b_upper = max_value - a*np.exp(11)
    b = rand.choice(np.linspace(b_lower, b_upper, num=10), p=[0.1, 0.05, 0.03, 0.11, 0.02, 0.14, 0.2, 0.15, 0.16, 0.04])
    return a*np.exp(x)+b

def linear(x, c, seed=None):
    rand = np.random.RandomState(seed)
    min_value, max_value, range_value = c_value(c)
    a = (range_value/11)*rand.choice(np.linspace(0.5, 1, num=20))
    b_lower = min_value
    b_upper = max_value - 11*a
    b = rand.choice(np.linspace(b_lower, b_upper, num=20))
    return a*x+b

def log_n(x, c, seed=None):
    rand = np.random.RandomState(seed)
    min_value, max_value, range_value = c_value(c)
    a = (range_value/np.log(12))*rand.choice(np.linspace(0.5, 1, num=20))
    b_lower = min_value
    b_upper = max_value - a * np.log(12)
    b = rand.choice(np.linspace(b_lower, b_upper, num=20))
    return a*np.log(x+1)+b

def trig (x, c, seed=None):
    rand = np.random.RandomState(seed)
    min_value, max_value, range_value = c_value(c)
    a = range_value * rand.choice(np.linspace(0.5, 1, num=20))
    b_lower = min_value
    b_upper = max_value - a
    b = rand.choice(np.linspace(b_lower, b_upper, num=20))
    return a*np.abs(np.cos((x*2*np.pi)/11))+b # The function is des    igned to have a full circle rotation in the range [0,11]
 
 # This function determines the min, max and range of the other functions in this script.
def c_value(c):
    while True:
        if c=='Temperature':
            min_value = 10
            max_value = 60
            range_value = 5 + 2.5 * random.random() # This adds rand    omness to the value_range
        elif c=='Humidity':
            min_value = 0
            max_value = 100
            range_value = 10 + 5 * random.random()
        break
    return min_value, max_value, range_value
