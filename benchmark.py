'''
This script receives functions in pandas and estimates the amount of time (in seconds) the it takes to perform that operation:
Usage: benchmark(lambda: func(a, b))
'''

def benchmark(func):
    import time
    start = time.time()
    result = func()
    return time.time() - start
