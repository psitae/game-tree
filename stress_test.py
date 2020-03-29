#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 01:56:46 2020

@author: tony
"""


import operators as ops
import timeit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from resource import getpagesize

PAGESIZE = getpagesize()
PATH = Path('/proc/self/statm')

def get_resident_set_size() -> int:
    """Return the current resident set size in bytes."""
    # statm columns are: size resident shared text lib data dt
    statm = PATH.read_text()
    fields = statm.split()
    return int(fields[1]) * PAGESIZE

data = []
start_memory = get_resident_set_size()
n = 1
t = 0
under = 1 # runtime in seconds
while t < under:  
    qc = [2] * n

    start = timeit.default_timer()
    ops.encode_state(qc)
    stop = timeit.default_timer() 

    t = stop - start
    mem_usage = get_resident_set_size() - start_memory
    if mem_usage == 0:
        mem_usage = 1e-6
    data.append( (n, t, mem_usage) )
    
    print('N:\t\t', n)
    print('Time: \t', t)
    print('Memory:\t', mem_usage )

    n += 1
    
data = np.array(data)
x = data[:,0]
runtime = data[:,1]
mem = data[:,2]

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(x, runtime, 'go')
ax2.plot(x, mem, 'bo')

ax1.set_yscale('log')
ax2.set_yscale('log')
         
ax1.set_xlabel('2^N')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_ylabel('Runtime (s)')
ax2.set_ylabel('Memory usage (B)')

plt.title('Encoding states stress test under ' + str(under) + 's')
plt.show()
  
         
         
         
         
         