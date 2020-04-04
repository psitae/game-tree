#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:24:38 2020

@author: tony
"""

from numpy import *
from matplotlib.pyplot import *
import sympy as sp

def encode_state(circuit, Print=False):

    l = len(circuit)
    state = [0]*l
    encoding = [ tuple(state) ]
    last_state = array(circuit) - 1
    while not array_equiv(encoding[-1], last_state) :

        look = -1
        incrementable = False
        while not incrementable:   
            # this loop looks at each digit and changes it to the next
            # state 
            incrementable = state[look] + 1 < circuit[look]
            if incrementable: 
                state[look] += 1
            else:
                state[look] = 0
                look -= 1

        encoding.append( tuple(state) )

    if Print == True:
        print(encoding)
        
    return encoding

def makelist(*args):
    returns = []
    for a in args:
        if type(a) == list: returns.append( a )
        if type(a) == int: returns.append([a])
        if type(a) == array: returns.append( list(a) )
        
    return returns

def get_location(circuit, code): 
    # e.g. circuit: [2, 2, 2]
    #         code: [1, 0, 1]
    # returns 5  
    multipliers = [ int(prod(circuit[i:])) for i in range(1,len(circuit)+1) ]
    sum_ = 0    
    for i, digit in enumerate(code): sum_ += digit * multipliers[i]
    return sum_

def get_encoding(circuit, location):
    divisors = [ int(prod(circuit[i:])) for i in range(1,len(circuit)+1) ]   
    encoding = [0]*len(circuit)
    for i in range(len(circuit)): encoding[i] = floor(location/divisors[i]) % circuit[i]
    return encoding

def arith(circuit, control, target, mode='add', Print=False):
    # flexible input
    # control, target = makelist(control, target)
    
    tt = truth_table(circuit)
    encoding = encode_state(circuit)
    control_sum = []
    for code in encoding:
        control_sum.append(sum(code[control]))
        
    if mode == 'add':
        for i, in_code in enumerate(encoding):
            sum_ = sum(in_code[target] + control_sum[i] ) % circuit[target]
            out_code = list( in_code )
            out_code[target] = sum_
            tt.table[in_code] = tuple(out_code)
        result = gate(tt, mat_type ='perm', mode = 'full' )
            
    if mode == 'subtract':
        # subtract is the inverse of add
        # consider arith( add ) and then invert the dictionary
        
        subtract = arith(circuit, control, target, mode = 'add', Print=Print)
        subtract_table = tt_flip(subtract.tt.table.items() )
        subtract.tt.table = subtract_table
        result = subtract
        # for i, in_code in enumerate(encoding):
        #     diff_ = sum(in_code[target] - control_sum[i] ) % circuit[target]
        #     out_code = list( in_code )
        #     out_code[target] = diff_
        #     tt.table[in_code] = tuple(out_code)
        # result = gate(tt, mat_type = 'perm', mode = 'full' )
    
    if Print:
        [ print(key, ':', val) for key, val in tt.table.items() ]
    
    return result
            
class truth_table:
    def __init__(self, circuit, diffuse=False):
        self.table = {}
        self.circuit = circuit
        self.size = prod(circuit)
        self.diffuse = diffuse

class full_gate(gate):
    def __init__(self,  tt):
        self.mat = tt2mat(tt)
        self.unitary = is_unitary(self.mat)
        gate.__init__(self, tt)
        
class one_to_many(gate):
class gate:
    def __init__(self, tt, notes='A gate'):
        self.tt = tt
        self.circuit = tt.circuit
        self.notes = notes
        self.size = tt.size
        
        # this changes with different modes
        self.mode = 'fast'
        self.mat = None
        self.unitary = None

    def init_full(self):
        self.mode = 'full'
        self.mat = tt2mat(tt)
        self.unitary = self.is_unitary()
        
    def printout(self):
        if self.mode == 'fast':
            print('Fast mode -- no matrix')
            return
        matshow(self.mat)
        # colorbar()
        if self.size < 100:
            encoding = encode_state(self.circuit)
            xticks(range(self.size), encoding, rotation=90) # input
            yticks(range(self.size), encoding)              # output
        title(self.notes, pad = 50)
    
    def is_unitary(self):
        candidate = self.mat.dot(self.mat.transpose())
        result = array_equiv(identity(self.size), candidate.round(3))
        # round exists to correct floating point errors, to do with square roots
        return result
    

def tt2mat(tt):
    # this only works on permutation matrices
    mat = identity(tt.size, dtype = int8)
    for in_code in tt.table:
        out_code = tt.table[in_code]
        in_loc = get_location(tt.circuit, out_code)
        out_loc = get_location(tt.circuit, in_code)
        mat[in_loc, in_loc ] = 0
        mat[out_loc, out_loc ] = 0
        mat[out_loc, in_loc] = 1

    return mat

def tt_flip(tt):
    return dict(map(reverse, tt.items() ))

def fan_out(dim, control, target, Print=False):
    # flexible input
    [target] = makelist(target)
    size = len(target) + 1
    circuit = [dim] * size
    tt = truth_table(circuit)
    
    for i in range(1, dim):
        in_code = array([0]*size)
        in_code[control] = i
        out_code = tuple([i]*size)
        in_code = tuple(in_code)
        tt.table[in_code] = out_code
        tt.table[out_code] = in_code
    
    return gate(tt, 'symmetric', notes='Fan out gate')

def D(i):
    # these operators diffuse from the |0> state evenly to all the other states
    # this structure reflects the game tree parent-children node connection 
    # credit: https://quantumcomputing.stackexchange.com/
    # questions/10239/how-can-i-fill-a-unitary-knowing-only-its-first-column
    
    prefactor = sp.Rational(1,i)
    diag_ = ones(i+1) * (i-1)
    diag_[0] = 0
    result = -ones([i+1, i+1], dtype=object)
    result[:,0] = sp.sqrt(i)
    result[0,:] = sp.sqrt(i)
    fill_diagonal(result, (i-1))
    result[0,0] = 0

    return prefactor*result

def mat2tt(mat):
    # for right now, this function is specifically designed
    # to work with D matrices, which apply to one qudit only
    
    dim = shape.mat[0]
    encoding = [ tuple(i) for i in range(dim) ]
    tt = truth_table([dim], diffuse=True)
    
    
    
def diffuse():
    1

D(3)






