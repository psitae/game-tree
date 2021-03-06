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
        
    return tuple(encoding)

def is_unitary(mat):
    candidate = mat @ mat.transpose()
    size = mat.shape[0]
    result = array_equiv(identity(size, object), candidate)
    return result
    

def get_location(circuit, code): 
    """
    e.g. circuit: [2, 2, 2]
             code: [1, 0, 1]
     returns 5  
    """
    multipliers = [ int(prod(circuit[i:])) for i in range(1,len(circuit)+1) ]
    sum_ = 0    
    for i, digit in enumerate(code): sum_ += digit * multipliers[i]
    return sum_

def get_encoding(circuit, location):
    divisors = [ int(prod(circuit[i:])) for i in range(1,len(circuit)+1) ]
    encoding = [0] * len(circuit)
    for i in range(len(circuit)):
        encoding[i] = int( floor(location/divisors[i]) % circuit[i] )
    return encoding

def arith(circuit, control, target, op='add', Print=False):
    # flexible input
    # control, target = makelist(control, target)
    
    tt = truth_table(circuit)
    encoding = encode_state(circuit)
    control_sum = []
    for code in encoding:
        control_sum.append(sum(code[control]))
        
    if op == 'add':
        for i, in_code in enumerate(encoding):
            sum_ = sum(in_code[target] + control_sum[i] ) % circuit[target]
            out_code = list( in_code )
            out_code[target] = sum_
            tt.table[in_code] = tuple(out_code)
        result = perm_gate(tt, mode = 'full' )
            
    if op == 'subtract':
        # subtract is the inverse of add
        # consider arith( add ) and then invert the dictionary
        
        subtract = arith(circuit, control, target, mode = 'add', Print=Print)
        subtract_table = subtract.tt.invert_table() )
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
        self.size = int(prod(circuit))
        
    def perm_io(self, in_code, out_code):
        self.table[in_code] = out_code
        self.table[out_code] = in_code
    
    def invert_table(self):
        """"
        This only works with 1-1 tables
        """
        
        return dict(map(reverse, tt.items() ))

        
class gate:
    def __init__(self, tt, notes='A gate'):
        self.tt = tt
        self.circuit = tt.circuit
        self.notes = notes
        self.size = tt.size
        
    def lookup_one(self, tt, state):
        """
        This method expects a list of only the populated states,
        listed as ( amplitude, basis ) tuples.
        It simply changes each basis according to the truth table,
        leaving the amplitudes untouched.
        Appropriate for permutation operations.
        """
        in_basis = [ s[1] for s in state ]
        for i, basis in enumerate(in_basis):
            out_basis = tt.table.get(basis)
            state[i][1] = out_basis
        
        return state
    
class perm_gate(gate):
    def __init__(self, tt, notes='A perm gate'):
        gate.__init__(self, tt, notes)
        
    def apply(self, state):
        return self.lookup_one(self.tt, state)
    
    
class mat_gate(gate):
    def __init__(self,  tt, mat=None, notes='A mat gate'):
        if mat == None:
            self.mat = tt2mat(tt)
        self.unitary = is_unitary(self.mat)
        gate.__init__(self, tt)
        self.type = gate.mat.dtype
    
    

class diff_gate(gate):
    def __init__(self, tt):
        gate.__init(self, tt)

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

def fan_out(dim, control, target, Print=False):
    """
    This changes the basis for each amplitued, so it is a permutation gate.
    It accepts the dimension of the two qudits, and their locations. It copies
    the basis value of the control onto the target.
    If everything is working properly, it will only ever copy the basis value
    of one qudit onto a basis this is currently 0. Otherwise, a quantum logic
    error has occurred.
    """
    
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
    
    return perm_gate(tt, notes='Fan out gate')

def D(n, m = None):
    """
    These operators act on initialized states ( |0> ) of size n.
    They cause the population in |0> to equally distribute to the first m states
    |1>, |2>, ... |m>
    
    These gates are their own inverse.
    This structure reflects the game tree parent-children node connection 
    credit: https://quantumcomputing.stackexchange.com/questions/10239/how-can-i-fill-a-unitary-knowing-only-its-first-column
    """
    if m == None: m = n-1
    
    prefactor = sp.Rational(-1,m)
    block = -ones([m+1,m+1], dtype=object)
    # diag_ = ones(m+1) * (m-1)
    fill_diagonal(block, (m-1))
    block[:,0] = sp.sqrt(m)
    block[0,:] = sp.sqrt(m)
    block[0,0] = 0
    block = block * prefactor
    
    result = identity(n, dtype=object)
    result[:m+1,:m+1] = block

    return result

def mat2tt(mat):
    """
    This function creates a truth table with keys as
    |0>,|1>, ...
    and values as
    ((amp0,|0>), (amp1, |1>), ...)
    
    """
    dim = mat.shape[0]
    encoding = [ (int(i),) for i in range(dim) ]
    tt = truth_table([dim], diffuse=True)
    
    for i in range(dim):
        in_code = encoding[i]
        out_amps = mat[:,i]
        out_codes = [ (amp, i) for i, amp in enumerate(out_amps) ]
        tt.table[in_code] = tuple(out_codes)
    
    return tt

def printout(mat, encoding = None, notes = None):
    """
    This formats the matrix for human-readible inspection of the matrices
    """
    
    fig, ax = subplots(figsize=(5,5))
    mat_ax = ax.matshow(mat.astype(float), cmap='rainbow')
    mat_ax.set_clim([-1,1])
    # cbar.set_ticks = arange(-1,1,1/9) 
    dim = int(mat.shape[0] - 1)
    
    if mat.dtype == object:
        cbar = colorbar(mat_ax, ticks = arange(-1,1+1/dim,1/dim) ) 
        if dim > 10: cbar.ax.tick_params(rotation=90)
        generator = [r'$\frac{' + f"{i}" r'}{' + str(dim) + r'}$'  for i in range(1,dim) ]
        rev_generator = [ i[:1] + '-' +  i[1:] for i in reversed(generator) ] 
        cbar_ticklabels = ['-1'] + rev_generator + ['0'] + generator + ['1']
        cbar.set_ticklabels(cbar_ticklabels)
        
        
        judge = empty([dim+1,dim+1])
        for i in range(dim+1):
            for j in range(dim+1):
                judge[i,j] = len((mat[i,j].simplify()).args) == 2
                
        if any(judge):
            mat = mat.astype(float)
            mat = mat.round(2)

    for (i, j), z in ndenumerate(mat):
        ax.text(j, i, z, ha='center', va='center')
        
    if dim < 100 and encoding is not None:
        ax.set_xticks(range(dim+1)) # input
        ax.set_yticks(range(dim+1)) # output
        ax.set_xticklabels(encoding, rotation=90)
        ax.set_yticklabels(encoding)
    
    title(notes, pad = 50)
    show()

def go_to_state(n,m):
    """
    For equal superpositions of states in a n-dimensional system,
    this function will create an operator that sends all the amplitude
    to state |m>
    """
    # preprocessing
    overlap = 1/sp.sqrt(n)
    
    # compute angles
    theta_val = ( sp.asin(overlap) )
    phi_val = ( sp.acos(overlap)/2 )
    
    # flip state & operator
    f = sp.cos( theta_val + phi_val ), sp.sin( phi_val + theta_val ) 
    flip_ =  nans.dot(f[0]) + ans.dot(f[1])
    diffuse_op = 2 * outer(flip_, flip_) - identity(n, object)
    
    return diffuse_op

def AND(control, target):
    """
    expecting controls and qutrits and target as bit
    control & target are the indexes
    """
    len_ = len(control + [target])
    circuit = array([2]*len_)
    tt = truth_table(circuit)
    out_code = (1,) * len_
    in_code = list(out_code)
    in_code[target] = 0
    in_code = tuple(in_code)
    
    tt.table[in_code] = out_code
    tt.table[out_code] = in_code
    
    return perm_gate(tt, notes='AND gate')

def OR(control, target):
    """
    as AND()
    """
    len_ = len(control + [target])
    circuit = array([2]*len_)
    
    tt = truth_table(circuit)
    encoding = encode_state(circuit[control])
    
    for control_in in encoding:
        if not any(control_in): 
            continue
        in_code = list(control_in)
        in_code.insert(target, 0)
        out_code = copy(in_code)
        out_code[target] = 1
        in_code = tuple(in_code)
        out_code = tuple(out_code)
        print('Control_in', control_in)
        print('In_code', in_code)
        print('Out code', out_code)
        tt.perm_io(in_code, out_code)
    
    return perm_gate(tt, notes='OR gate')
    
def copy32(control, target):
    """
    This function copies the contents of the |1> |2> qutrit states 
    onto a target qubit
    control and target give the indices 
    """
    circuit = [3]*2
    circuit[target] = 2
    tt = truth_table(circuit)
    in_code = (2,0)
    out_code = (2,1)
    tt.perm_io( in_code, out_code )
    
    return perm_gate(tt, notes='Copy 32')
    
    
# g = copy32(0,1)
# print(g.tt.table)
# printout(tt2mat(g.tt), encoding = encode_state(g.tt.circuit, Print=False), notes=g.notes )
get_encoding([2,2], 0)