#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:24:38 2020

@author: tony
"""

from numpy import *
from matplotlib.pyplot import *
import sympy as sp

sp.init_printing(use_unicode=True)

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
    # improve this
    if isinstance(code, int):
        code = [code]
    elif isinstance(code, tuple): 
        code = list(code)
    
    multipliers = [ int(prod(circuit[i:])) for i in range(1,len(circuit)+1) ]
    sum_ = 0    
    for i, digit in enumerate(code): 
        sum_ += digit * multipliers[i]
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
        subtract_table = subtract.tt.invert_table()
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
        self.diffuse = diffuse
        
    def perm_io(self, in_code, out_code):
        """
        Only works for permutation tables
        """
        
        if self.diffuse: return
        self.table[in_code] = out_code
        self.table[out_code] = in_code
    
    def invert_table(self):
        """"
        This only works with 1-1 tables
        """
        if self.diffuse: return
        return dict(map(reverse, tt.items() ))

        
class gate:
    def __init__(self, tt, notes='A gate'):
        self.tt = tt
        self.circuit = tt.circuit
        self.notes = notes
        self.size = tt.size
        

class perm_gate(gate):
    def __init__(self, tt, notes='A perm gate'):
        gate.__init__(self, tt, notes)
        
    def apply(self, state):
        """
        This method accepts a state object, which has a dict of 
        { basis : amplitude } items.
        Using indx, it only looks at the basis digits 
        It simply changes each basis according to the truth table,
        leaving the amplitudes untouched.
        Appropriate for permutation operations.
        """
        #               don't change the iterate object
        for in_basis in list(state.keys()):
            # failing to find in_basis in the table will return in_basis
            out_basis = self.tt.table.get(in_basis, in_basis)
            if out_basis is not in_basis:
                # associated amplitude of in_basis with out_basis
                state[out_basis] = state.pop(in_basis)
        
        return state

class diff_gate(gate):
    """
    Note that every diff_gate() object gets passed a truth table that is the
    same size as the corresponding matrix.
    """
    def __init__(self, tt, notes='A diffusion gate'):
        gate.__init__(self, tt, notes)
        
    def apply(self, state):
        """
        This method accepts a state object, which has a dict of 
        { basis : amplitude } items. It transforms each basis into a list of 
        amplitudes, adding the amplitudes to a state vector. After all the
        adding, the nonzero elements of the state vector get re-encoded back 
        into basis : amp dictionarys
        Using indx, it only looks at the basis digits 

        """
        #empty state vector
        vector = zeros(self.size, object)
        #                  don't change the iterate object
        for in_basis, in_amp in list(state.items()):
            # account for single qudit gates
            if not isinstance(in_basis, tuple): in_basis = (in_basis,)

            out_pairs = self.tt.table.get(in_basis)
            out_bases = [ pair[0] for pair in out_pairs ]
            out_amps  = [ pair[1] * in_amp for pair in out_pairs ]
            out_locs  = [ get_location(self.circuit, code) for code in out_bases ]
            for loc, amp in zip(out_locs, out_amps):
                vector[loc] += amp
        out_state = {}
        for loc, amp in enumerate(vector):
            if amp == 0: continue
            basis = get_encoding(self.circuit, loc)
            out_state[ tuple(basis) ] = amp
        return out_state
    
class mat_gate(gate):
    def __init__(self,  tt, mat=None, notes='A mat gate'):
        if mat == None:
            self.mat = tt2mat(tt)
        self.unitary = is_unitary(self.mat)
        gate.__init__(self, tt)
        self.type = gate.mat.dtype
    
def tt2mat(tt):
    """
    This only works on permutation matrices.
    """
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
    target = [target]
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

def branch(n, m=None):
    """
    Returs a gate object based on the D() function
    """
    if m == None: m = n - 1
    
    tt = mat2tt(D(n, m))
    
    notes = 'Branching into first ' + str(m) + ' states'
    return diff_gate(tt, notes)
    
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
    ( (|0>, amp00), (|1>, amp01), ... ), 
    ( (|0>, amp10), (|1>, amp11), ... ), ...
    """
    dim = mat.shape[0]
    encoding = encode_state([dim])
    tt = truth_table([dim], diffuse=True)
    
    for i in range(dim):
        in_code = encoding[i]
        out_amps = mat[:,i]
        out_codes = [ (j, amp) for j, amp in enumerate(out_amps) ]
        tt.table[in_code] = tuple(out_codes)
    
    return tt

def printout(mat, encoding = None, notes = None):
    """
    This formats the matrix for human-readible inspection of the matrices
    """
    
    fig, ax = subplots(figsize=(8,8))
    mat_ax = ax.matshow(mat.astype(float), cmap='rainbow')
    mat_ax.set_clim([-1,1])
    # cbar.set_ticks = arange(-1,1,1/9) 
    dim = int(mat.shape[0] - 1)
    
    if mat.dtype == object:
        cbar = colorbar(mat_ax, ticks = arange(-1,1+1/dim,1/dim) ) 
        if dim > 10: cbar.ax.tick_params(rotation=90)
        generator = [r'$\frac{' + f"{i}" r'}{'
                     + str(dim) + r'}$'  for i in range(1,dim) ]
        rev_generator = [ i[:1] + '-' +  i[1:] for i in reversed(generator) ] 
        cbar_ticklabels = ['-1'] + rev_generator + ['0'] + generator + ['1']
        cbar.set_ticklabels(cbar_ticklabels)
        
        
        # judge = empty([dim+1,dim+1])
        # for i in range(dim+1):
        #     for j in range(dim+1):
        #         judge[i,j] = len((mat[i,j].simplify()).args) == 2
                
        # if any(judge):
        # mat = mat.astype(float)
        # mat = mat.round(4)

    for (i, j), z in ndenumerate(mat):
        ax.text(j, i, z, ha='center', va='center')
        
    if dim < 100 and encoding is not None:
        ax.set_xticks(range(dim+1)) # input
        ax.set_yticks(range(dim+1)) # output
        ax.set_xticklabels(encoding, rotation=90)
        ax.set_yticklabels(encoding)
    
    title(notes, pad = 50)
    show()

def goto_state(n, send=0, Print=False):
    """
    For equal superpositions of states in a n-dimensional system,
    this function will create an operator that sends all the amplitude
    to state |send>
    """
    m = n - 1
    base = (m*sp.sqrt(n))**-1
    y = sp.Rational(1,m) - base
    x = -sp.Rational(m-1,m) - base

    mat = ones([n,n], object) * y
    fill_diagonal(mat, x)
    mat[:,send] = 1/sp.sqrt(n)
    mat[send,:] = 1/sp.sqrt(n)
    
    if Print:
        printout(mat, encode_state([n]), notes='N=' + str(n) + ' Goto gate')
    
    tt = mat2tt(mat)
    notes = 'Size ' + str(n) + ', send '+ str(send) + ' goto_state gate'
    return diff_gate(tt, notes)

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
    tt.perm_io( (2,0), (2,1) )
    
    return perm_gate(tt, notes='Copy 32 from here')
    
if __name__ == '__main__':
    
    import q_program
    