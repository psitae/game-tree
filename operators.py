#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:24:38 2020

@author: tony
"""

from numpy import *
from numpy import copy as numcopy
from matplotlib.pyplot import *
import sympy as sp
import copy
from math import log

# global unicode characters

arrow = '\u2192'

def gates(name, dim=2, reverse=False):
    if name == 'not':
        tt = p_truth_table([2])
        tt.perm_io( 0, 1)
        return perm_gate(tt, name='Not Gate')
    
    if name == 'cz':
        tt = d_truth_table([2,2])
        tt.table[ (1,1) ] =  {(1,1): -1}
        return diff_gate(tt, name='CZ gate')
    
    if name == 'flip':
        tt = d_truth_table([2])
        tt.table[ (1,) ] = { (1,): -1}
        return diff_gate(tt, name='Flip gate')
    
    if name == 'cnot':
        tt = p_truth_table([2,2])
        tt.perm_io( (1,1), (1,0) )
        return perm_gate(tt, name='CNot Gate')
    
    if name =='tonc':
        tt = p_truth_table([2,2])
        tt.perm_io( (1,1), (0,1) )
        return perm_gate(tt, name='Reverse CNot gate')
    
    if name == 'ccnot':
        tt = p_truth_table([2,2,2])
        tt.perm_io( (1,1,1), (1,1,0) )
        return perm_gate(tt, name='CCNot Gate')
    
    if name == 'hadamard':
        # produced hadamard matrix (dim * dim)
        # dim must be a power of two
        exponent = int( log(dim, 2) )
        if reverse is True:
            h2 = array([[-1,1],[1,1]], object)
        else:
            h2 = array([[1,1],[1,-1]], object)
        mat = 1
        for _ in range(exponent):
            mat = kron(mat, h2)
        
        prefactor = 1/sp.sqrt(dim)
        tt = mat2tt( prefactor * mat )
        return diff_gate(tt, name='N=' + str(dim) + ' Hadamard')

def encode_state(dims, type_='tuple', ket=False, Print=False):

    l = len(dims)
    state = [0]*l
    encoding = [ tuple(state) ]
    last_state = array(dims) - 1
    while not array_equiv(encoding[-1], last_state) :

        look = -1
        incrementable = False
        while not incrementable:   
            # this loop looks at each digit and changes it to the next
            # state 
            incrementable = state[look] + 1 < dims[look]
            if incrementable: 
                state[look] += 1
            else:
                state[look] = 0
                look -= 1

        encoding.append( tuple(state) )
    
    if type_ == 'tuple': pass
    if type_ == 'list': 
        for i in range(len(encoding)):
            encoding[i] = list(encoding[i])
    if ket:
        for i in range(len(encoding)):
            encoding[i] = '|' + ''.join(map(str,encoding[i])) + '>'

    if Print == True:
        print(encoding)
    
    return tuple(encoding)

def is_unitary(mat):
    candidate = mat @ mat.transpose()
    size = mat.shape[0]
    result = array_equiv(identity(size, object), candidate)
    
    if not result:
        # try simplifying all the entries
        for i in range(size):
            for j in range(size):
                candidate[i,j] = sp.simplify( candidate[i,j] )
        result = array_equiv(identity(size, object), candidate)
    return result

def makelist(*args):
    """
    I can't find a way for this to work when there is only one input
    """

    result = []
    for arg in args:
        if isinstance(arg, list): result.append(arg)
        elif isinstance(arg, tuple): result.append(list(arg))
        elif isinstance(arg, int): result.append([arg])
    
    return tuple(result)

def maketuple(*args):
    """
    I can't find a way for this to work when there is only one input
    """

    result = []
    for arg in args:
        if isinstance(arg, list): result.append(tuple(arg))
        elif isinstance(arg, int): result.append((arg,))
        elif isinstance(arg, ndarray): result.append(tuple(arg))
        elif isinstance(arg, tuple): result.append(arg)
    
    return tuple(result)

def printout(gate, title_ = None):
    """
    This formats the truth tabel and matrix for human-readible inspection 
    """
    if title_ == None: title_ = gate.name
    
    # create a figure with 2 subplots for the matrix and truth table
    fig, (mat_ax, tt_ax) = subplots(1, 2, figsize=(20,8)) #, constrained_layout=True)
    
    tt_ax.axis('off')
    mat_ax.set_title('Matrix', size=18, pad = 30)
    
    # print matrix
    if gate.mat is None: gate.mat = tt2mat(gate.tt)
    
    set_mat = mat_ax.matshow(gate.mat.astype(float), cmap='rainbow')
    set_mat.set_clim(-1,1)
    
    dim = int(gate.mat.shape[0] - 1)
    if gate.mat.dtype == object:
        cbar = colorbar(set_mat, ax=mat_ax, ticks = arange(-1,1+1/dim,1/dim) )
        # if dim > 10: cbar.ax.tick_params(rotation=90)
        # generator = [r'$\frac{' + f"{i}" r'}{'
        #              + str(dim) + r'}$'  for i in range(1,dim) ]
        # rev_generator = [ i[:1] + '-' +  i[1:] for i in reversed(generator) ]
        # cbar_ticklabels = ['-1'] + rev_generator + ['0'] + generator + ['1']
        # cbar.set_ticklabels(cbar_ticklabels)
    else:
        cbar = colorbar(set_mat, ax=mat_ax, ticks = arange(-1, 1, .5))
    
    for (i, j), z in ndenumerate(gate.mat):
        mat_ax.text(j, i, z, ha='center', va='center')
    
    # touch up matrix axis
    mat_ax.tick_params(labelsize = 15)
    cbar.ax.tick_params(labelsize = 15)
    if dim < 100:
        mat_ax.set_xticks(range(dim+1)) # input
        mat_ax.set_yticks(range(dim+1)) # output
        mat_ax.set_xticklabels(encode_state(gate.dims, ket=True), rotation=90)
        mat_ax.set_yticklabels(encode_state(gate.dims, ket=True))
    
    # print truth table
    tt_ax.text(0, .93, 'Truth Table', size = 18)
    tt_ax.text(0, .6, gate.stringify(), size = 14)
    
    fig.suptitle(title_, size=20, y=0.99) # pad = 50)
    subplots_adjust(wspace = 0)
    
def get_location(dims, code): 
    """
    e.g. dims: [2, 2, 2]
             code: [1, 0, 1]
     returns 5  
    """
    # improve this
    if isinstance(code, int):
        code = [code]
    elif isinstance(code, tuple):
        code = list(code)
    elif isinstance(code, str):
        code = [ int(char) for char in code ]
    
    multipliers = [ int(prod(dims[i:])) for i in range(1,len(dims)+1) ]
    sum_ = 0    
    for i, digit in enumerate(code): 
        sum_ += digit * multipliers[i]
    return sum_

def get_encoding(dims, location, type_='list'):
    divisors = [ int(prod(dims[i:])) for i in range(1,len(dims)+1) ]
    encoding = [0] * len(dims)
    for i in range(len(dims)):
        encoding[i] = int( floor(location/divisors[i]) % dims[i] )
    
    if type_ == 'tuple':
        encoding = tuple(encoding)
        
    return encoding

class truth_table:
    def __init__(self, dims):
        self.table = {}
        self.dims = dims
        self.size = int(prod(dims))
        self.mat  = None

class d_truth_table(truth_table):
    def __init__(self, dims):
        truth_table.__init__(self, dims)
        
class p_truth_table(truth_table):
    def __init__(self, dims):
        truth_table.__init__(self, dims)
        
    def perm_io(self, in_code, out_code):
        in_code, out_code = maketuple(in_code, out_code)
        if in_code != out_code:
            self.table[in_code] = out_code
            self.table[out_code] = in_code
    
    def invert_table(self):
        """"
        This only works with perm tables
        """
        return dict(map(reverse, tt.items() ))

class gate:
    def __init__(self, tt, name='A gate'):
        self.tt = tt
        self.table = tt.table
        self.name = name
        self.size = tt.size
        self.dims = tt.dims
        self.mat  = None
    
    def change_dims(self, new_dims):
        """
        This allows a gate to apply to different dimensions of qubits 
        as long as the product of the dimensions remains the same
        """
        
        if prod(new_dims) != self.size:
            print('Change dimension error')
            return
        
        # redo truth table encodings
        self.substitute_encoding(new_dims)
        
        # update gate dimensions
        self.dims = new_dims
    
    def substitute_encoding():
        print('Wrong substitute encoding')
        return
    
class perm_gate(gate):
    def __init__(self, tt, name='A perm gate'):
        gate.__init__(self, tt, name)
    
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
            out_basis = self.table.get(in_basis, in_basis)
            if out_basis is not in_basis:
                # associated amplitude of in_basis with out_basis
                state[out_basis] = state.pop(in_basis)
        
        return state
    
    def matrix_integration(self, circ_dims, indx):
        # set up repetitions
        rest_circ = list(circ_dims)
        for num, i in enumerate(indx):
            rest_circ.pop(i - num)
            
        rest_encoding = encode_state(rest_circ, type_='list')
        full_encoding = encode_state(circ_dims, type_='list')
        
        mat = identity(prod(circ_dims), uint8)
        for in_basis, out_basis in self.table.items():
            for code in rest_encoding:
                full_in_code = copy.copy(code)
                full_out_code = copy.copy(code)
                for jay, i in enumerate(indx):
                    full_in_code.insert(i, in_basis[jay])
                    full_out_code.insert(i, out_basis[jay])
                in_loc = full_encoding.index(full_in_code)
                out_loc = full_encoding.index(full_out_code)
                mat[ in_loc, in_loc ] = 0
                mat[ out_loc, in_loc ] = 1
        return mat
    
    def stringify(self):
        """
        Human-readable format of the truth table
        """
        fmt_items = []
        for key, val in self.table.items():
            key_str = '|' + ''.join(map(str, key)) + '>'
            val_str = '|' + ''.join(map(str, val)) + '>'
            fmt_items.append( key_str + arrow + val_str + '\n')
            
        return ''.join(fmt_items)
    
    def substitute_encoding(self, new_dims):
        new_table = {}
        for in_basis, out_basis in self.table.items():
            in_loc  = get_location(self.dims, in_basis)
            out_loc = get_location(self.dims, out_basis)
            new_in  = get_encoding(new_dims, in_loc, 'tuple')
            new_out = get_encoding(new_dims, out_loc, 'tuple')
            new_table[ new_in ] = new_out
        self.table = new_table
    
class diff_gate(gate):
    """
    Note that every diff_gate() object gets passed a truth table that is not
    necessarily the same size as the corresponding matrix.
    """
    def __init__(self, tt, name='A diffusion gate'):
        gate.__init__(self, tt, name)
        self.mat = tt.mat
        
    def apply(self, state):
        """
        This method accepts a state object, which has a dict of 
        { basis : amplitude } items. It transforms each basis into a list of 
        amplitudes, adding the amplitudes to a state vector. After all the
        adding, the nonzero elements of the state vector get re-encoded back 
        into { basis : amp } state objects
        """
        #empty state vector
        vector = zeros(self.size, object)
        #                  don't change the iterate object
        for in_basis, in_amp in list(state.items()):
            # account for single qudit gates
            if not isinstance(in_basis, tuple): in_basis = (in_basis,)
            
            out_dict = self.table.get(  in_basis, {in_basis: 1}  )
            for out_basis, out_amp in out_dict.items():
                loc = get_location(self.dims, out_basis) 
                vector[loc] += sp.simplify( out_amp * in_amp )
        state = {}
        for loc, amp in enumerate(vector):
            if amp == 0: continue
            basis = get_encoding(self.dims, loc)
            state[ tuple(basis) ] = amp
        return state
    
    def matrix_integration(self, circ_dims, indx):
        # set up repetitions
        rest_circ = list(circ_dims)
        for num, i in enumerate(indx):
            rest_circ.pop(i - num)
        
        rest_encoding = encode_state(rest_circ, type_='list')
        full_encoding = encode_state(circ_dims, type_='list')
        
        mat = identity(prod(circ_dims), object)
        for in_basis, out_dict in self.table.items():
            for code in rest_encoding:
                full_in_code = copy.copy(code)
                [ full_in_code.insert(i, in_basis[j]) for j, i in enumerate(indx)]
                in_loc = full_encoding.index(full_in_code)
                mat[ in_loc, in_loc ] = 0
                for basis, amp in out_dict.items():
                    full_out_code = copy.copy(code)
                    [ full_out_code.insert(i, basis[j])
                     for j, i in enumerate(indx) ]
                    out_loc = full_encoding.index(full_out_code)
                    mat[out_loc, in_loc] = amp
        return mat
    
    def stringify(self):
        """
        Returns human-readable format of truth table object
        """
        fmt_items = []
        for in_basis, out_dict in self.table.items():
            in_str = '|' + ''.join(map(str, in_basis)) + '>'
            output_str = ''
            for count, (basis, amp) in enumerate(out_dict.items()):
                if count != 0: output_str += ' + '
                basis = '|' + ''.join(map(str, basis)) + '>'
                amp = str(amp)
                output_str += amp + basis
            
            fmt_items.append( in_str + arrow + output_str + '\n')
        
        return ''.join(fmt_items)
    
    def substitute_encoding(self, new_dims):
        new_table = {}
        for in_basis, out_dict in self.table.items():
            in_loc = get_location(self.dims, in_basis)
            new_in = get_encoding(new_dims, in_loc, 'tuple')
            new_out_dict = {}
            for out_basis, out_amp in out_dict.items():
                out_loc = get_location(self.dims, out_basis)
                new_out = get_encoding(new_dims, out_loc, 'tuple')
                new_out_dict[ new_out ] = out_amp
            new_table[ new_in ] = new_out_dict
        self.table = new_table
    
class control_gate(gate):
    def __init__(self, which, tt, name = 'A control gate'):
        """
        which should be a perm_gate or diff_gate object
        """
        gate.__init__(self, tt, name)
        self.internal_gate = which(tt, name)
        
    def stringify(self):
        return self.internal_gate.stringify()
    #     """
    #     Returns human-readable format of truth table object
    #     """
    #     fmt_items = []
    #     for in_basis, outs in self.table.items():
    #         in_str = '|' + ''.join(map(str, in_basis)) + '>'
    #         output_str = ''
    #         for count, (basis, amp) in enumerate(outs.items()):
    #             if count != 0: output_str += ' + '
    #             basis = '|' + ''.join(map(str, basis)) + '>'
    #             amp = str(amp)
    #             output_str += amp + basis
            
    #         fmt_items.append( in_str + arrow + output_str + '\n')
        
    #     return ''.join(fmt_items)
    
    def apply(self, state):
        return self.internal_gate.apply(state)
    
    def matrix_integration(self, circ_dims, indx):
        return self.internal_gate.matrix_integration(circ_dims, indx)

def gate_concat(gate1, gate2):
    """
    Combines two gates into larger gate
    """
    dims = gate1.dims + gate2.dims
    l1, l2 = len(gate1.dims), len(gate2.dims)
    if isinstance(gate1, perm_gate) and isinstance(gate1, perm_gate):
        tt = p_truth_table(dims)
    else:
        tt = d_truth_table(dims)
    
    if isinstance(tt, p_truth_table):
        for code in encode_state(dims, type_='list'):
            in1, in2 = maketuple(code[:l1], code[-l2:])
            out1 = gate1.table.get(in1, in1)
            out2 = gate2.table.get(in2, in2)
            in_ = in1 + in2
            out = out1 + out2
            tt.perm_io( in_, out )
            
        gate3 = perm_gate(tt, gate1.name + ' [and] ' + gate2.name)
        
    elif isinstance(tt, d_truth_table):
        # chaos ensues
        pass
    
    return gate3
    
def tt2mat(tt):
    """
    Truth table to matrix 
    """
    if isinstance(tt, d_truth_table):
        mat = identity(tt.size, object)
        for in_code, out_table in tt.table.items():
            in_loc = get_location(tt.dims, in_code)
            mat[ in_loc, in_loc ] = 0
            for basis, amp in out_table.items():
                out_loc = get_location(tt.dims, basis)
                mat[ out_loc, in_loc ] = amp
    
    if isinstance(tt, p_truth_table):
        mat = identity(tt.size, uint8)
        for in_code, out_code in tt.table.items():
            in_loc = get_location(tt.dims, in_code)
            out_loc = get_location(tt.dims, out_code)
            mat[in_loc, in_loc ] = 0
            mat[out_loc, in_loc] = 1
    
    return mat

def mat2tt(mat):
    """
    This function creates a truth table with keys as
    |0>,|1>, ...
    and values as
    { |0>:amp00, |1>:amp01 , ... }, 
    { |0>:amp10, |1>:amp11, ... }, ...
     and saves the matrix in tt.mat
    
    This function currently only works on single-qudit operations
    """
    dim = mat.shape[0]
    encoding = encode_state([dim])
    tt = d_truth_table([dim])
    tt.mat = mat
    
    for i in range(dim):
        in_code = encoding[i]
        out_amps = mat[:,i]
        out_codes = [ ((j,), amp) for j, amp in enumerate(out_amps) if amp != 0]
        tt.table[in_code] = dict(out_codes)
    
    # search for redundant entries like |00> --> |00>
    for in_basis, out_dict in list(tt.table.items()):
        for out_basis, amp in out_dict.items():
            if in_basis == out_basis and amp == 1:
                tt.table.pop(in_basis)
        
    return tt

def fan_out(dim, control, target, Print=False):
    """
    This changes the basis for each amplitude, so it is a permutation gate.
    It accepts the dimension of the two qudits, and their locations. It copies
    the basis value of the control onto the target.
    If everything is working properly, it will only ever copy the basis value
    of one qudit onto a basis this is currently 0. Otherwise, a quantum logic
    error has occurred.
    """
    target = [target]
    size = len(target) + 1
    dims = [dim] * size
    tt = p_truth_table(dims)
    
    for i in range(1, dim):
        in_code = array([0]*size)
        in_code[control] = i
        out_code = tuple([i]*size)
        in_code = tuple(in_code)
        tt.table[in_code] = out_code
        tt.table[out_code] = in_code
    
    return perm_gate(tt, name='Fan out gate')

def arith(dims, control, target, op='add'):
    """
    This gate adds the controls together and then either performs modular 
    addition or subtract with the target
    """
    
    tt = p_truth_table(dims)
    encoding = encode_state(dims)
    control_sum = []
    for code in encoding:
        control_sum.append(sum(code[control]))
        
    if op == 'add':
        for i, in_code in enumerate(encoding):
            sum_ = sum(in_code[target] + control_sum[i] ) % dims[target]
            out_code = list( in_code )
            out_code[target] = sum_
            tt.perm_io( in_code, out_code )
        result = perm_gate(tt, name = 'Add mod ' + str(dims[control]) )
            
    if op == 'subtract':
        # subtract is the inverse of add
        # consider arith( add ) and then invert the dictionary
        
        subtract = arith(dims, control, target, mode = 'add')
        subtract_table = subtract.tt.invert_table()
        subtract.tt.table = subtract_table
        subtract.name = 'Subtract mod ' + str(dims[control])
        result = subtract
        # for i, in_code in enumerate(encoding):
        #     diff_ = sum(in_code[target] - control_sum[i] ) % dims[target]
        #     out_code = list( in_code )
        #     out_code[target] = diff_
        #     tt.table[in_code] = tuple(out_code)
        # result = gate(tt, mat_type = 'perm', mode = 'full' )
    
    return result

def branch(n, m=None):
    """
    Returns a gate object based on the D() function
    """
    if m == None: m = n - 1
    
    tt = mat2tt(D(n, m))
    
    name = 'Branching gate, n=' + str(n)
    return diff_gate(tt, name)
    
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
    
    prefactor = sp.Rational(1,m)
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

def goto_state(n, send=1):
    """
    For equal superpositions of non-0 states in a n-dimensional system,
    this function will create an operator that sends all the amplitude
    to state |send>
    
    Note: currently not set up to send to 0
    """
    m = n - 1
    p = n - 2
    base = (p*sp.sqrt(m))**-1
    y = sp.together( sp.Rational(1,p) - base )
    x = sp.together(-sp.Rational(p-1,p) - base )
    # y = sp.Rational(1,p) - base 
    # x = -sp.Rational(p-1,p) - base 

    mat = ones([m,m], object) * y
    fill_diagonal(mat, x)
    mat[:,send-1] = 1/sp.sqrt(m)
    mat[send-1,:] = 1/sp.sqrt(m)
    
    result = identity(n, object)
    result[1:,1:] = mat
    
    tt = mat2tt(result)
    
    name = 'Size ' + str(n) + ', ' + arrow + str(send) + ' Goto-state gate'
    return diff_gate(tt, name)

def swap(dim, state1, state2):
    """
    swaps state1 and state2 in a D-dimensional system
    """
    tt = p_truth_table([dim])
    tt.perm_io(state1, state2)
    
    name = str(dim) + '-D Swap ' + str(state1) + '-' + str(state2)
    return perm_gate(tt, name )

def AND(control=[0,1], target=2):
    """
    expecting controls and target as qubits
    control & target are the indexes
    """
    len_ = len(control + [target])
    dims = array([2]*len_)
    tt = p_truth_table(dims)
    out_code = (1,) * len_
    in_code = list(out_code)
    in_code[target] = 0
    in_code = tuple(in_code)
    
    tt.table[in_code] = out_code
    tt.table[out_code] = in_code
    
    return perm_gate(tt, name='AND gate')

def SAME(control=[0,1], target=2):
    """
    as AND()
    """
    len_ = len(control + [target])
    dims = array([2]*len_)
    tt = p_truth_table(dims)
    out_code = (1,) * len_
    in_code = list(out_code)
    in_code[target] = 0
    tt.perm_io( in_code, out_code)
    
    in_code = (0,0,0)
    out_code = [0,0,0]
    out_code[target] = 1
    tt.perm_io( in_code, out_code )
    
    return perm_gate(tt, name='SAME gate')

def OR(control=[0,1], target=2):
    """
    as AND()
    """
    len_ = len(control + [target])
    dims = array([2]*len_)
    
    tt = p_truth_table(dims)
    encoding = encode_state(dims[control])
    
    for control_in in encoding:
        if not any(control_in): 
            continue
        in_code = list(control_in)
        in_code.insert(target, 0)
        out_code = numcopy(in_code)
        out_code[target] = 1
        tt.perm_io(in_code, out_code)
    
    return perm_gate(tt, name='OR gate')
    
def copy32(control=0, target=1):
    """
    This function copies the contents of the |1> |2> qutrit states 
    onto a target qubit
    control and target give the indices 
    """
    dims = [3]*2
    dims[target] = 2
    tt = p_truth_table(dims)
    in_ = [0,0]
    in_[control] = 2
    out = [1,1]
    out[control] = 2
    tt.perm_io( in_, out )
    
    return perm_gate(tt, name='Copy 32')

def not32(control=0, target=1):
    """ 
    Copy32() + NOT gate
    """
    dims = [3]*2
    dims[target] = 2
    tt = p_truth_table(dims)
    in_ = [0,0]
    in_[control] = 1
    out = [1,1]
    tt.perm_io( in_, out )
    
    return perm_gate(tt, name='Not-Copy 32')
    
def create_control(dims, control, target, directions):
    """
    Creates a control gate. Contol and Target give indices.
    Directions is a dictionary pairing of control basis states and
    operations on the target(s)
    """
    control, target = makelist( control, target )
    if any([isinstance(val, diff_gate) for val in directions.values() ]):
        tt = d_truth_table(dims)
        name = 'Diffusion control gate'
        parent = diff_gate
    else:
        tt = p_truth_table(dims)
        name = 'Permutation control gate'
        parent = perm_gate
    
    if isinstance(tt, d_truth_table):
        for ctl, gate in directions.items():
            for in_, out_dict in gate.table.items():
                extend_in = list(in_)
                [ extend_in.insert(i, ctl[j]) for j, i in enumerate(control) ]
                extend_out = {}
                for out_basis, out_amp in out_dict.items():
                    extend_basis = list(out_basis)
                    [ extend_basis.insert(i, ctl[j]) for j, i in enumerate(control) ]
                    extend_basis = tuple(extend_basis)
                    extend_out[ extend_basis ] = out_amp
                    extend_in = tuple(extend_in)
                tt.table[ extend_in ] = extend_out
    
    if isinstance(tt, p_truth_table):
        for ctl, gate in directions.items():
            for in_, out in gate.table.items():
                extend_in, extend_out = makelist(in_,out)
                for j, i in enumerate(control):
                    extend_in.insert(i, ctl[j])
                    extend_out.insert(i, ctl[j])
                tt.perm_io( extend_in, extend_out )
        
    return control_gate(parent, tt, name)

def one_shot_grover():
    s = ones(4, object) * sp.Rational(1,2)
    mat = 2*outer(s,s) - identity(4, object)
    tt = mat2tt(mat)
    
    return diff_gate(tt, name='One shot grover')
    
    
if __name__ == '__main__':
    
    # import q_program
    g = SAME()
    # goto_false = swap(3, 0, 1)
    # dirs2 = { (0,) : goto_false }
    # ctrl2 = create_control([2,3], 0, 1, dirs2)
    printout(g)
