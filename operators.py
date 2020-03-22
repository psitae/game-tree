# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:19:25 2020

@author: T Dean
"""

from numpy import *
from matplotlib.pyplot import *
from math import log
from more_itertools import consecutive_groups

def k_add(a,b): # expects two square matrices 
    # I can't believe I had to program this. It is a standard direct sum operation.
    # it will create a block diagonal matrix with the two inputs
    #
    # k is for kronecker, which is a letter that sticks out nicely
    
    tr = zeros([a.shape[0],b.shape[1]])
    bl = tr.transpose()
    top = concatenate([a, tr], axis=1)
    bottom = concatenate([bl, b], axis=1)
    both = concatenate([top, bottom])
    return both

def many_kadd(x): # expecting a list of matrices
    # calls k_add multiple times
    
    current = k_add(x[0],x[1])

    for i in range(len(x)-2):
        current = k_add(current, x[i+2])
    return current

def many_kron(x): # expecting a list of matrices
    # the input list will all be tensor producted together in order
    
    current = kron(x[0], x[1])
    for i in range(len(x)-2):
        current = kron(current, x[i+2])
    return current
        
def state_swap(dim, a=0, b=1): # swaps ath and bth element in a (dim x dim) matrix
    # currently unused
    mat = diag(ones(dim))
    mat[b,b] = 0 
    mat[a,a] = 0
    mat[b,a] = 1
    mat[a,b] = 1
    return mat
    
def is_unitary(mat):
    size = mat.shape[0]
    candidate = mat.dot(mat.transpose())
    result = array_equiv(identity(size), candidate.round(10))
    # round exists to correct floating point errors, to do with square roots
    return result 

def jumble(mat, a, b): # expecting a square matrix
    # a and b are the two factors that multiple to dim(mat)
    
    # to improve: the matrix will always have a clean tensor product in normal use.
    # just take one from the u-l of each block, and put that matrix as the tensor of the identity: identity (x) stuff
    
    # also to improve: jumble groups of qudits together
    
    # make dictionary
    # index = list(range(a*b))
    # piece = list(range(b))
    # groups = []
    # total = 0
    # for i in range(a):

    #     groups.append([ b*i*total + x for x in piece ])
    #     total = total + 1
    
    new_order = []
    total = 0
    j_range1 = range(a)
    for i in range(b):
        j_range2 = [ b*k + i for k in j_range1 ]
        for j in j_range2:
            new_order.append(j) 
            total += 1
            
    rows = []
    cols = []
    
    for i in range(a*b):
        rows.append(tuple(mat[:, i]))
    
    for i in range(a*b):
        mat[:, i ] = rows[ new_order[i] ]

    for i in range(a*b):
        cols.append(tuple(mat[i, :]))

    for i in range(a*b):
        mat[ i, : ] = cols[ new_order[i] ]
    return mat

def copy(x): # expecting qubit dimension
    # this operator will copy one basis vector to another
    # ONLY IF the target starts out at |0>
    # i.e. |x0> --> |xx>
    # 
    # currently unused

    result = identity(x**2)
    for i in range(1,x):
        result[ i*x, i*x ] = 0
        result[ i*x+i, i*x+i ] = 0
        result[ i*x+i, i*x] = 1
        result[ i*x, i*x + i ] = 1
        
    return result

def middle_tensor(small_op, division, repeat):
    # small_op will be divided up, kroneckered with id(repeat),
    # put back together and returned
    
    repeat_op = identity(repeat)
    # divide small_op up according to control dimensions before the middle
    dim = small_op.shape[0]
    small_dim = dim // division
    pieces = zeros([division, division], dtype=object)
    for i in range(division):   
        for j in range(division):
            slice1 = slice(small_dim*i, small_dim*i + small_dim)
            slice2 = slice(small_dim*j, small_dim*j + small_dim)
            piece = small_op[slice1,slice2]
            # kron copies the information, representing middle qudits
            pieces[i][j] = kron(repeat_op, piece)
    
    # put the repeated_pieces back together into one matrix
    tot_size = division * small_dim * repeat
    result = zeros([tot_size, tot_size])
    for i in range(division):
        for j in range(division):
            ri = small_dim * i * repeat
            rj = small_dim * j * repeat
            slice1 = slice(ri, ri + small_dim * repeat)
            slice2 = slice(rj, rj + small_dim * repeat)
            result[slice1, slice2] = pieces[i][j]
            
    return result        

def integrate(circuit, control, target, small_op, Print=False):
    # circuit : list of dimensions of all qudits
    # control & target : list of indices
    # input: small_op is the operation, acting on a list of control 
    # and target indices
    # integrate() creates the uber matrix for the quantum circuit
    # 
    # assume all controls before all targets
    print('circuit', circuit)
    print('index  ', list(range(len(circuit))))
    print('control', control)
    print('target', target)
    # front
    front_iterable = range(0, control[0])
    front_slice = slice(0, control[0])
    front_size = prod(circuit[front_slice])
    print('Front size', front_size)
    front = identity(int(front_size))
    print('Front\n', front)

    # end
    end_iterable = range(target[-1]+1,len(circuit))
    end_slice = slice(target[-1]+1, -1)
    end_size = prod(circuit[target[0]:-1])
    end = identity(int(end_size))
    print('end\n', end)
 
    # distinguish meaningful from redundant instructions
    # within the target range
    actors = list( hstack( [control, target] ))
    nonactors = list(range(len(circuit)))
    print('actors', actors)
    # do this part better later
    [ nonactors.remove(i) for i in actors ]
    # maybes
    maybes = [front_slice, end_slice]
    [ nonactors.remove(i) for i in front_iterable ]
    [ nonactors.remove(i) for i in end_iterable ]
    # what's left should only be middle nonactors
    print('nonactors', nonactors)
    # group indices of the circuit into consecutive sequences
    # consec. seq. are already tensored together, just figure out the rest
    nonactor_groups = [ list(group) for group in consecutive_groups(nonactors) ]
    actor_groups = [ list(group) for group in consecutive_groups(actors) ]
    actor_groups.pop() # we don't want to count the target as a control, 
    # but it's not a nonactor either
    print('nonactor groups', nonactor_groups)
    print('actor groups', actor_groups)
    semifinal = small_op
    print('semifinal\n', semifinal)
    for index, group in enumerate(nonactor_groups):
        print('index', index)
        print('group', group)
        # what's the total dimensionality of this groups of middles?
        starting_control = group[0] - 1
        last_mid = group[-1]
        repeat = prod(circuit[starting_control:last_mid])
        print('Repeat ', repeat)
        
        # what's the total dimensionality of the controls before this group?
        # (both group lists are always the same size)
        # i = nonactor_groups.index(mid) # where are we in the list of nonactor groups?
        group = actor_groups[index] # find the corresponding group of actors
        # print(group)
        slice1 = slice(group[0], group[-1]+1) # here's the range in [ circuit ]
        # print('Slice')            
        # print(slice1)
        dims = circuit[slice1] # extract dims  
        # print('Dims ' )
        # print(dims)
        division = prod(dims) # multiply the dims together --> answer
        print('Division ', division)
        
        
        
        
        # apply redundant size increase
        semifinal = middle_tensor(semifinal, division, repeat)
        print('semifinal\n', semifinal)

        
    final = many_kron([front, semifinal, end])
    
    if Print == True:
        matshow(final)
        
    return final
                 
def D(i):
    # these operators diffuse from the |0> state evenly to all the other states
    # this structure reflects the game tree parent-children node connection 
    if i == 1:
        return array([[0,1],[1,0]])
    
    if i == 2:
        return array([[0,0, 1],
           [sqrt(1/2), sqrt(1/2), 0],
           [sqrt(1/2), -sqrt(1/2),0]])
    
    if i == 3:
        return array([[0,0,0,1],
            [sqrt(1/3), -sqrt(2/3), 0,0],
            [sqrt(1/3), sqrt(1/6), sqrt(1/2), 0],
            [sqrt(1/3), sqrt(1/6), -sqrt(1/2), 0]])

    if i == 4:
        return array([[0,0,0,0,1],
            [.5,.5,.5,.5,0],
            [.5,-.5,.5,-.5,0],
            [.5,.5,-.5,-.5,0],
            [.5,-.5,-.5,.5,0]])
    
def single(circuit, i, gate): # how can i accept one OR two inputs for index
    # integrates single-qudit operations into the whole circuit,
    # which amounts to much tensor producting with identity matrices
    
    before = int(prod(circuit[0:i]))
    after = int(prod(circuit[i+1:]))

    result = many_kron([identity(before), gate, identity(after)])
    return result

def recursive_enumeration(digits, encoding = [], attention = 1, first = True):
    # this function figures out how to increment all the digits together,
    # it is the main workhorse of encode_state()
    
    if first == True:
        string = ''
        for i in digits:
            string += str(i.val)
        encoding.append(string)
        
    # can you add 1 to the digit?
    if digits[-attention].val + 1 == digits[-attention].d:      # no
        digits[-attention].val = 0
        attention += 1
        if attention > len(digits):
            return encoding
        return recursive_enumeration(digits, encoding, attention, first = False)
    else:                                                   # yes
        digits[-attention].increment()
        string = ''
        for i in digits:
            string += str(i.val)
        encoding.append(string)
        attention = 1
        return recursive_enumeration(digits, encoding, attention, first = False)
     
class digit(): # used to encode states
    def __init__(self, d, val):
        self.d = d
        self.val = val

    def increment(self):
        if self.val + 1 == self.d:
            return 0
        else:
            self.val += 1
            return
      
def encode_state(circuit, Print=False):
    # produces a dictionary that counts 00001, 00010, etc
    # circuit is a list, not an object
    
    digit_order = []
    for i in circuit:
        digit_order.append(digit(i, 0))
        
    encoding = recursive_enumeration(digit_order)
    
    if Print == True:
        print(encoding)
    return encoding

class display_object(): # used to print out states with amplitude
    def __init__(self, amp, code):
        self.amp = amp
        self.code = code
        
def output_state(circuit, state, amplitude='no'):
    # this function prints out states formatted as xxx|yyy> + ...
    # xxx is the amplitude, yyy is the basis vector
    
    encoding = encode_state(circuit)
    objs = []
    size = prod(circuit)
    
    if amplitude is 'no':
        for i in range(size):
            if state[i] != 0:             # amp      state
                objs.append( display_object('', encoding[i]) )
    else:
        for i in range(size):
            if state[i] != 0:                     # amp                 state
                objs.append( display_object(str(state[i].round(3)), encoding[i]) )
        
    strings = []
    for i in objs:
        strings.append( i.amp + '|' + i.code  +'> ' )

    state_string = strings[0]
    
    for i in range(1,len(strings)):
        state_string += '+ ' + strings[i]
        
    print(state_string)
    
def diffuse(dim, swaps): # swaps is a list of the diffusees
    # this function produces a limited diffusion to certain states using the D() function
    
    d = D(len(swaps))
    result = identity(dim)
    swaps.insert(0,0)

    for row in range(len(swaps)):
        for col in range(len(swaps)):
            # print('Next')
            # print(row, col)
            # print(swaps[row], swaps[col])
            result[swaps[row], swaps[col]] = d[row, col]
    
    return result
    
def ttt_move(move, immute): # move - which move is it
    #  this function productes an operator to act on states with (move - 1), 
    #  bringing it to (move)
    
    m2 = []
    unchange = (1, 2, 3, 4)
    for i in a:
        diffuse_target = list(unchange)
        diffuse_target.remove(i)
        c = control(qc, 0, i)
        t = target(qc, 1, diffuse(5, diffuse_target))
        m2.append( create_control(qc, c, t) )
    
    result = m2[0]
    
    for i in range(1,4):
        result = m2[i] @ result
        
    return result

def create_list():
    # this function programmatically defines the operations for the double control state
    creation = {}
    unchange = (1, 2, 3, 4)
    for i in unchange:
        less = list(unchange)
        less.remove(i)
        nexter = tuple(less)
        for j in less:
            diffuse_target = list(nexter)            
            diffuse_target.remove(j)
            key = str(i) + str(j)
            creation[key] =  diffuse(5, diffuse_target)
            
    return creation
            
def create_control(control_size, target_size, instruct, control_encoding, Print=False):
    #        size : product of all control dimensions
    #        target_size : dimension of target action. Should correspond to operations in instruct
    #        instruct : dictionary that pairs control states and prescribed operations to the target
    #        control_encoding : similar to entire state encoding, except it's just for the control states
    #
    #     future: consider cases where control qubits are below targets
    
    trivial_partitions = [identity(target_size)] * control_size
    
    partitions = trivial_partitions
    for i in instruct:
        operation = instruct[i] # dict[key] = val
        placement = control_encoding.index(i)
        partitions[placement] = operation
        
    final = many_kadd(partitions)
    
    if Print == True:
        matshow(final)
        # fancy_print(mat, encode_state()) this is difficult atm
        title('Control operation', pad = 10)
    
    return final

def qudit_swap(circuit, i1, i2, Print=False):
    # swaps the states of two d=d qudits
    # 
    # Tony's notes have the associated pictures that
    # make the variable names make sense
    
    d_mid = int( prod(circuit[i1+1:i2]) )
    d_i = circuit[i1]
    d_tot = prod(circuit)
    j_combo = []
    small_row = zeros(d_tot)
    immute = tuple(small_row)
    jump_len = d_i * d_mid
    k_displacement = d_i
    for i in range(d_i):
        k_combo = []
        for j in range(d_mid):
            small_rows = []
            for k in range(d_i):
                current = list(immute)
                index = k*jump_len + k_displacement*j + i
                current[index] = 1
                small_rows.append(current)
            k_combo.append(vstack(small_rows))
        j_combo.append(vstack(k_combo))
    
        
    # bottom = flip(top) # flips h and v wise 
    # doing something like this can half the computation time
    
    swap = vstack(j_combo)
    
    if Print==True:
        fancy_print(swap, encode_state(circuit))
    return swap

def basis_add(circuit, i1, o):
    # this operation performs modular addition of i1 and o,
    # storing the result in o
    
    # new technique: don't dick around with patterns of the matrices,
    # just implement the truth table directly
    
    encoding = encode_state(circuit, Print=True )
    io = {}
    for in_ in encoding:
        out = ['0', '0' ]
        out[i1] = in_[i1]
        out[o] = str( (int(in_[i1]) + int(in_[o])) % circuit[o])
        
        io[in_] = ''.join(out) # this combines out from ['x', 'y'] to 'xy'
    
    matrix = truth_table_matrix(io, encoding)
    return matrix

def truth_table_matrix(dictionary, encoding):
    dim = len(dictionary)
    result = zeros([dim,dim])
    print(dictionary)
    for i in dictionary:
        i1 = encoding.index(i)
        i2 = encoding.index(dictionary[i])
        result[i2, i1] = 1
        
    return result

def init_state(circuit):
    dim = prod(circuit)
    state = zeros(dim)
    state[0] = 1
    return state

def fancy_print(mat, encoding):
    matshow(mat)
    dim = mat.shape[0]
    xticks(range(dim), encoding)
    yticks(range(dim), encoding)
    
def nim_move(circuit):
    # for now, I'll assume (3,3) (x) (3) format and generalize later
    
    # board controls history's first move
    instruct = {'2': diffuse(3,[1,2]), 
                '1': diffuse(3, [1]), 
                '0': identity(3) }
    
    board_c_hist = create_control(3, 3, instruct, ['0', '1', '2'], Print=True)
    
    return board_c_hist

def hadamard(dim):
    # produced hadamard matrix (dim * dim)
    # dim must be a power of two
    
    exponent = int( log(dim, 2) )
    
    h2 = array([[1,1],[1,-1]])
    result = 1
    for i in range(exponent):
        result = kron(result, h2)
    
    prefactor = sqrt(1/dim)
    return result * prefactor

if __name__ == "__main__":
    # nim = [3,3,3] # history (3,3) (x) board (3)
    # state = init_state(nim)
    # output_state(nim, state)
    
    # move = nim_move(nim)
    # output_state(nim, kron(move, identity(3)) @ state )
    
    integrate_test = [2] * 9
    swap = qudit_swap([2,2], 0, 1, Print=True)
    mini = create_control(4, 4, {'11': hadamard(4), '00': swap}, 
                    encode_state([2,2]), Print=True )
    mat = integrate(integrate_test, [2-2, 4-2], [7-2, 8-2], mini, Print=True)
    
    
