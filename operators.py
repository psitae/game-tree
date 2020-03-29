# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:19:25 2020

@author: T Dean
"""

import sys, timeit
from numpy import *
from matplotlib.pyplot import *
from math import log
from more_itertools import consecutive_groups

def gates(name, dim=2):
    two = identity(2)
    
    if name == 'not':
        return array([[0,1],[1,0]])
    
    if name == 'cnot':
        return k_add(two,gates('not'))
    
    if name == 'ccnot':
        return many_kadd([two,two,two,gates('not')])
    
    if name == 'hadamard':
        # produced hadamard matrix (dim * dim)
        # dim must be a power of two
        exponent = int( log(dim, 2) )
        h2 = array([[1,1],[1,-1]])
        result = 1
        for i in range(exponent):
            result = kron(result, h2)
        
        prefactor = sqrt(1/dim)
        return result * prefactor

    
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
        
    
def is_unitary(mat):
    size = mat.shape[0]
    candidate = mat.dot(mat.transpose())
    result = array_equiv(identity(size), candidate.round(3))
    # round exists to correct floating point errors, to do with square roots
    return result 

def copy(x): # expecting qubit dimension
    # aka fan-out 
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
                
def D(i):
    # these operators diffuse from the |0> state evenly to all the other states
    # this structure reflects the game tree parent-children node connection 
    # credit: https://quantumcomputing.stackexchange.com/
    # questions/10239/how-can-i-fill-a-unitary-knowing-only-its-first-column
    
    prefactor = 1/i
    diag_ = ones(i+1) * (i-1)
    diag_[0] = 0
    result = -ones([i+1, i+1])
    result[:,0] = sqrt(i)
    result[0,:] = sqrt(i)
    fill_diagonal(result, (i-1))
    result[0,0] = 0

    return prefactor * result

    
def single(circuit, i, gate): # how can i accept one OR two inputs for index
    # integrates single-qudit operations into the whole circuit,
    # which amounts to much tensor producting with identity matrices
    
    before = int(prod(circuit[0:i]))
    after = int(prod(circuit[i+1:]))

    result = many_kron([identity(before), gate, identity(after)])
    return result
    
def diffuse(dim, swaps, Print=False): # swaps is a list of the diffusees
    # this function produces a limited diffusion to certain states using the D() function
    swaps.sort()
    d = D(len(swaps))
    result = identity(dim)
    swaps.insert(0,0)

    for row in range(len(swaps)):
        for col in range(len(swaps)):
            # print('------------')
            # print(row, col)
            # print(swaps[row], swaps[col])
            result[swaps[row], swaps[col]] = d[row, col]
    
    if Print==True:
        matshow(result)
        title('Diffuse matrix')
    
    return result
    
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

def truth_table_matrix(orig, new, flip, Print=False):
    # transforms for original encoding to new encoding
    
    dim = len(orig)
    result = zeros([dim,dim])
    
    for i in flip:
        # print('examining ', i)
        i1 = flip.index(i)
        i2 = new.index(i)
        # print(i1, 'th goes to ', i2, 'th')
        result[i2, i1] = 1
        
    if Print == True:
        print('Printing from tt matrix()')
        fancy_print(result, orig, new, 'QuMit/QuNit swap_op')
        
    if is_unitary(result):
        return result
    else:
        print('Not a unitary matrix')
        
def fancy_print(mat, encoding, encoding2=None, title_='Stock title'):

    if encoding2 == None:
        encoding2 = encoding
        
    matshow(mat)
    dim = mat.shape[0]        
    xticks(range(dim), encoding, rotation=90) # input
    yticks(range(dim), encoding2) # output
    title(title_, pad = 30)
    
def integrate(circuit, control, target, small_op, Print=False):
    # circuit : list of dimensions of all qudits
    # control & target : list of indices
    # input: small_op is the operation, acting on a list of control 
    # and target indices
    # integrate() creates the uber matrix for the quantum circuit

    print('\nIntegrating...')

    # front and end
    front_iterable = range(0, control[0])
    front_slice = slice(0, control[0])
    front_size = prod(circuit[front_slice])
    front = identity(int(front_size))
    end_iterable = range(target[-1]+1,len(circuit))
    end_slice = slice(target[-1]+1, -1)
    end_size = prod(circuit[target[-1]+1:])
    end = identity(int(end_size))
    # print('Front size', front_size)
    # print('Front\n', front)
    # print('End size', end_size)
    # print('End\n', end) 
    
    # distinguish meaningful from redundant instructions
    # within the target range
    actors = list( hstack( [control, target] ))
    nonactors = list(range(len(circuit)))
    [ nonactors.remove(i) for i in actors ] # do this part better later
    [ nonactors.remove(i) for i in front_iterable ]
    [ nonactors.remove(i) for i in end_iterable ]
    # what's left should only be middle nonactors
    # print('actors', actors)
    # print('nonactors', nonactors)
    
    # check to see if integration is necessary
    if len(nonactors) == 0 and front_size-1 == 0 and end_size-1 == 0:
        print('No integration necessary.')
        return small_op
    # else:
        # role_call(circuit, control, target)
    
    # group indices of the circuit into consecutive sequences
    # consec. seq. are already tensored together, just figure out the rest
    nonactor_groups = [ list(group) for group in consecutive_groups(nonactors) ]
    actor_groups = [ list(group) for group in consecutive_groups(actors) ]
    actor_groups.pop() 
    # we don't want to count the targets as controls, 
    # but it's not a nonactor either
    # print('nonactor groups', nonactor_groups)
    # print('actor groups', actor_groups)    

    semifinal = small_op
    # print('small\n', small_op)
    for index_, group in enumerate(nonactor_groups):
        # what's the total dimensionality of this groups of middles?
        starting_control = group[0] - 1
        last_mid = group[-1]
        repeat = prod(circuit[starting_control:last_mid])
        # print('Repeat ', repeat)
        
        # what's the total dimensionality of the controls before this group?
        # (both group lists are always the same size)

        group = actor_groups[index_] # find the corresponding group of actors
        slice1 = slice(group[0], group[-1]+1) # here's the range in [ circuit ]
        dims = circuit[slice1] # extract dims  
        division = prod(dims) # multiply the dims together --> answer
        
        # apply redundant size increase
        semifinal = middle_tensor(semifinal, division, repeat)
        # print('semifinal\n', semifinal)
       
    final = many_kron([front, semifinal, end])
    
    if Print == True:
        matshow(final)
        
    return final

def backwards_control_workaround(circuit, control, target, instruct, Print, title_):
    print('\nDiscovered inverted control.')
    # role_call(circuit, control, target)
    
    # find proper permutation, as create_control() expects the targets at the end
    index_ = list(range(len(circuit))) 
    front = index_[:target[0]]
    middle = index_[target[0]:control[-1]+1]
    end = index_[control[-1]+1:]
    
    print('front\t', front)
    print('middle\t', middle)
    print('end \t', end)
    

    # keep the order of the targets, and put them at the end
    # of the new order
    shifted_target = []
    [ shifted_target.insert(0,t) for t in reversed(target) ]
    
    not_target = list(middle)
    [ not_target.remove(t) for t in target ]
    print('target(s)\t', target)
    print('not targets\t', not_target)    
    
    reorder = front + not_target + shifted_target + end
    print('reorder \t', reorder)    

    # perm is a permutation using standard group theoretic notation
    # perm = empty_like(middle)    
    perm = [ reorder.index(i) for i in range(len(reorder)) ]  
    print('perm \t', perm)

    swap_op, new_circuit, new_c, new_t = swap(perm, circuit, control, target)

    if max(new_c) > max(new_t):
        print('Error: swap did not create a good create_control() call')
        return
    
    print('\nCreating control from workaround()')
    need_swap = create_control(new_circuit, new_c, new_t, instruct, Print)
    
    # print('need_swap shape\t', need_swap.shape)
    # print('swap op shape\t', swap_op.shape)
    # print('proper size:\t', prod(circuit))
    # faulty need_swap count .
    # faulty swap_op   count 
    print('\nSwapping this control back to')
    role_call(circuit, control, target)
    
    swapped = swap_op.transpose() @ need_swap @ swap_op 
    # operators need modification from both sides
    # this might need some review M^T X M or M X M^T ?
    
    if Print == True:
        print('Printing from workaround()')
        csts = list(control) + list(target)
        csts.sort()
        dims = [ circuit[i] for i in csts ] 
        fancy_print(need_swap, encode_state(new_circuit))
        title('needs swapping', pad = 30)
        fancy_print(swapped, encode_state(circuit) ) 
        title('swapped', pad = 30)
        
    return swapped
        
def create_control(circuit, control, target, instruct, 
                   Print=False, title_='Control operation'):
    # circuit, control, target are normal parameters
    # instruct : dictionary that pairs control states and prescribed operations to the target
    # this is defined programmatically 
    
    
    # user friendly feature: if control and target are solo, put them into a list
    if type(control) == int:       
        control = [ control ]
    if type(target) == int:
        target = [ target ]

    #
    control.sort()
    target.sort()
    
    # calculate some stuff
    control_dims = [ circuit[i] for i in control ]
    control_size = prod(control_dims)
    target_dims = [ circuit[i] for i in target ]
    target_size = prod(target_dims)
    actors = len(control) + len(target)    
    
    # do the instruction give an operation with the proper dims?
    for key in instruct:
        if instruct[key].shape[0] != target_size:
            print('Improper instructions')
            return
        
    # consider cases where control qudits are below targets
    if max(control) > max(target):
        return backwards_control_workaround(
            circuit, control, target, instruct, Print, title_)

    # enough checking. We're ready to start
    print('\nCreating control.')
    role_call(circuit, control, target)

    # begin contstructing operator as all identity operations on the target
    trivial_partitions = [identity(target_size)] * control_size
    
    # replace certain spaces with instructions
    controls = [ circuit[i] for i in control ]
    control_encoding = encode_state(controls)
    # print('controls', controls)
    # print('control encoding', control_encoding) 
    partitions = trivial_partitions
    for i in instruct:
        operation = instruct[i] # dict[key] = val
        placement = control_encoding.index(i)
        partitions[placement] = operation
        
    # combine with direct sum
    semifinal = many_kadd(partitions)
    # add in all other qudits with tensor products in the right places
    final = integrate(circuit, control, target, semifinal)
    
    if Print == True:
        print('Printing from create_control()')
        csts = control + target
        csts.sort()
        dims = [ circuit[i] for i in csts ] 
        fancy_print(semifinal, encode_state(dims) ) 
        title(title_, pad = 30)
        
        # if not array_equiv(final, semifinal):
        #     fancy_print(final, encode_state(circuit) )
        #     title('final', pad = 30)
        
    return final

def swap(perm, circuit, control, target):
    print('\nComputing swaps.')
    immute = tuple(perm)
    in_order = array(immute)
    in_order.sort()
    
    print('control\t', control)
    print('target\t', target)
    # rearrange control, target, circuit
    # full_perm = list(range(0,perm[-1])) + list(perm)
    new_control = []
    new_target = []
    [ new_control.append(perm[i]) for i in control ]
    [ new_target.append(perm[i]) for i in target ]
  
    new_circuit = array(circuit)    
    new_circuit[perm] = new_circuit[in_order]
    
    print('new_control\t', new_control)
    print('new_target\t', new_target)      
    print('old_circuit\t', array(circuit))    
    print('new_circuit\t', new_circuit)
    
    orig_encoding = encode_state(circuit)
    new_encoding = encode_state(new_circuit)
    flip = []
    for code in orig_encoding:
        str_array = array(list(code))
        str_array[perm] = str_array[in_order]
        flipped = ''.join(str_array)
        # print('--------------')
        # print('code \t', str(code))        
        # print('flipped\t', flipped)
        flip.append(flipped) 

    swap_op = truth_table_matrix(orig_encoding, new_encoding, flip, Print=False)
    # print('orig\n', orig_encoding)
    # print('new\n', new_encoding)
    
    return swap_op, new_circuit, new_control, new_target

def role_call(circuit, control, target):
    roles = array(['.']*len(circuit))
    roles[control] = 'c'
    roles[target] = 't'
    
    p_circ = [ f"{i} " for i in circuit ]
    p_i = [ f"{i} " for i in array(range(len(circuit))) ]
    p_roles = [ f"{i} " for i in roles ]
    print('Circuit\t|', *array(p_circ))
    print('Index\t|', *array(p_i))
    print('Roles\t|', *array(p_roles))

def encode_state(circuit, Print=False):
    l = len(circuit)
    
    start_memory = get_resident_set_size()
    
    state = [0]*l
    encoding = ['0'*l]
    last_state = ''.join([str(i-1) for i in circuit ])
    while encoding[-1] != last_state :
        look = -1
        
        # print('\n' + str(len(encoding)) + 'th encoding')
        # print('Input:\t', ''.join([ str(i) for i in state ]) )
        # print('Examining ', state[look], ' at position ' + str(look) )
        
        incrementable = False
        while not incrementable:   
            # this loop looks at each digit and changes it to the next
            # state. After that, it breaks, calling the else: marked (.)
            
            # print('Examining', state[look], ' at position ' + str(look) )
            incrementable = state[look] + 1 < circuit[look]
            if incrementable: 
                state[look] += 1
                # print('Incrementing to ' + str(state[look]) )
            else:
                state[look] = 0
                # print('Digit ' + str(look) + ' now set to 0')
                look -= 1
                
            # print('Current state: ' + ''.join([ str(i) for i in state ]))
       
        else: # (.)
            # record newest edition
            # print('Add state: ' + ''.join([ str(i) for i in state ]) )
            encoding.append(''.join([ str(i) for i in state ]))
        
    if Print == True:
        print(encoding)
    
    mem_usage = get_resident_set_size() - start_memory
    return mem_usage


from pathlib import Path
from resource import getpagesize



def get_resident_set_size() -> int:
    PAGESIZE = getpagesize()
    PATH = Path('/proc/self/statm')
    """Return the current resident set size in bytes."""
    # statm columns are: size resident shared text lib data dt
    statm = PATH.read_text()
    fields = statm.split()
    return int(fields[1]) * PAGESIZE 

if __name__ == "__main__":
    fancy_print(gates('cnot'), encode_state([2,2]))
    