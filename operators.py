# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:19:25 2020

@author: T Dean
"""

from numpy import *
from matplotlib.pyplot import *
import itertools

def k_add(a,b): # expects two square matrices 
    # I can't believe I had to program this. It is a standard direct sum operation.
    # it will create a block diagonal matrix with the two inputs
    
    tr = zeros([a.shape[0],b.shape[1]])
    bl = tr.transpose()
    top = concatenate([a, tr], axis=1)
    bottom = concatenate([bl, b], axis=1)
    both = concatenate([top, bottom])
    return both

def many_ksum(x): # expecting a list of matrices
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
        
def create_swap(dim, a=0, b=1): # swaps ath and bth element in a (dim x dim) matrix
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
    # ONLY IF the second one starts out at |0>
    
    result = identity(x**2)
    for i in range(1,x):
        result[ i*x, i*x ] = 0
        result[ i*x+i, i*x+i ] = 0
        result[ i*x+i, i*x] = 1
        result[ i*x, i*x + i ] = 1
        
    return result
          
def integrate(quantum_circuit, control, target):
    # this doesn't work right now
    # eventually, it might be combined with single()
    

    if control.index < target.index: # forwards control
        # front
        front_size = prod(quantum_circuit.dim[0:control.index])
        front = identity(int(front_size))

        # middle
        middle_size = prod(quantum_circuit.dim[control.index+1:target.index])
        middle = identity(int(middle_size))
        # print('Middle: ' + str(middle_size) )
        
        # end
        end_size = prod(quantum_circuit.dim[target.index:-1])
        end = identity(int(end_size))
 
        # partitions
        repeated_swaps = kron(middle, target.gate) 
        partition_size = int(middle_size) * target.size
        
        # print('Partition size: ' + str(partition_size) )
        
        trivial_partitions = [identity(partition_size)] * quantum_circuit.dim[control.index]
        # print('List length: ' + str(len(trivial_partitions)))
        # print('Entries:')
        # print(trivial_partitions[0])
        non_trivial_partition = control.control_state
        
        # insert non trivial partition(s)
        partitions = trivial_partitions
        partitions[non_trivial_partition] = repeated_swaps
        # print(partitions)
        
        # combine partitions with direct sum
        
        semifinal = many_ksum(partitions)
        
        final = many_kron([front, semifinal, end])

        return final
                 
class quantum_circuit(): # expecting a list of dimensions
    # object containing a list of all the dimensions of the qudits, primarily
    def __init__(self, qubits):
        self.size = prod(qubits)
        self.dim = qubits
        
class control():
    # is this even still used? I don't think its very useful
    def __init__(self, qc, i, state):
        self.index = i
        self.control_state = state
        self.size = qc.dim[i]
        
class target():
    # same as control object
    def __init__(self, qc, i, gate):
        self.index = i # (i+1)th qubit down
        self.gate = gate # control action
        self.size = qc.dim[i]

def D(i):
    # these operators diffuse from the |0> state evenly to all the other states
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
    
    before = int(prod(circuit.dim[0:i]))
    after = int(prod(circuit.dim[i+1:]))

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
    
    if amplitude is 'no':
        for i in range(circuit.size):
            if state[i] != 0:             # amp      state
                objs.append( display_object('', encoding[i]) )
    else:
        for i in range(circuit.size):
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
    print(d)
    # print('Entering Diffuse...')
    # print('dim')
    # print(dim)
    # print('swaps')
    # print(swaps)
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
    #  bring it to (move)
    
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
            
def create_control(size, target_size, instruct, control_encoding, Print=False):
    #        size : product of all control dimensions
    #        target_size : dimension of target action. Should correspond to operations in instruct
    #        instruct : dictionary that pairs control states and prescribed operations to the target
    #        control_encoding : similar to entire state encoding, except it's just for the control states
    #
    #     future: consider cases where control qubits are below targets
    
    trivial_partitions = [identity(target.size)] * size
    partitions = trivial_partitions

    for i in instruct:
        operation = instruct[i] # dict[key] = val
        placement = control_encoding.index(i)
        partitions[placement] = operation
        

    final = many_ksum(partitions)
    
    if Print == True:
        matshow(final)
    
    return final
    
if __name__ == "__main__":
    qc = quantum_circuit([5,5,5,5])
    
    control_encoding = encode_state([5,5], Print=True)
    create_control(25, t, create_list(), control_encoding, Print=True)
    # state = []
    # state.append( zeros(qc.size)  )
    # state[0][0] = 1

    # print('Move 0')
    # output_state(qc, state[-1])
    # move_one = single(qc, 0, D(4) )
    
    # print('Move 1')    
    # state.append(move_one @ state[-1] )
    # output_state(qc, state[-1])
    
    # print('Move 2')
    # output_state(qc, move_2(qc) @ state[1] )
     
    
    # double_control(qc, [5, 5], target(qc, 2, ''), m3_pairs)
    # matshow(diffuse(5,[3,4]))
    

    