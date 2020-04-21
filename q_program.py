#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:40:03 2020

@author: tony

"""


import operators as ops
import sympy as sp
from numpy import *
import matplotlib.pyplot as plt
import copy
import collections

null  = '\u2205'
check = '\u2714'

def indexer(lst, *indices):
    return (lst[i] for i in indices)

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [ a for i in x for a in flatten(i) ]
    else:
        return [x]

class display_object:
    """
    used to print out state-kets w/wo amplitude
    """
    def __init__(self, amp, code):
        self.amp = amp
        self.code = '|' + code + '>'
        
class instruction:
    """
    This object combines gates with a list of qudits it applies to, all the
    information needed for one quantum gate on the whole circuit
    """
    def __init__(self, num, note, *pairs):
        self.gate_set = []
        for gate, indx in pairs:
            self.gate_set.append( (gate, indx) )
        
        self.note = note
        self.num  = str(num)
    
    def integrate(self, state, halt=None):
        """
        Each instruct object can have, in general, a repetition of a gate to
        many independent qubits. E.g., all qubits getting a hadamard gate at
        the same time.
        """
        for gate, indx in self.gate_set:
            collect = {}
            for full_basis, amp in state.items():
                sub_basis = tuple([ full_basis[i] for i in indx ])
                fulls = collect.get(sub_basis, {})
                fulls[full_basis] = amp
                collect[ sub_basis ] = fulls
            
            # collect is an association between sub-basis extractions and
            # their full-basis counterparts, each entry is as:
            # { sub-basis : fulls }
            # where fulls = { orig_basis_0 : orig_amp_0, 
            #                 orig_basis_1  : orig_amp_1, ... }
            
            # construct new state by iterating through each basis in state
            # and integrating the result of gate.apply() to each
            state = {}
            for sub_basis, fulls in collect.items():
                # make a dummy call to gate.apply()
                dummy_out = gate.apply({ sub_basis : 1 })
                for full_basis, amp in fulls.items():
                    full_basis = list(full_basis)
                    for dummy_basis, dummy_amp in dummy_out.items():
                        for i, jay in zip(indx, range(len(indx))):
                            full_basis[i] = dummy_basis[jay]
                        add_to_state = tuple(full_basis)
                        # add amplitude to value in dict if it exists
                        present = state.get(add_to_state, 0)
                        simp = sp.simplify(amp * dummy_amp)
                        state[add_to_state] =  present + simp
            # remove zero amps if they exist
            [ state.pop(basis) for basis, amp in list(state.items()) if amp == 0 ]
        
        
        return state
    
class mat_instruction(instruction):
    """
    Adds matrix functionality to the instructions class.
    The intended use of this class is for human-readably small cases,
    to check if the normal sped-up methods are giving the right answers
    """
    def __init__(self, num, note, mat, uni, *pairs):
        instruction.__init__(self, num, note, *pairs)
        self.mat = mat
        self.is_unitary = uni
    
    def integrate(self, qc):
        # normal integration methods with truth tables
        qc.state = super().integrate(qc.state)
        
        # displax full matrix visually, unitary or not
        ax = plt.matshow(self.mat.astype(float), cmap='rainbow')
        cbar = plt.colorbar( ticks = [] )
        ax.set_clim(-1, 1)
        plt.title('Gate: ' + self.num + '\nFull Matrix operation', pad=20)
        
        if self.is_unitary:
            # perform update of column
            print('\nMatrix implementation checks:')
            print('Unitary Operation:\t' + check)
            unsimp = self.mat @ qc.column                    #  ✔
            qc.column = [ sp.simplify(i) for i in unsimp ] 
            
            # normalized state
            norm = inner(qc.column, qc.column)
            if norm == 1:
                print('Normalized state: \t' + check)
            else:
                print('State not normalized. Halting')
                qc.halt = True
            
            # display gate as matrix and truth table
            title_ = 'Gate ' + self.num + ':\n' + self.gate.name
            ops.printout(self.gate, title_ )
            
            # compare state 
            qc.col2state()
            
            
        else:
            print(qc.error_dividor, 'Non-unitary operation')
            qc.halt = True
        

class quantum_circuit:
    """
    The main object in the algorithm.
    """
    # class variables
    
    dividor = '\n' + '='*50
    error_dividor = '\n' + 'X'*50 + '\n'
    
    def __init__(self, dims, divisions = [], name=False, show_amp=False):
        """
        Divisions is a list of the sizes of each division, it will be cum-added
        """
        # printing settings
        if name: 
            self.name = '\"' + name + '\"'
        else:
            self.name = r'"A quantum circuit"'
        self.show_amp = show_amp
        self.encodings = []
        self.instruct_note = None
        
        # calculation attributes
        self.dims = array(dims)
        self.size = prod(dims)
        self.length = len(dims)
        self.instruct_set = []
        self.depth = 0
        self.divisions = list(cumsum(divisions))
        if len(self.divisions) > 1: self.divisions.pop()
        self.halt = False
        
        # initialize state
        self.state = { (0,) * self.length : 1 }
        
        print('Creating quantum circuit ' + self.name + ':')
        print(self.dims, ' with size ' + str(self.size) )
    
    def write_state(self, *state):
        """
        Accepts strings like '0101' and converts them to dict objects.
        State should be a dict object with { basis : amplitude } pairs.
        """
        # clear state reversibly
        temp = self.state
        self.state = {}
        
        norm = 1/sp.sqrt(len(state))
        for st in state:
            if len(st) != self.length:
                print('Improper state assignment')
                self.halt = True
                self.state = temp
                return
            
            basis = tuple([ int(num) for num in st ])
            self.state[basis] =  norm 
            
    def order_state(self):
        """
        Returns a list of ordered states,
        ( basis, amp )
        """
        lets_sort =  [ [basis, amp] for basis, amp in self.state.items() ]
        for entry in lets_sort:
            entry.append( ops.get_location(self.dims, entry[0]) )
        
        lets_sort.sort(key=lambda entry: entry[2])
        
        return [ (a, b) for a, b, _ in lets_sort ]
    
    def special_encoding(self, scheme, *indx):
        if scheme == 'null':
            swap_dict =  { 0: null, 1:'F', 2:'T'}
        elif scheme == 'TF':
            swap_dict = { 0: 'F', 1: 'T'}
        elif isinstance(scheme, dict):
            swap_dict = scheme
        else:
            print('Scheme error')
            self.halt = True
            return
        
        [ self.encodings.append( (i, swap_dict) ) for i in indx ]
    
    def index_aide(self):
        """
        Prints dims and related indices with dividors
        """
        fmt_dims = list(self.dims)
        fmt_indx = list(range(self.length))
        for i, j in enumerate(self.divisions):
            fmt_indx.insert(i+j, ';')
            fmt_dims.insert(i+j, ';')
        
        fmt_dims = [str(i) for i in fmt_dims ]
        fmt_indx = [str(i) for i in fmt_indx ]
        print('\n\nIndex Aide' + '\n---------')
        print('Dims:\t' + ' '.join(fmt_dims))
        print('Indx:\t' + ' '.join(fmt_indx))
    
    def printout(self):
        disp_objs = []
        for basis, amp in self.order_state():
            basis =  list(basis)
            # modify string with special encodings
            for indx, swap_dict in self.encodings:
                basis[indx] = swap_dict.get(basis[indx])
            
            # insert ; for divisioning modules
            [ basis.insert(i+j,';') for i,j in enumerate(self.divisions) ]

            # form a string
            state_str =  ''.join([str(i) for i in basis])
            
            if self.show_amp:
                shown_amp = str(amp)
            else: 
                shown_amp = ''
            disp_objs.append( display_object(shown_amp, state_str ) )
    
        output_list = []
        for base in disp_objs:
            output_list.append(base.amp + base.code)
        
        # intersperse with '+'
        together = [' + '] * (len(disp_objs) * 2 - 1)
        together[0::2] = output_list
        output_str = ''.join([s for s in together])
        
        print(output_str)
        
    def run(self):
        """
        Checks for the proper conditions, then applies each gate in order,
        printing each state as the program progresses.
        """
        
        # Are we really ready to run the circuit?
        if self.halt:
            print(quantum_circuit.error_dividor)
            print('\nHalted')
            return
        
        if self.depth == 0:
            print(quantum_circuit.error_dividor)
            print('\nNo operations to apply')
            return
        
        print(quantum_circuit.dividor)
        print('Running quantum circuit ' + self.name + ':\n')
        print('Initial state:')
        self.printout()
        
        for instruct in self.instruct_set:
            if self.halt: break
            print('\n' + '_'*50+ '\n')
            print('Instruction ' + instruct.num + '\n' + '-'*20)
            if instruct.note is not None: print('Note: '  + instruct.note)
            
            for (gate, indx) in instruct.gate_set:
                print( gate.name + ' acting on qudit(s)', indx)

            # a little bit of oop ugliness
            if isinstance(self, mat_quantum_circuit):
                # mat_instruct gets whole qc object passed
                instruct.integrate(self)
            else:
                self.state = instruct.integrate(self.state)
            
            print('\nState ' + instruct.num + ':')
            self.printout()
        
        if not self.halt:
            print(quantum_circuit.dividor)
            print('\nFinal State:')
            self.printout()
        
    def instruct_notes(self, note):
        """
        Adds annotations to each operation. self.instruct_note is reset to 
        None after each add_instruct() call, making the default no none at all
        """
        self.instruct_note = note
        
    def add_instruct(self, gate, *indx):
        if self.halt: return
        
        # check validity of indices. Are any repeated?
        if len(flatten(indx)) != len(set(flatten(indx))):
            print(quantum_circuit.error_dividor)
            print('Invalid gate copy operation:', gate.name)
            print(self.instruct_note)
            self.halt = True
            return
        
        pairs = []
        for i in indx:
            # check validity of each gate
            self.validate_instruct(gate, i)
            if self.halt: return
            
            pairs.append( (gate, i) )
        
        self.depth += 1
        instruct = instruction(self.depth, self.instruct_note, *pairs)
        self.instruct_note = None
        
        self.instruct_set.append(instruct)
        
    def validate_instruct(self, gate, indx):
        """
        Checks to see if the gate + indx jives with the circuit dimensions
        Return bool determining quantum_circuit.halt
        """
        # too many indices for gate dims?
        dim_check = ( 'Lengths', len(gate.dims), len(indx) )
        if dim_check[1] == dim_check[2]:
            # all indices point to gate dims that match the quantum circuit?
            dim_check = [ (self.dims[i], gate.dims[jay])
                         for jay, i in enumerate(indx)]
            if all([ a==b for (a,b) in dim_check]):
                self.halt = False
                return
            dim_check.insert(0, 'Relative dimensions')
        
        print(quantum_circuit.error_dividor)
        print('Invalid instruct:', gate.name)
        print(self.instruct_note)
        print(dim_check)
        self.halt = True
    
class mat_quantum_circuit(quantum_circuit):
    
    def __init__(self, dims, divisions = [], name=False, show_amp=False):
        print('Creating matrix quantum circuit\n')
        quantum_circuit.__init__(self, dims, divisions, name, show_amp)
        self.init_column()
    
    def init_column(self):
        col = zeros(self.size, object)
        col[0] = 1
        self.column = col
        
    def write_state(self, *state, Print=False):
        """
        Call write_state() in the parent function as normal.
        Also prepares the matrix state by finding the corresponding 
        state vector and store it in self.column
        """
        
        super(mat_quantum_circuit, self).write_state(*state)
        
        # clear state
        self.column = zeros(self.size, object)
        
        # assuming equal superposition of all input state strings
        norm = 1/sp.sqrt(len(state))
        
        state_locations = []
        for st in state:
            
            # check validity of input
            if len(st) > self.length:
                print('\nState too long to specify')
                self.halt = True
                return
            
            state_locations.append(ops.get_location(self.dims, st))
            self.column[state_locations] = norm
            
    def add_instruct(self, gate, *indx):
        """
        *indx is 1 or more tuples the gate applies to
        Creates a matrix equivalent to the (gate + index)
        Calls add_instruct() in the parent function, passing gate, indx and mat
        """
        super().add_instruct(gate, *indx)
        
        # do we need matrix integration?
        if len(*indx) == len(self.dims):
            if gate.mat is None: gate.mat = ops.tt2mat(gate.tt)
            mat = gate.mat
        else:
            # make full truth table for entire circuit and all repeated gates
            print('THIS IS NOT READY TO RUN YET')
            mat = new_gate.matrix_integration(self.dims, i)
            
        uni = ops.is_unitary(mat)
        self.instruct_set[-1].set_matrix(mat, uni)
        
        print('Finished loading gate ' + str(self.depth) + ': ' + gate.name)
    
    def col2state(self):
        """
        Convert the column object into a normal state object and compare
        with self.state
        """

        col_state = { tuple(ops.get_encoding(self.dims, loc)) : amp 
                for loc, amp in enumerate(self.column) if amp != 0 }
        
        if self.state == col_state:
            print('State argreement: \t' + check)
        else:
            print(quantum_circuit.error_dividor)
            print('Matrix state does not agree. Halting.')
            temp = self.state
            self.state = col_state
            print('Matrix state:')
            self.printout()
            self.state = temp
            self.halt = True

def test_fan_out():
    qc = mat_quantum_circuit([2,2,2], divisions = [], name='Test fan out')
    qc.write_state('010', '000')
    
    gate = ops.fan_out(2, 0, 1)
    instruct = instruction(gate, (1, 2))
    qc.add_instruct(instruct)
    qc.run(Print=True, show_amp = True)

def test_diffusion(test=2):
    if test == 1:
        qc = mat_quantum_circuit([4,4], show_amp=True)
        qc.write_state('10', '20', '30')
        
        gate1 = ops.goto_state(4, send=1)
        qc.add_instruct(gate1, [0])
        
        gate2 = ops.branch(4)
        qc.add_instruct(gate2, [1])
        
        qc.run()
    
    if test == 2:
        qc = mat_quantum_circuit([3], show_amp=True)
        qc.special_encoding('null', 0)
        
        b = ops.branch(3)
        qc.add_instruct(b)
        
        gt2 =ops.goto_state(3, 2)
        qc.add_instruct(gt2)
        
        
        qc.run()
        
def test_logic():
    qc = mat_quantum_circuit([3,3,2,2,2], divisions=[2,2,1], name='Test AND')
    
    b = ops.branch(3)
    qc.add_instruct( b, [0] )
    qc.add_instruct( b, [1] )
    
    c32 = ops.copy32(0,1)
    qc.add_instruct( c32, [0, 2] )
    qc.add_instruct( c32, [1, 3] )
    
    qc2 = copy.deepcopy(qc)
    qc2.name = 'Test OR'
    
    AND = ops.AND([0,1],2)
    qc.add_instruct( AND, [2, 3, 4] )
    
    OR = ops.OR([0,1], 2)
    qc2.add_instruct( OR, [2, 3, 4] )
    
    qc.run()
    print('\n\n\n')
    print('*'*80)
    print('\n\n\n')
    qc2.run()

def test_matrix_check():
    qc = mat_quantum_circuit([2,3,2], name='Test matrix check')
    
    g1 = ops.branch(3)
    qc.add_instruct( g1, [1] )
    g2 = ops.goto_state(3)
    qc.add_instruct( g2, [1] )
    
    qc.run()
    
def test_control_ops():
    qc = mat_quantum_circuit([2,2], name='Test control operations', show_amp=True)
    qc.write_state('00', '10')
    
    # go1 = ops.goto_state(3, send=1)
    # go2 = ops.goto_state(3, send=2)
    # directions = { (0,) : go1, (1,) : go2 }
    # directions = { (1,1,1) : go1 }
    # ctl_go = ops.create_control([2,3], 0, 1, directions)
    # qc.add_instruct(ctl_go, [0,2])
    qc.add_instruct(ops.gates('not'), [1])
    
    directions = { (0,): ops.gates('hadamard'),
                  (1,) : ops.gates('hadamard', reverse=True) }
    
    ctrl = ops.create_control([2,2], 1, 0, directions)
    qc.add_instruct(ctrl)
    qc.run()
    
def test_idea():
    
    qc = quantum_circuit([3,3,2,2,2,2,2,2], divisions = [2,3,3],
        name='Test idea', show_amp=False)
    qc.special_encoding('null', 0,1)
    qc.special_encoding('TF', 2,3,4)
    
    b = ops.branch(3)
    qc.add_instruct( b , [0])
    qc.add_instruct( b , [1])
    
    c32 = ops.copy32(0,1)
    n32 = ops.not32(0,1)
    
    qc.add_instruct( c32, [0, 2])
    qc.add_instruct( c32, [1, 4])
    qc.add_instruct( n32, [0, 3])
    
    fan = ops.fan_out(2, 0, 1)
    OR = ops.OR([0,1], 2)
    AND = ops.AND([0,1], 2)
    qc.add_instruct(fan, [2, 5] )
    qc.add_instruct(OR, [3, 4, 6] )
    qc.add_instruct(AND, [5, 6, 7] )
    # now everything is evaluated
    
    # reset q5 and q6, use as goto control
    qc.add_instruct( fan, [2, 5] )
    qc.add_instruct(OR, [3, 4, 6] )
    
    # target |.F>
    directions = { (1,): ops.gates('not') }
    ctrl = ops.create_control([2,3], [1], 0, directions)
    flip_branch = qc.add_instruct(ctrl, [1,5,7] )

    qc.run()
    qc.index_aide()
    
def test_grover():
    qc = mat_quantum_circuit([4,2], divisions=[1,1], name='Test grover', show_amp=True)
    # qc.special_encoding( {0:'(FF)', 1:'(FT)', 2:'(TF)', 3:'(TT)'}, 0)
    
    had4 = ops.gates('hadamard', 4)
    qc.add_instruct(had4, [0])
    
    AND = ops.AND()
    AND.change_dims([4,2])
    qc.add_instruct(AND)
    
    # directions = { (1,): ops.gates('flip') }
    cond_flip = ops.gates('flip')
    qc.add_instruct(cond_flip, [1])
    
    qc.add_instruct(AND)
    
    # qc.add_instruct(ops.one_shot_grover(),[0])
    # qc.add_instruct(ops.one_shot_grover(),[0])
    
    add4 = ops.arith([4], 0, 0)
    qc.add_instruct(add4, [0])
    qc.run()
    
def test_goto():
    qc = mat_quantum_circuit([3,3,2,2], [2,2], 'Test goto', True)
    qc.special_encoding('null', 0, 1)
    qc.write_state('1100', '1200', '2100', '2210')
    
    direct1 = { (2,) : ops.gates('not') }
    tx = ops.create_control([3,2], 0, 1, direct1)
    direct2 = { (1,) : tx }
    info_transfer = ops.create_control([3,2,2], 1, [0, 2], direct2)
    # qc.add_instruct(tx, [0,3])
    qc.add_instruct(info_transfer, [1,2,3])
    
    qc.run()
    
def test_idea2():
    qc = quantum_circuit([3,3,2,2,2,2], [2,2,2], 'Test idea 2', True)
    qc.special_encoding('null', 0, 1)
    qc.special_encoding('TF', 2, 3)
    
    #1
    qc.instruct_notes('branch @ 0')
    qc.add_instruct( ops.branch(3), [0])
    
    #2
    qc.instruct_notes('|null> - |T> swap @ 1')
    swap = ops.swap(3, 0, 2) 
    qc.add_instruct( swap, [1])
    
    #3
    qc.instruct_notes('Copy node info to ancillary memory')
    c32 = ops.copy32()
    apply_to = (0,2) , (1,3)
    qc.add_instruct( ops.copy32(), *apply_to )
    
    #4
    qc.instruct_notes('evaluate node @ val')
    qc.add_instruct( ops.SAME(), [2,3,4])
    
    #5
    qc.instruct_notes('copy val to continue')
    qc.add_instruct( ops.gates('cnot'), [4,5] )
    
    #6
    qc.instruct_notes('CONTINUE CONTROL: load other node')
    # double_swap = ops.gate_concat(ops.swap(3, 2, 1), ops.gates('not'))
    dirs5 = { (0,) : ops.gates('not') }
    ctrl5 = ops.create_control([2,2], 0, 1, dirs5 )
    qc.add_instruct( ctrl5, [5,3] )
    
    #7
    qc.instruct_notes('CONTINUE CONTROL: evaluate node')
    dirs6 = { (0,) : ops.SAME() }
    ctrl6 = ops.create_control([2,2,2,2], 0, [1,2,3], dirs6)
    qc.add_instruct( ctrl6, [5,2,3,4])
    
    #8
    qc.instruct_notes('Uncopy continue control')
    qc.add_instruct( ops.gates('cnot'), [4,5] )
    
    #9
    qc.instruct_notes('PREPARE MERGE: unevaluate')
    qc.add_instruct( ops.SAME(), [2,3,4] )
    
    
    # qc.instruct_notes('Move amplitudes up one node on the tree')
    # qc.add_instruct( swap, [1] )
    
    # #9
    # qc.instruct_notes('Unbranch @ 0')
    # qc.add_instruct( ops.branch(3), [0] )
    
    qc.run()
    qc.index_aide()
    
# test_diffusion()
# test_logic()
# test_matrix_check()
# test_control_ops()
test_idea2()
# test_grover()
# test_goto()
