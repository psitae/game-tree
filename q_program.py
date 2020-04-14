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

null  = '\u2205'

def indexer(lst, *indices):
    return (lst[i] for i in indices)

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
    def __init__(self, gate, indx, num):
        self.gate = gate
        self.indx = indx
        self.num  = str(num)
    
    def integrate(self, state, halt=None):
        
        collect = {}
        for full_basis, amp in state.items():
            sub_basis = tuple([ full_basis[i] for i in self.indx ])
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
            dummy_out = self.gate.apply({ sub_basis : 1 })
            for full_basis, amp in fulls.items():
                full_basis = list(full_basis)
                for dummy_basis, dummy_amp in dummy_out.items():
                    for i, jay in zip(self.indx, range(len(self.indx))):
                        full_basis[i] = dummy_basis[jay]
                    add_to_state = tuple(full_basis)
                    # add amplitude to value in dict if it exists
                    present = state.get(add_to_state, 0)
                    state[add_to_state] =  present +  amp * dummy_amp

        # remove zero amps if they exist
        [ state.pop(basis) for basis, amp in list(state.items()) if amp == 0 ]
        return state
    
class mat_instruction(instruction):
    """
    Adds matrix functionality to the instructions class.
    The intended use of this class is for human-readably small cases,
    to check if the normal sped-up methods are giving the right answers
    """
    def __init__(self, gate, indx, num, mat, uni):
        instruction.__init__(self, gate, indx, num)
        self.mat = mat
        self.is_unitary = uni
    
    def integrate(self, qc):
        # normal integration methods with truth tables
        qc.state = super().integrate(qc.state)
        
        # display uber matrix visually, unitary or not
        ax = plt.matshow(self.mat.astype(float), cmap='rainbow')
        cbar = plt.colorbar( ticks = [] )
        ax.set_clim(-1, 1)
        plt.title('Gate: ' + self.num + '\nFull Matrix operation', pad=20)
        
        if self.is_unitary:
            # perform update of column
            print('\nMatrix implementation: Unitary Operation \u2714')
            unsimp = self.mat @ qc.column                    #  ✔
            qc.column = [ sp.simplify(i) for i in unsimp ] 
            
            # display gate as matrix and truth table
            title_ = 'Gate ' + self.num + ':\n' + self.gate.notes
            ops.printout(self.gate, title_ )
            
            # print out state 
            print('Matrix State: ' + self.num)
            qc.col2state(qc.show_amp)
            
        else:
            print(error_dividor, 'Non-unitary operation')
            qc.halt = True
        
        
        

class quantum_circuit:
    """
    The main object in the algorithm.
    """
    
    def __init__(self, dims, divisions = [], name=False, show_amp=False):
        """
        Divisions is a list of the sizes of each division, it will be cum-added
        """
        # printing settings
        if name: self.name = '\"' + name + '\"'
        self.show_amp = show_amp
        self.encodings = []
        
        # calculation attributes
        self.dims = array(dims)
        self.size = prod(dims)
        self.length = len(dims)
        self.gate_set = []
        self.depth = 0
        self.divisions = list(cumsum(divisions))
        if len(self.divisions) > 1: self.divisions.pop()
        self.halt = False
        
        # initialize state
        self.state = { (0,) * self.length : 1 }
        
        print('Creating quantum circuit' + self.name + ':')
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
        for s in state:
            if len(s) != self.length:
                print('Improper state assignment')
                self.halt = True
                self.state = temp
                return
            
            basis = tuple([ int(num) for num in s ])
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
            swap_dict =  { 0: null, 1:'T', 2:'F'}
        elif scheme == 'TF':
            swap_dict = { 0: 'T', 1: 'F'}
        elif isinstance(scheme, dict):
            swap_dict = scheme
        else:
            print('Scheme error')
            return
        
        [ self.encodings.append( (i, swap_dict) ) for i in indx ]
    
    def printout(self, show_amp=False):
        disp_objs = []
        for basis, amp in self.order_state():
            
            # insert ; for divisioning modules
            basis =  list(basis)
            [ basis.insert(i+j,';') for i,j in enumerate(self.divisions) ]
            # form a string
            state_str =  ''.join([str(i) for i in basis])
            # modify string with special encodings
            for indx, swap_dict in self.encodings:
                state_str[indx] = swap_dict.get(int(state_str[indx]))
            
            if show_amp:
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
        
    def run(self, Print=False, show_amp=False):
        """
        Checks for the proper conditions, then applies each gate in order,
        printing each state as the program progresses.
        """
        dividor = '\n' + '='*50
        error_dividor = '\n' + 'X'*50 + '\n'
        
        # Are we really ready to run the circuit?
        if self.halt:
            print(error_dividor, 'Halted')
            return
        
        if self.depth == 0:
            print(error_dividor, 'No operations to apply')
            return
        
        print(dividor)
        print('Running quantum circuit', self.name)
        print('Initial state: \t')
        self.printout(show_amp)
        
        for instruct in self.gate_set:
            print('\n' + '_'*50+ '\n')
            print('Instruction ' + instruct.num + '\n' + '-'*20)
            print( instruct.gate.notes, ' acting on qudit(s) ', instruct.indx)

            # a little bit of oop ugliness
            if isinstance(self, mat_quantum_circuit):
                # mat_instruct gets whole qc object passed
                instruct.integrate(self)
            else:
                self.state = instruct.integrate(self.state)
                
            print('\nState ' + instruct.num + ':')
            self.printout(self.show_amp)
        
        if not self.halt:
            print(dividor, '\nFinal State:')
            self.printout(self.show_amp)
        
    def add_instruct(self, gate, indx):
        self.depth += 1
        instruct = instruction(gate, indx, self.depth)
        self.gate_set.append(instruct)
        
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
        
        super(mat_quantum_circuit, self).write_state(state)
        
        # assuming equal superposition of all input state strings
        norm = 1/sp.sqrt(len(state))
        
        state_locations = []
        for st in state:
            
            # check validity of input
            if len(st) > self.length:
                print('\nState too long to specify')
                return
            
            state_locations.append(ops.get_location(self.dims, st))
            self.column[state_locations] = norm
            
    def add_instruct(self, gate, indx, Print=False):
        """
        Creates a matrix equivalent to the (gate + index)
        Calls add_instruct() in the parent function, passing gate, indx and mat
        """
        print('adding matrix instruct' + gate.notes)
        # do we need matrix integration?
        if len(indx) == len(self.dims):
            if gate.mat is None: gate.mat = ops.tt2mat(gate.tt)
            uber = gate.mat
        else:
            # set up repetitions
            rest_circ = list(self.dims)
            for num, idx in enumerate(indx):
                rest_circ.pop(idx - num)
            rest_encoding = ops.encode_state(rest_circ, type_='list')
            full_encoding = ops.encode_state(self.dims, type_='list')
            
            # matrix integration
            # diffuse gates
            if isinstance(gate, ops.diff_gate):
                uber = identity(self.size, object)
                for in_basis, out_pairs in gate.tt.table.items():
                    for code in rest_encoding:
                        full_in_code = copy.copy(code)
                        [ full_in_code.insert(i, in_basis[j]) for j, i in enumerate(indx)]
                        in_loc = full_encoding.index(full_in_code)
                        uber[ in_loc, in_loc ] = 0
                        for pair in out_pairs:
                            full_out_code = copy.copy(code)
                            [ full_out_code.insert(i, pair[0][j])
                             for j, i in enumerate(indx) ]
                            out_loc = full_encoding.index(full_out_code)
                            uber[out_loc, in_loc] = pair[1]
            
            # perm gates
            elif isinstance(gate, ops.perm_gate):
                uber = identity(self.size, uint8)
                for in_basis, out_basis in gate.tt.table.items():
                    for code in rest_encoding:
                        full_in_code = copy.copy(code)
                        full_out_code = copy.copy(code)
                        for j, i in enumerate(indx):
                            full_in_code.insert(i, in_basis[j])
                            full_out_code.insert(i, out_basis[j])
                        in_loc = full_encoding.index(full_in_code)
                        out_loc = full_encoding.index(full_out_code)
                        uber[ in_loc, in_loc ] = 0
                        uber[ out_loc, in_loc ] = 1
        
        uni = ops.is_unitary(uber)
        self.depth += 1
        instruct = mat_instruction(gate, indx, self.depth, uber, uni)
        self.gate_set.append(instruct)
        # super(mat_quantum_circuit, self).add_instruct(gate, indx, uber, uni, Print)
    
    def col2state(self, show_amp):
        """
        Convert the column object into a normal state object and prints
        it out for comparison to self.state
        """
        temp = self.state
        col_state = { tuple(ops.get_encoding(self.dims, loc)) : amp 
                for loc, amp in enumerate(self.column) if amp != 0 }
        
        self.state = col_state
        self.printout(show_amp)
        self.state = temp

def test_fan_out():
    qc = mat_quantum_circuit([2,2,2], divisions = [], name='Test fan out')
    qc.write_state('010', '000')
    
    gate = ops.fan_out(2, 0, 1)
    instruct = instruction(gate, (1, 2))
    qc.add_instruct(instruct)
    qc.run(Print=True, show_amp = True)

def test_diffusion():
    qc = quantum_circuit([4,4])
    qc.write_state('00', '10', '20', '30')
    
    gate1 = ops.goto_state(4, send=1)
    instruct = instruction(gate1, [0])
    qc.add_instruct(instruct)
    
    gate2 = ops.branch(4)
    instruct = instruction(gate2, [1])
    qc.add_instruct(instruct)
    
    qc.run()
    
def test_logic():
    qc = mat_quantum_circuit([3,3,2,2,2], divisions=[2,2,1], name='Test AND')
    
    b = ops.goto_state(3)
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
    
    qc.run(True)
    qc2.run(True)

def test_matrix_check():
    qc = mat_quantum_circuit([2,3,2], name='Test matrix check')
    
    g1 = ops.branch(3)
    qc.add_instruct( g1, [1] )
    g2 = ops.goto_state(3)
    qc.add_instruct( g2, [1] )
    
    qc.run(show_amp=True)
    
def test_control_ops():
    qc = mat_quantum_circuit([3,3], name='Test control operations')
    
    b = ops.branch(3)
    qc.add_instruct( b, [0])
    qc.add_instruct( b, [1])
    
    go1 = ops.goto_state(3)
    go2 = ops.goto_state(3, 2)
    
    directions = { (1,) : go1, (2,) : go2 }
    
    ctl_go = ops.create_control([3,3], 0, 1, directions)
    qc.add_instruct(ctl_go, [0, 1])
    
    qc.run(show_amp=True)
    
def test_idea():
    
    qc = mat_quantum_circuit([3,3,2,2,2,2], divisions = [2,3,1],
        name='Test integrate', show_amp=False)
    qc.special_encoding('null', 0,1)
    qc.special_encoding('TF', 2,3,4)
                        
    b = ops.branch(3)
    qc.add_instruct( b , [0])
    qc.add_instruct( b , [1])
    
    c32 = ops.copy32(0,1)
    n32 = ops.not32(0,1)
    
    qc.add_instruct( c32, [0, 2])
    qc.add_instruct( c32, [1, 3])
    
    qc.add_instruct( n32, [1, 4])
    
    qc.run()
    
    
    
    
    
# test_logic()
# test_matrix_check()
# test_control_ops()
# test_integrate()
test_idea()
