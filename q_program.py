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
    def __init__(self, gate, indx, mat, uni):
        self.gate = gate
        self.indx = indx
        self.mat  = mat
        self.is_unitary = uni
        
class gate_set:
    def __init__(self, *instruct):
        """
        Instruct should pass instruction objects.
        This class makes a list and gives the ability to iterate over the 
        gate or the gate index
        """
        self.depth = len(instruct)
        self.instructions = []
        for i in instruct:
            self.instructions.append(i)
    
    def add_instruct(self, instruct):
        if isinstance(instruct, instruction):
            self.instructions.append(instruct)
            self.depth += 1
        else:
            print('Did not pass instruction object')
        
    def get_gates(self):
        gates = []
        for i in self.instructions:
            gates.append(i.gate)
        
        return gates
    
    def get_indx(self):
        indx = []
        for i in self.instructions:
            gates.append(i.iindx)
        
        return indx
        
class quantum_circuit:
    """
    The main object in the algorithm.
    """
    
    def __init__(self, dims, divisions = [], name=False, Print=False):
        """
        Divisions is a list of the sizes of each division, it will be cum-added
        """
        if name: self.name = '\"' + name + '\"'
        
        self.dims = array(dims)
        self.size = prod(dims)
        self.length = len(dims)
        self.gate_set = gate_set()
        self.divisions = list(cumsum(divisions))
        if len(self.divisions) > 1: self.divisions.pop()
        self.halt = False
        
        # the state starts initialized to all 0, this can be changed by 
        # the write_state() method
        self.state = { (0,) * self.length : 1 }
        
        # matrix implementation
        self.mat = None
        self.column = None
        
        if Print: 
            print('Creating quantum circuit ', self.dims, 
                        ' with size ' + str(self.size) )
    
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
    def printout(self, show_amp=False):
        disp_objs = []
        for basis, amp in self.order_state():
            
            # insert ; for divisioning modules
            basis =  list(basis)
            [ basis.insert(i+j,';') for i,j in enumerate(self.divisions) ]
            state_str =  ''.join([str(i) for i in basis])
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
        
        if self.gate_set.depth == 0:
            print(error_dividor, 'No operations to apply')
            return
        
        print(dividor)
        print('Running quantum circuit', self.name)
        print('Initial state: \t')
        self.printout(show_amp)
        
        for i, instruct in enumerate(self.gate_set.instructions):
            num = str(i+1)
            print('\nInstruction ' + num + '\n' + '-'*20)
            print( instruct.gate.notes, ' acting on qudit(s) ', instruct.indx)
            if isinstance(self, mat_quantum_circuit):
                # print uber matrix, unitary or not
                # plt.figure(figsize=(10,10))
                ax = plt.matshow(instruct.mat.astype(float), cmap='rainbow')
                cbar = plt.colorbar( ticks = [] )
                ax.set_clim(-1, 1)
                # plt.show()
                
                plt.title('Gate: ' + num + '\nFull Matrix operation', pad=20)
                if instruct.is_unitary:
                    print('\nMatrix implementation: Unitary Operation \u2714')
                    unsimp = instruct.mat @ self.column              #  ✔
                    self.column = [ sp.simplify(i) for i in unsimp ] 
                    title_ = 'Gate ' + num + ':\n' + instruct.gate.notes
                    ops.printout(instruct.gate, title_ )
                    print('Matrix State: ' + num)
                    self.col2state(show_amp)
                else:
                    print(error_dividor, 'Non-unitary operation')
                    self.halt = True
                    break
                    
            # this is the integration method
            # maybe it's worth making a state object that does that by itself
            extract = []
            for basis, amp in self.state.items():
                extract.append([ basis,
                                amp,
                                [ basis[i] for i in instruct.indx ] ])
            # extract is [ original-basis , original amp, relevent sub-basis ]
            #                   0               1               2
            for ex in extract:#  
                dummy_state = { tuple(ex[2]) : ex[1] }
                ex[2] = instruct.gate.apply(dummy_state)
            # revelant sub-basis --> transformed sub-basis complete
            # extract is now a nested list where each element is the following
            # [  0 one basis element from the current state
            #    1 amplitude A corresponding to that state
            #    2 transformed relevant sub-basis list: [ b, amp_A*amp_B ] * n
            #   this is labeled bunch below
            self.state = {}
            for bunch in extract:
                # put the transformation back into the complete basis 
                
                in_basis = list(bunch[0])
                for out_pair in bunch[2].items():
                    for i, j in enumerate(instruct.indx):
                        in_basis[j] = out_pair[0][i]
                    in_key = tuple(in_basis)
                    if self.state.get( in_key ) == None:
                        self.state[ in_key] = out_pair[1]
                    else:
                        self.state[ in_key ] += out_pair[1]
                    #  self.printout(show_amp=True)
            
            # remove zero amps if they exist
            [ self.state.pop(basis) for basis, amp in list(self.state.items()) if amp == 0 ]
            
            print('\nState ' + num + ':')
            self.printout(show_amp)
        
        if not self.halt:
            print(dividor, '\nFinal State:')
            self.printout(show_amp)
        
    def add_instruct(self, gate, indx, mat=None, uni=None, Print=False):
        instruct = instruction(gate, indx, mat, uni)
        self.gate_set.add_instruct(instruct)
        
class mat_quantum_circuit(quantum_circuit):
    
    def __init__(self, dims, divisions = [], name=False, Print=False):
        quantum_circuit.__init__(self, dims, divisions, name, Print)
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
            if gate.tt.type == 'diffuse':
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
            elif gate.tt.type == 'perm':
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
        super(mat_quantum_circuit, self).add_instruct(gate, indx, uber, uni, Print)
    
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
    
# test_logic()
# test_matrix_check()
test_control_ops()






