#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:40:03 2020

@author: tony
"""


import operators as ops
from numpy import *

class display_object(): # used to print out states with amplitude
    def __init__(self, amp, code):
        self.amp = amp
        self.code = code
        
def output_state(circuit, state, divisions, show_amp=False):
    # this function prints out states formatted as xxx|yyy> + ...
    # xxx is the amplitude, yyy is the basis vector

    objs = []
    size = prod(circuit)
    
    state_strings = []

    for i in range(size):
        if state[i] != 0:                  
            state_array = ops.get_encoding(circuit, i)
            [ state_array.insert(i+j,';') for i,j in enumerate(divisions) ]
            state_string =  ''.join([str(i) for i in state_array])
            state_strings.append(state_string)
    
    if show_amp:                                                        # amp                 state
        for i in range(len(state_strings)): objs.append( display_object(str(state[i].round(3)), state_strings[i]) ) 
                                                   
    else:                                                     # amp     state
        for i in range(len(state_strings)): objs.append( display_object('', state_strings[i] ))

    strings = []
    for i in objs:
        strings.append( i.amp + '|' + i.code  +'> ' )

    state_string = strings[0]
    
    for i in range(1,len(strings)):
        state_string += '+ ' + strings[i]
        
    print(state_string)
    
class nim:
    
    def __init__(self, board, Print=False):
        board_dims = list(array(board) + 1)
        self.depth = sum(board)
        self.l = len(board)
        history = [ self.depth - i for i in range(self.depth) ]
        ancillas = board_dims
        circuit = board_dims + history + ancillas

        
        if Print:
            print('Nim game ' + str(board) + ' makes a circuit ')
            print(array(circuit))
            print(array(range(len(circuit))))
            
        self.circuit = quantum_circuit(circuit, divisions = [self.l, self.l + self.depth ],  Print=True)
        init_state = [ board + [0]*self.depth + [0]*self.l ]
        self.circuit.specify_state(init_state, Print=True)

    def move(self):
        # # copy the board information to ancillas
        # for i in range(self.l):
        #     copy_op = ops.copy(self.circuit.dims[i])
        #     self.circuit.add_gate(copy_op, [i, self.l + self.depth + i], Print=True)
        
        add_op = ops.basis_add()
        self.circuit.add_gate( )        
        self.circuit.run(Print=True)
        
   
class quantum_circuit():
    def __init__(self, dims, divisions = [], Print=False):

        self.dims = array(dims)
        self.size = prod(dims)
        self.length = len(dims)
        self.state = zeros(self.size)
        self.gates = []
        self.divisions = divisions
        if Print: print('Creating quantum circuit ', array(dims), ' with size ' + str(self.size) )
    
    def printout(self, show_amp=False):
        output_state(self.dims, self.state, self.divisions, show_amp)

    def init_state(self):
        self.state[0]
        
    def specify_state(self, states, Print=False):
        if any([ len(i) - self.length for i in states ]):
            print('\nCannot specify state')
            return
        l = len(states)
        norm = sqrt(1/l)
        state_locations = [ ops.get_location(self.dims, s) for s in states ]
        self.state[state_locations] = norm
        print('\nSpecifying state(s) ' + str(states) + ' located at ' + str(state_locations))
        
    def run(self, Print=False, amp=False):
        # initialize if not already done so
        if len(self.gates) == 0:
            print('\n' + 'X'*50)
            print('No operations to apply')
            return
        elif not any(self.state):
            print('\n' + 'X'*50)
            print('Cannot run quantum circuit without initializing state')
            return

        print('\n' + '='*50)
        print('Running quantum circuit')
        for gate in self.gates:
            if Print:
                print('\nApplying gate:\n', gate)
                print('with shape ', gate.shape)
                print('to state:')
                self.printout()
            self.state = gate @ self.state
            if Print:
                print('\nResult:')
                self.printout(amp)        

    def add_gate(self, gate, i=None, Print=False):
        
        print('\nAdding gate to circuit')
        if Print: print(gate)
        
        if not ops.is_unitary(gate):
            print('Gate not unitary')
            return
        
        dim = gate.shape[0]
        if dim == self.size:
            self.gates.append(gate)
        else:
            print('\nDiscovered need to integrate gate into larger circuit')
            if Print: print('Indexes: ', i)
            self.gates.append( ops.subsection(self.dims, i, gate, Print) )
    
    def basis_add(self, addens, receiver, Print=False):
        print('\nDoing basis add from q_program')
        addens.sort()
        actors = addens + receiver
        actors_i = list(range(len(actors)))
        ops.role_call(self.dims, actors)
        fake_receiver = addens[-1] + 1
        circuit_section = self.dims[actors]
        
        if Print:
            print('Actors\t', actors)
            print('C sect\t', circuit_section)
            print('actors_i', actors_i)
        
        gate_to_add = ops.basis_add(circuit_section, actors_i[:-1], actors_i[-1], Print)
        self.add_gate( gate_to_add, actors, Print)
        
    def fan_out(self, from_, to, Print=False):
        

        for i in range(l): self.state[state_locations] = norm
        if Print: self.printout()

    def practice_swap(self):
        perm = [1, 0, 2]
        swap_op = ops.swap2(self.dims, perm, True)
        self.add_gate(swap_op)
        
# qc = quantum_circuit([2,9,2])
# qc.basis_add([0], [len(qc.dims)-1], Print=True)
# qc.specify_state([[1,0,0],[1,1,0]])
# qc.run(Print=True)

n = nim([1,2])
n.move()


