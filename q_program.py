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
    
class quantum_circuit():
    def __init__(self, dims, divisions = [], Print=False):

        self.dims = array(dims)
        self.size = prod(dims)
        self.state = zeros(self.size)
        self.gates = []
        self.divisions = divisions
        if Print: print('Creating quantum circuit ', array(dims), ' with size ' + str(self.size) )
    
    def printout(self, show_amp=False):
        output_state(self.dims, self.state, self.divisions, show_amp)

    def run(self, Print=False, amp=False):
        # initialize if not already done so
        if not any(self.state): self.state[0] = 1
        print('\n' + '-'*50)
        print('Running quantum circuit')
        for gate in self.gates:
            if Print:
                print('\nApplying gate:\n', gate)
                print('with shape ', gate.shape)
                print('to state:')
                self.printout()
            self.state = gate @ self.state
            if Print:
                print('Result:\n')
                self.printout(amp)        

    def add_gate(self, gate, i=None):
        
        print('\nAdding gate to circuit:\n', gate)
        
        if not ops.is_unitary(gate):
            print('Gate not unitary')
            return
        
        dim = gate.shape[0]
        if dim == self.size:
            self.gates.append(gate)
        else:
            print('Discovered need to integrate gate into larger circuit')
            print('Indexes: ', i)
            self.gates.append( ops.subsection(self.dims, i, gate, True) )
    
    def basis_add(self, addens, receiver, Print=False):
        print('\nDoing basis add from q_program')
        addens.sort()
        actors = addens + receiver
        actors_i = list(range(len(actors)))
        ops.role_call(self.dims, actors)
        fake_receiver = addens[-1] + 1
        circuit_section = self.dims[actors]
        print('Actors\t', actors)
        print('C sect\t', circuit_section)
        print('actors_i', actors_i)
        self.add_gate( ops.basis_add(circuit_section, actors_i[:-1], actors_i[-1], Print), actors)
        
    def specify_state(self, states, Print=False):
        l = len(states)
        norm = sqrt(1/l)
        state_locations = [ ops.get_location(self.dims, s) for s in states ]
        print('\nSpecifying state(s) ' + str(states) + ' located at ' + str(state_locations))

        for i in range(l): self.state[state_locations] = norm
        if Print: self.printout()

class nim:
    
    def __init__(self, board):
        board_dims = array(board) + 1
        depth = sum(board)
        l = len(board)
        history = [ depth - i for i in range(depth) ]
        circuit = list(board_dims) + history + [depth+1]

        print('Nim game ' + str(board) + ' makes a circuit ')
        print(array(circuit))
        print(array(range(len(circuit))))
        self.game = quantum_circuit(circuit, divisions = [len(board), len(board) + depth ],  Print=True)
        self.game.specify_state([board + [0]*depth], Print=True)

    def move(self):
        self.game.basis_add([0,1], [3], Print=True)
        self.game.run(Print=True)
        self.game.printout()
        

qc = quantum_circuit([2,2,3], Print=True)
qc.basis_add([1], [2], Print=True)
qc.specify_state([[0,1,1],[1,1,1],[1,0,1]], Print=True)
qc.run(Print=True)


