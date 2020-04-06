#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:40:03 2020

@author: tony
"""


import operators2 as ops
import sympy as sp
from numpy import *


class display_object: # used to print out states with amplitude
    def __init__(self, amp, code):
        self.amp = amp
        self.code = '|' + code + '>'
        
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
    
    if show_amp:
        for i in range(len(state_strings)):
            #                                       amp                 state
            objs.append( display_object(str(state[i].round(3)), state_strings[i]) ) 
                                                   
    else:
        for i in range(len(state_strings)):
                                         # amp     state
            objs.append( display_object('', state_strings[i] ))

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

        history = [ self.depth -i + 1 for i in range(self.depth) ]
        ancilla = [ self.depth + 1 ]

        self.board_i   = list(range(self.l))
        next_index = self.board_i[-1] + 1
        self.history_i = list(range(next_index, next_index + len(history) ))
        next_index = self.history_i[-1] + 1
        self.ancilla_i = list(range(next_index, next_index + len(ancilla) ))
        print('Indices:\n', 'board\t', self.board_i, '\nhistory\t', self.history_i, '\nancillas\t', self.ancilla_i)
        circuit = board_dims + history + ancilla

        if Print:
            print('Nim game ' + str(board) + ' makes a circuit ')
            print(array(circuit))
            print(array(range(len(circuit))))
            
        self.circuit = quantum_circuit(circuit, divisions = [self.l, self.l + self.depth ],  Print=True)
        #                       history       + ancilla
        init_state = [ board + [0]*self.depth + [0] ]
        self.circuit.specify_state(init_state, Print=True)

    def move(self):
        # # copy the board information to ancillas
        # for i in range(self.l):
        #     copy_op = ops.copy(self.circuit.dims[i])
        #     self.circuit.add_gate(copy_op, [i, self.l + self.depth + i], Print=True)
        
        # count number of possible moves
        add_op = ops.basis_add(
            self.circuit.dims[self.board_i], 
            self.circuit.dims[self.ancilla_i[0]], 
            Print=True )
        self.circuit.add_gate(add_op, self.board_i + self.ancilla_i)
        
        # conditionally diffuse
        instruct_1 = ops.conditional_diffusion(self.circuit.dims[self.history_i[0]])
        [ print(i,instruct_1[i]) for i in instruct_1 ]
        cond_diff_op = ops.create_control(self.circuit.dims, 
                  self.ancilla_i[0], self.history_i[0], instruct_1, Print=True)
        self.circuit.add_gate(cond_diff_op, self.history_i[0] + self.ancilla_i[0], True)
        
        # history changes board
        
        
        self.circuit.run(Print=True)
        
    def move_num(self, num):
        if num == 1 : 
            subtract_op = ops.basis_add([4], 2, mode='subtract', Print=True)
            self.circuit.add_gate(subtract_op, [2,0])
            # self.circuit.run()
            
class quantum_circuit:
    def __init__(self, dims, divisions = [], Print=False):

        self.dims = array(dims)
        self.size = prod(dims)
        self.length = len(dims)
        self.instructs = []
        self.divisions = divisions
        self.state = [0] * self.length
        if Print: print('Creating quantum circuit ', self.dims, ' with size ' + str(self.size) )
    
    def printout(self, show_amp=False):
        
        # search for populated states and create a list of
        # display objects to printout
        populated = []
        shown_amp = ''
        for loc, amp in enumerate(self.state):
            if amp != 0:
                state_array = ops.get_encoding(self.dims, loc)
                [ state_array.insert(i+j,';') for i,j in enumerate(self.divisions) ]
                state_str =  ''.join([str(i) for i in state_array])
                if show_amp:
                    shown_amp = amp
                populated.append( display_object(shown_amp, state_str ) )
        
        output_list = []
        for base in populated:
            output_list.append(base.amp + base.code)
        
        # intersperse with '+'
        together = [' + '] * (len(populated) * 2 - 1)
        together[0::2] = output_list
        output_str = ''.join([s for s in together])
        
        print(output_str)
                

    def specify_state(self, state_str, Print=False):
        """
        state_str is a string, like '01100' or list of such strings
        """
        
        # preprocessing
        if type(state_str) == str:
            states = [[ int(s) for s in state_str ]]
        if type(state_str) == list:
            states = []
            for state in state_str:
                states.append([ int(s) for s in state ])
                
            
        # check validity of input
        # if any([ len(state) - self.length for state in states ]):
        #     print('\nCannot specify state')
        #     return
        
        norm = 1/sp.sqrt(self.length)
        state_locations = [ ops.get_location(self.dims, s) for s in states ]
        self.state[state_locations] = norm
        print('\nSpecifying state(s) ')
        [ print(s) for s in state_str] 
        print('located at ' + str(state_locations) )
        
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
                print('\nApplying gate:\n', gate.name)
                print('to qudits ', gate.indx)
                print('to state:')
                self.printout()
            self.state = gate.apply(self.state)
            if Print:
                print('\nResult:')
                self.printout(amp)

    def add_gate(self, gate, indx=[], Print=False):
        
        self.gates.append((gate, indx))
        
def test_fan_out():
    qc = quantum_circuit([2,2], divisions = [])
    qc.specify_state(['10'])
    qc.printout()
    
    gate = ops.fan_out(qc.dims, 0, 1)
    qc.add_gate(gate)
    
    
test_add()



