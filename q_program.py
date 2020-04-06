#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:40:03 2020

@author: tony
"""


import operators as ops
import sympy as sp
from numpy import *


class display_object:
    """
    used to print out state-kets w/wo amplitude
    """
    def __init__(self, amp, code):
        self.amp = amp
        self.code = '|' + code + '>'
        
    
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
            self.circuit.run()
        
class instruction:
    def __init__(self, gate, indx):
        self.gate = gate
        self.indx = indx
        
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
    
    def __init__(self, dims, divisions = [], Print=False):

        self.dims = array(dims)
        self.size = prod(dims)
        self.length = len(dims)
        self.gate_set = gate_set()
        self.divisions = divisions
        
        # the state starts initialized to all 0, this can be changed by 
        # the write_state() method
        self.state = { '1' : [0] * self.length }
        if Print: 
            print('Creating quantum circuit ', self.dims, 
                        ' with size ' + str(self.size) )
    def write_state(self, *state):
        """
        State should be a dict object with
        { basis : amplitude } pairs.
        Also accepts strings like '0101' and converts them to dict objects
        """
        if isinstance(state, dict): 
            self.state = state
        elif all([ isinstance(s, str) for s in state ]):
            norm = 1/sp.sqrt(len(state))
            for each_state in state:
                self.state = { tuple([ int(s) for s in each_state ]) : norm }
        else:
            print('Improper state')
    def printout(self, show_amp = False):
        
        disp_objs = []
        for basis, amp in self.state.items():
            
            # insert ; for divisioning modules
            [ basis.insert(i+j,';') for i,j in enumerate(self.divisions) ]
            state_str =  ''.join([str(i) for i in basis])
            if show_amp:
                shown_amp = amp
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
        
        if self.gate_set.depth == 0:
            print('\n' + 'X'*50)
            print('No operations to apply')
            return
        
        print('\n' + '='*50)
        print('Running quantum circuit')
        for instruct in self.gate_set.instructions:
            if Print:
                print('\nApplying gate:\t', instruct.gate.notes)
                print('to qudits ', instruct.indx)
                
            self.state = instruct.gate.apply(self.state, Print)
    
    def add_instruct(self, instruct, Print=False):
        self.gate_set.add_instruct(instruct)
        
def mat_quantum_circuit(quantum_circuit):
    def __init__(self, dims, divisions = [], Print=False):
        quantum_circuit.__init__(self, dims, divisions, Print)
        self.mat = ops.tt2mat(self.tt)
        self.column = self.init_column()
        
    def init_column(self):
        col = zeros(self.length)
        col[0] = 1
        self.column = col
        
    def specify_state(self, *state_str, Print=False):
        """
        state_str is a string, like '01100' or list of such strings.
        This function finds the corresponding state vector and stores
        it in self.column
        """
        
        # assuming equal superposition of all input state strings
        norm = 1/sp.sqrt(len(state_str))
        
        state_locations = []
        for str_ in state_str:
            
            # check validity of input
            if len(str_) > self.length:
                print('\nState too long to specify')
                return
            
            state_locations.append(ops.get_location(self.dims, str_))
            self.state[state_locations] = norm
            print('\nSpecifying state(s) ')
            [ print(s) for s in state_str] 
            print('located at ' + str(state_locations) )
            
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
        
def test_fan_out():
    qc = quantum_circuit([2,2], divisions = [])
    qc.write_state('10', '00')
    qc.printout()
    
    gate = ops.fan_out(2, 0, 1)
    instruct = instruction(gate, (0, 1))
    qc.add_instruct(instruct)
    qc.run(Print=True, show_amp = True)

test_fan_out()


