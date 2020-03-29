#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:40:03 2020

@author: tony
"""


from operators import *

def init_state(circuit):
    dim = prod(circuit)
    state = zeros(dim)
    state[0] = 1
    return state

def nim_move(circuit):
    # for now, I'll assume (3,3) (x) (3) format and generalize later
    #                index: 0 1       2
    
    # board controls history's first move
    z = ones([3,3])
    instruct = {'2': -z, #diffuse(3, [1,2]), 
                '1': 2*z    }
    
    control_codes = encode_state([3], Print=True)
    board_c_hist = create_control(circuit, 1, 0, instruct)
    
    return board_c_hist

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
    
class quantum_circuit():
    def __init__(self, dims, Print=False):
        if Print:
            print('Creating quantum circuit')
            print( array(dims) )
        self.dims = dims
        self.size = prod(dims)
        self.state = init_state(self.dims)
        self.gates = []
        
    def printout(self):
        output_state(self.dims, self.state, amplitude='no')
        
    def add_gate(self, gate):
        self.gates.append(gate)
    
    def run(self, Print=False):
        for gate in self.gates:
            if Print:
                print('Applying gate:\n', gate)
                print('to state:\n', self.printout() )
            self.state = gate @ self.state
            if Print:
                print('Result:\n')
                self.printout()
            
if __name__ == '__main__':
    qc = quantum_circuit([2,2], Print=True)
    
    qc.printout()    
    
    qc.add_gate(
        single( qc.dims, 0, gates('hadamard') )
               )

    qc.add_gate(gates('cnot'))
    
    qc.run()
    
    qc.printout()



