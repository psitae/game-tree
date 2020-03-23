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