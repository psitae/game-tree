#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:25:28 2020

@author: tony
"""


from numpy import *
from matplotlib.pyplot import *

# NOTE: only round() things before they are string-ified

# preprocessing
n = 60
overlap = sqrt(1/n)

# create |s> , |ans> and |!ans>
s = tuple(ones(n) * overlap)
ans = zeros(n)
ans[0] = 1
nans = array(s) * sqrt(n/(n-1))
nans[0] = 0

# print('ans\t', ans)
# print('nans\t', nans)

# calculate 2D |s> projection
proj_ans = inner(s, ans)
proj_rest = sqrt(1 - proj_ans**2)
disp_s = proj_rest, proj_ans

# print('proj_ans', proj_ans)
# print('proj_rest1', proj_rest)
# print('proj_rest2', inner(nans, s))

# unicode characters
not_ = "\u00AC"
theta = "\u03F4"
phi = "\u03C6"
             
# compute angles
theta_val = ( arcsin(overlap) )
theta_text = theta + ' = ' + str(degrees(theta_val).round(4))
theta_loc = cos(theta_val/2) + .03, sin(theta_val/2) + .03
phi_val = ( arccos(overlap)/2 )
phi_text = phi + ' = ' + str(degrees(phi_val).round(4))
phi_loc_angle = theta_val + phi_val / 2
phi_loc = cos(phi_loc_angle) + .03, sin(phi_loc_angle) + .03


# flip state & operator
disp_f = cos( theta_val + phi_val ), sin( phi_val + theta_val ) 
flip_ =  nans.dot(disp_f[0]) + ans.dot(disp_f[1])
# print('flip', flip_)
flip_op = 2 * outer(flip_, flip_) - identity(n)

# compare to normal grovers
s_flip = 2 * outer(s, s) - identity(n)
entries = ones(n)
entries[0] = -1
not_flip = diag(entries)


# flip |s> to |ans>

diffused = flip_op @ s
# project
disp_d = inner(diffused, nans), inner(diffused, ans)

print('|s> --', proj_rest.round(3), proj_ans.round(3) )
print('|f> --', disp_f[0].round(3), disp_f[1].round(3) )
print('|d> --', disp_d[0].round(3), disp_d[1].round(3) )

# create plot
origin = [0], [0]

ax = gca()
ax.cla()

circle1 = Circle((0,0), 1, fill=False)
ax.add_artist(circle1)

ax.arrow(0, 0, *disp_s, head_width=0.05, head_length=0.05)
ax.arrow(0, 0, *disp_f, head_width=0.05, head_length=0.05)
ax.text( proj_rest + .1, proj_ans + .1, '|s>')
ax.text( disp_f[0] + .1, disp_f[1] + .1, '|f>')
ax.text( .9, .9, theta_text)
ax.text( *theta_loc, theta)
ax.text( 0.9, 1.2, phi_text)
ax.text( *phi_loc, phi)

a = 1.2
xlim(-a, a)
ylim(-a, a)
ax.set_aspect('equal', adjustable='box')

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Set ticks
ax.set_xticks(arange(0,.25,1))
ax.set_yticks(arange(0,.25,1))

# Eliminate tick labels
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])

# Label axes
xlab = ax.set_xlabel('|' + not_ + '0...0>', fontsize=12)
ax.xaxis.set_label_coords(1.05, .48)
ax.text(-0.5, 1.1, '|0...0>', fontsize=12)

title('Unit Circle', pad=10)
show()