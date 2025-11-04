#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:22:33 2025

@author: mmiramon
"""

import numpy as np
import matplotlib.pyplot as plt

"""--- DATA ---"""

X = [0.5,0.75,0.875,1.0,1.125,1.25,1.5] #kz
Y = []

X1 = [0.5,0.25,0.1,0.075,0.05,0.025,0.01]
Y1 = [0.335552,0.334525,0.332612,0.331773,0.331262,0.330131,0.327753]

# tags = ["0.03","0.05","0.1","0.25","0.5","0.75","1"]

"""--- Calculs ---"""

stress = []
Y1_E = []

# for i in Y1:
#     s = i/25
#     stress.append(s)

# for i in stress:
#     e = i/0.005
#     Y1_E.append(e)

for i in Y1:
    r = i/(0.175)
    Y1_E.append(r)

"""--- Tracer de courbes ---"""

plt.figure(figsize=(10,6))
plt.grid()

# plt.plot(X1, Y1, 'o-', color='blue', label='X/Y')
plt.plot(X1, Y1_E, 'o-', color='red', label='Mesh/E*')

# plt.xticks(X1)

"""--- Etiquettes sur valeurs ---"""

for x, y in zip(X1, Y1_E):
    plt.text(x, y, f"{x:.2f}", ha='left', va='bottom', fontsize=9)

"""--- Lignes en pointillé ---"""    

ymin, ymax = plt.ylim()
plt.vlines(X1, ymin, Y1_E, colors='black', linestyles='dotted', alpha=0.6)
plt.ylim(ymin, ymax)

"""--- Optiond des courbes ---"""

plt.xlabel('Maillage')
plt.gca().invert_xaxis()
plt.ylabel('E* (MPa)')
plt.title("Etude de convergence")
plt.legend()
plt.show()