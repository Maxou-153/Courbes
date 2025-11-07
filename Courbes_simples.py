#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 15:22:33 2025

@author: mmiramon
"""

import numpy as np
import matplotlib.pyplot as plt

"""--- DATA ---"""

X = [0.16,0.27,0.38,0.49] #kz
Y = []

E_Es_kz05 = [0.6913861224489797,1.2145779591836734,1.7701769795918365,2.362675918367347]
E_Es_kz075 = [0.5299624489795919,0.9369355102040816,1.4068796734693878,1.904906448979592]
E_Es_kz0875 = [0.4670661224489796,0.8382432653061225,1.2600024489795918,1.717410612244898]
E_Es_kz1 = [0.41328979591836734,0.7505697959183674,1.138921306122449,1.56284]
E_Es_kz1_125 = [0.3638432653061225,0.674106612244898,1.0270460408163264,1.418537632653061]
E_Es_kz1_25 = [0.3232785306122449,0.6083332244897959,0.9355928163265307,1.2999182040816326]
E_Es_kz1_5 = [0.2292517551020408,0.4494238367346939,0.7071381224489797,0.9957288163265305]

# X1 = [0.5,0.25,0.1,0.075,0.06,0.05,0.048,0.047,0.046,0.045,0.04,0.025,0.01,0.0075]
# Y1 = [0.335552,0.334525,0.332612,0.331773,0.331392,0.331262,0.331233,0.331190,0.331091,0.330881,0.330686,0.330131,0.327753,0.326776]

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

# for i in Y1:
#     r = i/(0.175)
#     Y1_E.append(r)

"""--- Tracer de courbes ---"""

plt.figure(figsize=(10,6))
plt.grid()

plt.plot(X, E_Es_kz05, '*-', color='blue', label='kz0.5')
plt.plot(X, E_Es_kz075, 'o-', color='red', label='kz0.75')
plt.plot(X, E_Es_kz0875, '.-', color='k', label='kz0.875')
plt.plot(X, E_Es_kz1, '+-', color='magenta', label='kz1')
plt.plot(X, E_Es_kz1_125, 'x-', color='green', label='kz1.125')
plt.plot(X, E_Es_kz1_25, 'p-', color='cyan', label='kz1.25')
plt.plot(X, E_Es_kz1_5, 'd-', color='orange', label='kz1.5')

# plt.plot(X1, Y1_E, 'o-', color='red', label='Mesh/E*')

# plt.xticks(X1)

"""--- Etiquettes sur valeurs ---"""

# for x, y in zip(X, E_Es_kz05):
#     plt.text(x, y, f"{x:.3f}", ha='left', va='bottom', fontsize=9)

"""--- Lignes en pointillé ---"""    

# ymin, ymax = plt.ylim()
# plt.vlines(X1, ymin, Y1_E, colors='black', linestyles='dotted', alpha=0.6)
# plt.ylim(ymin, ymax)

"""--- Optiond des courbes ---"""

plt.xlabel(r"$\log\left(\frac{\rho^*}{\rho_s}\right)$")

plt.ylabel('E*/Es')
plt.title("Diagramme Gibson - Ashby")
plt.legend()
plt.show()# -*- coding: utf-8 -*-

