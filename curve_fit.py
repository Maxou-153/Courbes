#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:23:10 2025

@author: mmiramon
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import curve_fit

point_list = [0.49,0.36,0.36,0.31,0.27,0.25,0.17]
kz = [0.5,0.75,0.875,1.0,1.125,1.25,1.5]


def func(x, a, b, c):
    return a*x**2+b*x+c

xdata = kz
ydata = point_list

popt, pconv = curve_fit(func, xdata, ydata)

a, b, c = popt
print(f"les paramètres ajustés sont : a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

xfit = np.linspace(min(kz), max(kz), 100)
yfit = func(xfit, a, b, c)


plt.figure(figsize=(10,6))

plt.plot(xdata, ydata, 'o-', label='Ecarts en MPa')
plt.plot(xfit, yfit, 'r--', label='Ajustement quadratique')

plt.grid()
plt.xlabel("kz")
plt.ylabel("E* (MPa)")
plt.legend()
plt.title("Ecarts")

plt.show()

