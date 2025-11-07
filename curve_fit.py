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

exp = []
XKz = [0.5, 0.75, 0.875, 1, 1.125, 1.25, 1.5]

X = [0.16,0.27,0.38,0.49]
#Y 
"""--- Vlaeurs simu ---"""
E_Es_kz05_m    = [0.6913861224489797/6.6, 1.2145779591836734/6.6, 1.7701769795918365/6.6, 2.362675918367347/6.6]
E_Es_kz075_m   = [0.5299624489795919/6.6, 0.9369355102040816/6.6, 1.4068796734693878/6.6, 1.904906448979592/6.6]
E_Es_kz0875_m  = [0.4670661224489796/6.6, 0.8382432653061225/6.6, 1.2600024489795918/6.6, 1.717410612244898/6.6]
E_Es_kz1_m     = [0.41328979591836734/6.6, 0.7505697959183674/6.6, 1.138921306122449/6.6, 1.56284/6.6]
E_Es_kz1_125_m = [0.3638432653061225/6.6, 0.674106612244898/6.6, 1.0270460408163264/6.6, 1.418537632653061/6.6]
E_Es_kz1_25_m  = [0.3232785306122449/6.6, 0.6083332244897959/6.6, 0.9355928163265307/6.6, 1.2999182040816326/6.6]
E_Es_kz1_5_m   = [0.2292517551020408/6.6, 0.4494238367346939/6.6, 0.7071381224489797/6.6, 0.9957288163265305/6.6]

"""--- Valeurs Jules exp ---"""

E_Es_kz05 = [0.2/6.6, 0.56/6.6, 0.93/6.6, 1.31/6.6]
E_Es_kz075 = [0.17/6.6, 0.41/6.6, 0.76/6.6, 1.02/6.6]
E_Es_kz0875 = [0.11/6.6, 0.34/6.6, 0.58/6.6, 0.89/6.6]
E_Es_kz1 = [0.10/6.6, 0.29/6.6, 0.52/6.6, 0.86/6.6]
E_Es_kz1_125 = [0.09/6.6, 0.28/6.6, 0.46/6.6, 0.76/6.6]
E_Es_kz1_25 = [0.07/6.6, 0.22/6.6, 0.41/6.6, 0.67/6.6]
E_Es_kz1_5 = [0.06/6.6, 0.16/6.6, 0.37/6.6, 0.60/6.6]


# plt.semilogy(X,E_Es_kz05, label='k_z = 0.5')
# plt.semilogy(X,E_Es_kz075, label='k_z = 0.75')
# plt.semilogy(X,E_Es_kz0875, label='k_z = 0.875')
# plt.semilogy(X,E_Es_kz1, label='k_z = 1')
# plt.semilogy(X,E_Es_kz1_125, label='k_z = 1.125')
# plt.semilogy(X,E_Es_kz1_25, label='k_z = 1.25')
# plt.semilogy(X,E_Es_kz1_5, label='k_z = 1.5')


plt.semilogy(X,E_Es_kz05_m,'--', label='k_z_m = 0.5')
plt.semilogy(X,E_Es_kz075_m,'--', label='k_z_m = 0.75')
plt.semilogy(X,E_Es_kz0875_m,'--', label='k_z_m = 0.875')
plt.semilogy(X,E_Es_kz1_m,'--', label='k_z_m = 1')
plt.semilogy(X,E_Es_kz1_125_m,'--', label='k_z_m = 1.125')
plt.semilogy(X,E_Es_kz1_25_m,'--', label='k_z_m = 1.25')
plt.semilogy(X,E_Es_kz1_5_m,'--', label='k_z_m = 1.5')

# plt.show()
# assert 0

"""--- Fonction puissance ---"""

def func(x, a, n):
    return a*x**n

xdata = X

ydata1 = E_Es_kz05_m
ydata2 = E_Es_kz075_m
ydata3 = E_Es_kz0875_m
ydata4 = E_Es_kz1_m
ydata5 = E_Es_kz1_125_m
ydata6 = E_Es_kz1_25_m
ydata7 = E_Es_kz1_5_m

# popt1, pconv1 = curve_fit(func2, xdata, ydata7)
# popt2, pconv2 = curve_fit(func, xdata, ydata2)
# popt3, pconv3 = curve_fit(func, xdata, ydata3)
# popt4, pconv4 = curve_fit(func, xdata, ydata4)
# popt5, pconv5 = curve_fit(func, xdata, ydata5)
# popt6, pconv6 = curve_fit(func, xdata, ydata6)
# popt7, pconv7 = curve_fit(func, xdata, ydata7)

"""--- Version classique ---"""

# params, covariance = curve_fit(func, X, E_Es_kz05)
# a, n = params
# exp.append(n)

# params, covariance = curve_fit(func, X, E_Es_kz075)
# a2, n2 = params
# exp.append(n2)

# params, covariance = curve_fit(func, X, E_Es_kz0875)
# a3, n3 = params
# exp.append(n3)

# params, covariance = curve_fit(func, X, E_Es_kz1)
# a4, n4 = params
# exp.append(n4)

# params, covariance = curve_fit(func, X, E_Es_kz1_125)
# a5, n5 = params
# exp.append(n5)

# params, covariance = curve_fit(func, X, E_Es_kz1_25)
# a6, n6 = params
# exp.append(n6)

# params, covariance = curve_fit(func, X, E_Es_kz1_5)
# a7, n7 = params
# exp.append(n7)

"""--- Version Log ---"""

logx = np.log(X)
logy = np.log(E_Es_kz05_m)
logy2 = np.log(E_Es_kz075_m)
logy3 = np.log(E_Es_kz0875_m)
logy4 = np.log(E_Es_kz1_m)
logy5 = np.log(E_Es_kz1_125_m)
logy6 = np.log(E_Es_kz1_25_m)
logy7 = np.log(E_Es_kz1_5_m)

logx_sub = logx[-3:]
logy_sub = logy[-3:]
logy_sub2 = logy2[-3:]
logy_sub3 = logy3[-3:]
logy_sub4 = logy4[-3:]
logy_sub5 = logy5[-3:]
logy_sub6 = logy6[-3:]
logy_sub7 = logy7[-3:]


n, loga = np.polyfit(logx_sub, logy_sub, 1)
a = np.exp(loga)
exp.append(n)

n2, loga2 = np.polyfit(logx_sub, logy_sub2, 1)
a2 = np.exp(loga2)
exp.append(n2)

n3, loga3 = np.polyfit(logx_sub, logy_sub3, 1)
a3 = np.exp(loga3)
exp.append(n3)

n4, loga4 = np.polyfit(logx_sub, logy_sub4, 1)
a4 = np.exp(loga4)
exp.append(n4)

n5, loga5 = np.polyfit(logx_sub, logy_sub5, 1)
a5 = np.exp(loga5)
exp.append(n5)

n6, loga6 = np.polyfit(logx_sub, logy_sub6, 1)
a6 = np.exp(loga6)
exp.append(n6)

n7, loga7 = np.polyfit(logx_sub, logy_sub7, 1)
a7 = np.exp(loga7)
exp.append(n7)

"""--- Calcul du R² ---"""

# residuals = E_Es_kz05 - func(X, a, n)
# ss_res = np.sum(residuals**2)
# ss_tot = np.sum((E_Es_kz05 - np.mean(E_Es_kz05))**2)
# r2 = 1 - (ss_res / ss_tot)
# print(f"R² = {r2:.4f}")

"""--- Régression linéaire classique ---"""

# a2, b2, c2 = popt2
# a3, b3, c3 = popt3
# a4, b4, c4 = popt4
# a5, b5, c5 = popt5
# a6, b6, c6 = popt6
# a7, b7, c7 = popt7

# print(f"les paramètres ajustés sont : a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")

"""--- Tracer de la fonction ajustée ---"""

xfit = np.linspace(min(X), max(X), 100)

yfit1 = func(xfit, a, n)
yfit2 = func(xfit, a2, n2)
yfit3 = func(xfit, a3, n3)
yfit4 = func(xfit, a4, n4)
yfit5 = func(xfit, a5, n5)
yfit6 = func(xfit, a6, n6)
yfit7 = func(xfit, a7, n7)

# plt.figure(figsize=(10,6))

# plt.grid()
# plt.xlabel('ρ*/ρs')
# plt.ylabel('E*/Es')
# plt.legend()
# plt.title("Diagramme Gibson-Ashby")

"""--- Plot données ---"""

# plt.plot(xdata, ydata1, 'o-', label='kz0.5')
# plt.plot(xdata, ydata2, '.-', label='kz0.75')
# plt.plot(xdata, ydata3, 'd-', label='kz0.875')
# plt.plot(xdata, ydata4, 'p-', label='kz1')
# plt.plot(xdata, ydata5, '*-', label='kz1.125')
# plt.plot(xdata, ydata6, '.-', label='kz1.25')
# plt.plot(xdata, ydata7, 'D-', label='kz1.5')

"""--- Plot curve fit ---"""

plt.plot(xfit, yfit1, '.r--', label=f'Fit : y = {a:.2f}.x^{n:.2f}')
plt.plot(xfit, yfit2, 'c--', label=f'Fit : y = {a2:.2f}.x^{n2:.2f}')
plt.plot(xfit, yfit3, 'b--', label=f'Fit : y = {a3:.2f}.x^{n3:.2f}')
plt.plot(xfit, yfit4, 'm--', label=f'Fit : y = {a4:.2f}.x^{n4:.2f}')
plt.plot(xfit, yfit5, 'k--', label=f'Fit : y = {a5:.2f}.x^{n5:.2f}')
plt.plot(xfit, yfit6, 'g--', label=f'Fit : y = {a6:.2f}.x^{n6:.2f}')
plt.plot(xfit, yfit7, 'y--', label=f'Fit : y = {a7:.2f}.x^{n7:.2f}')

"""--- Plot n/kz ---"""
plt.figure(figsize=(10,6))
plt.plot(XKz, exp, 'o', color='r')

# plt.grid()
# plt.xlabel('dens')
# plt.ylabel('E*/Es')
# plt.legend()
# plt.title("Diagramme Gibson-Ashby")

# plt.show()