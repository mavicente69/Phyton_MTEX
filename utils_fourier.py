# -*- coding: utf-8 -*-
"""
Motor de Armónicos Esféricos y Matrices de Wigner.
Calcula los coeficientes triclínico-triclínico y simula ODFs.
Entorno: texturaPy3.10
"""
import numpy as np
from scipy.special import factorial

def _wigner_small_d(l, m, n, beta):
    """
    Calcula el elemento d^l_{mn}(beta) de la matriz Wigner-D.
    Optimizada y vectorizada: acepta escalares o arrays de NumPy (N,).
    """
    beta = np.asarray(beta)
    k_min = max(0, n - m)
    k_max = min(l - m, l + n)
    
    d_val = np.zeros_like(beta, dtype=float)
    cos_b = np.cos(beta / 2.0)
    sin_b = np.sin(beta / 2.0)
    
    # Al usar factoriales estándar (float64), evitamos el colapso de np.sqrt con enteros gigantes
    prefactor = np.sqrt(factorial(l + m) * factorial(l - m) * factorial(l + n) * factorial(l - n))
    
    for k in range(k_min, k_max + 1):
        denominador = factorial(l - m - k) * factorial(l + n - k) * factorial(k + m - n) * factorial(k)
        
        signo = (-1)**(k - m + n)
        pot_cos = 2 * l - m + n - 2 * k
        pot_sin = 2 * k + m - n
        
        # NumPy maneja 0**0 = 1.0 por defecto, lo que hace seguro el cálculo vectorizado
        termino = signo * (1.0 / denominador) * (cos_b**pot_cos) * (sin_b**pot_sin)
        d_val += termino
        
    return prefactor * d_val

def calc_component_fourier(kernel, orientacion, L_max):
    """
    EXTRACCIÓN: Calcula los coeficientes C_lmn (Triclínico-Triclínico).
    """
    tabla_kernel = kernel.calc_fourier_coeffs(L_max, return_full=False)
    K_l = np.zeros(L_max + 1)
    for fila in tabla_kernel:
        l, m, n, val = int(fila[0]), int(fila[1]), int(fila[2]), fila[3]
        if m == n:
            K_l[l] = val

    euler_angles = orientacion.to_euler().flatten()
    phi1, Phi, phi2 = euler_angles[0], euler_angles[1], euler_angles[2]
    
    alpha = phi1 - (np.pi / 2.0)
    beta  = Phi
    gamma = phi2 + (np.pi / 2.0)
    
    coefs = []
    for l in range(L_max + 1):
        kl = K_l[l]
        if kl == 0.0:
            continue
            
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                d_mn = _wigner_small_d(l, m, n, beta)
                fase = np.exp(1j * (m * gamma + n * alpha))
                c_val = kl * d_mn * fase * (2 * l + 1)
                coefs.append((l, m, n, c_val))
                
    return coefs

def eval_Tlmn(l, m, n, orientaciones):
    """
    Evalúa la función base T_lmn pura de Bunge para un array de orientaciones.
    """
    euler = orientaciones.to_euler()
    if euler.ndim == 1: euler = euler.reshape(1, 3)
    
    phi1, Phi, phi2 = euler[:, 0], euler[:, 1], euler[:, 2]
    
    alpha = phi1 - (np.pi / 2.0)
    beta  = Phi
    gamma = phi2 + (np.pi / 2.0)
    
    d_mn = _wigner_small_d(l, m, n, beta)
    
    # La síntesis requiere la fase conjugada (-) de la extracción (+)
    fase = np.exp(-1j * (m * gamma + n * alpha))
    
    return d_mn * fase

def eval_odf_from_fourier(coefs_dict, orientaciones):
    """
    SÍNTESIS: Reconstruye la densidad física de la ODF (MUD) 
    sumando la serie de Fourier para las orientaciones dadas.
    """
    euler = orientaciones.to_euler()
    if euler.ndim == 1: euler = euler.reshape(1, 3)
    
    phi1, Phi, phi2 = euler[:, 0], euler[:, 1], euler[:, 2]
    
    alpha = phi1 - (np.pi / 2.0)
    beta  = Phi
    gamma = phi2 + (np.pi / 2.0)
    
    densidad_compleja = np.zeros(len(beta), dtype=complex)
    
    for (l, m, n), c_val in coefs_dict.items():
        if abs(c_val) < 1e-8:
            continue
            
        d_mn = _wigner_small_d(l, m, n, beta)
        fase = np.exp(-1j * (m * gamma + n * alpha))
        
        densidad_compleja += c_val * d_mn * fase
        
    # La matemática asegura que la parte imaginaria se cancela entre armónicos opuestos.
    # Extraemos la parte real limpia.
    return np.real(densidad_compleja)