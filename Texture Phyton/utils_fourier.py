# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:21:16 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Motor de Armónicos Esféricos y Matrices de Wigner.
Calcula los coeficientes triclínico-triclínico.
Entorno: texturaPy3.10
"""
import numpy as np
from scipy.special import factorial

def _wigner_small_d(l, m, n, beta):
    """
    Calcula el elemento d^l_{mn}(beta) de la matriz Wigner-D (Matriz pequeña).
    Usa la fórmula de sumatoria exacta de la mecánica cuántica. 
    Seguro computacionalmente para L_max <= 34 gracias al exact=True.
    """
    # Determinamos los límites válidos para el índice k para evitar factoriales negativos
    k_min = max(0, n - m)
    k_max = min(l - m, l + n)
    
    d_val = 0.0
    cos_b = np.cos(beta / 2.0)
    sin_b = np.sin(beta / 2.0)
    
    # Prefactor constante con raíces de factoriales (exact=True evita desbordamientos en floats)
    prefactor = np.sqrt(factorial(l + m, exact=True) * factorial(l - m, exact=True) * factorial(l + n, exact=True) * factorial(l - n, exact=True))
    
    for k in range(k_min, k_max + 1):
        denominador = (factorial(l - m - k, exact=True) * factorial(l + n - k, exact=True) * factorial(k + m - n, exact=True) * factorial(k, exact=True))
        
        signo = (-1)**(k - m + n)
        pot_cos = 2 * l - m + n - 2 * k
        pot_sin = 2 * k + m - n
        
        # Manejo seguro de 0^0 si el ángulo es exactamente 0 o pi
        term_cos = cos_b**pot_cos if pot_cos > 0 else (1.0 if cos_b == 0.0 else cos_b**pot_cos)
        term_sin = sin_b**pot_sin if pot_sin > 0 else (1.0 if sin_b == 0.0 else sin_b**pot_sin)
        
        termino = signo * (1.0 / denominador) * term_cos * term_sin
        d_val += termino
        
    return prefactor * d_val

def calc_component_fourier(kernel, orientacion, L_max):
    """
    Calcula los coeficientes C_lmn (Triclínico-Triclínico) para una única componente.
    Convierte internamente Bunge (ZXZ) a Roe/Wigner (ZYZ) para usar la matriz estándar.
    
    Retorna:
        Una lista de tuplas con la estructura: [(l, m, n, valor_complejo), ...]
    """
    # 1. Extraemos los K_l del kernel en el origen
    # Le pedimos la tabla pero nos quedamos solo con la diagonal (m == n)
    tabla_kernel = kernel.calc_fourier_coeffs(L_max, return_full=False)
    K_l = np.zeros(L_max + 1)
    for fila in tabla_kernel:
        l, m, n, val = int(fila[0]), int(fila[1]), int(fila[2]), fila[3]
        if m == n:
            K_l[l] = val

    # 2. Obtenemos los ángulos de Euler de la orientación base (Convención Bunge ZXZ)
    # Aplanamos con .flatten() por seguridad por si orix devuelve (1, 3) o (3,)
    euler_angles = orientacion.to_euler().flatten()
    phi1, Phi, phi2 = euler_angles[0], euler_angles[1], euler_angles[2]
    
    # --- MAPEADO CRÍTICO: Bunge (ZXZ) a Roe/Wigner (ZYZ) ---
    alpha = phi1 - (np.pi / 2.0)
    beta  = Phi
    gamma = phi2 + (np.pi / 2.0)
    
    # 3. Construimos los coeficientes triclínicos C_lmn para esta orientación
    coefs = []
    for l in range(L_max + 1):
        kl = K_l[l]
        if kl == 0.0:
            continue
            
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                # Matriz pequeña de Wigner (ZYZ) evaluada en beta (Inclinación Phi)
                d_mn = _wigner_small_d(l, m, n, beta)
                
                # Fases complejas aplicadas sobre los ángulos ZYZ (Roe)
                # Convención Wigner-D: exp(-i * m * gamma) * d_mn * exp(-i * n * alpha)
                fase = np.exp(-1j * (m * gamma + n * alpha))
                
                # El coeficiente final es la proyección de la semilla del kernel
                c_val = kl * d_mn * fase
                coefs.append((l, m, n, c_val))
                
    return coefs