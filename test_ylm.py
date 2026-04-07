# -*- coding: utf-8 -*-
import numpy as np

try:
    from scipy.special import sph_harm_y
    def sph_harm(m, l, azim, pol): return sph_harm_y(l, m, pol, azim)
except ImportError:
    from scipy.special import sph_harm

from utils_fourier import _wigner_small_d

def test_reduccion_multiples_L():
    # Ángulos de Euler de prueba
    Phi  = np.radians(45.0)  
    phi1 = np.radians(30.0)  
    phi2 = np.radians(60.0)  
    
    print("=========================================================================")
    print(" STRESS TEST: REDUCCIÓN DE WIGNER A ARMÓNICOS ESFÉRICOS (Múltiples L)")
    print("=========================================================================")
    print(f"Ángulos base: Phi = {np.degrees(Phi)}°, phi1 = {np.degrees(phi1)}°, phi2 = {np.degrees(phi2)}°\n")
    
    print(f"{'L':>4} | {'Max Error Muestra (n)':>22} | {'Max Error Cristal (m)':>22} | {'Estado':>8}")
    print("-" * 65)

    # Probamos grados pares, impares y altos
    grados_a_probar = [2, 3, 4, 6, 10, 15, 20]
    
    for L in grados_a_probar:
        factor = np.sqrt(4.0 * np.pi / (2 * L + 1))
        
        alpha = phi1 - np.pi / 2.0
        gamma = phi2 + np.pi / 2.0
        
        max_err_n = 0.0
        max_err_m = 0.0
        
        # Test Muestra (con el parche de fase incorporado)
        for n in range(-L, L + 1):
            d_0n = _wigner_small_d(L, 0, n, np.array([Phi]))[0]
            T_0n = d_0n * np.exp(-1j * n * alpha)
            Y_n = np.conj(sph_harm(n, L, alpha, Phi)) * factor
            Y_n_corregido = Y_n * ((-1.0)**n)  # <--- PARCHE APLICADO
            
            err = np.abs(T_0n - Y_n_corregido)
            if err > max_err_n: max_err_n = err
            
        # Test Cristal (sin parche, pues la matriz nativa ya coincide)
        for m in range(-L, L + 1):
            d_m0 = _wigner_small_d(L, m, 0, np.array([Phi]))[0]
            T_m0 = d_m0 * np.exp(-1j * m * gamma)
            Y_m = np.conj(sph_harm(m, L, gamma, Phi)) * factor
            
            err = np.abs(T_m0 - Y_m)
            if err > max_err_m: max_err_m = err
            
        # Validamos si el error máximo es prácticamente nulo (cero de máquina)
        estado = "✅ OK" if (max_err_n < 1e-13 and max_err_m < 1e-13) else "❌ FAIL"
        
        print(f"{L:4d} | {max_err_n:22.2e} | {max_err_m:22.2e} | {estado:>8}")

if __name__ == '__main__':
    test_reduccion_multiples_L()