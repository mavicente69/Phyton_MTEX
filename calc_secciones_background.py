# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:50:04 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Funciones para calcular las componentes de atenuación: Absorción,
Dispersión Incoherente y Dispersión Inelástica.
"""
import numpy as np
from calc_sigma_elastic_coherent import calcular_seccion_eficaz_coherente, leer_red_cristalina

def calcular_background_total(lambdas, background_coeffs, lambda_ref=1.8):
    """
    Calcula la atenuación total del background para un vector de lambdas.
    Sigma = C_abs * (lambda / lambda_ref) + C_inc + C_inel * (lambda_ref / lambda)^2
    """
    c_abs, c_inc, c_inel = background_coeffs
    
    # 1. Absorción: escala lineal con lambda
    sigma_abs = c_abs * (lambdas / lambda_ref)
    
    # 2. Incoherente: constante
    sigma_inc = np.full_like(lambdas, c_inc)
    
    # 3. Inelástica: escala con 1/lambda^2 aprox
    sigma_inel = c_inel * (lambda_ref / lambdas)**2
    
    return sigma_abs + sigma_inc + sigma_inel



def evaluar_atenuacion_total(odf, nombre_dir, lambdas, lattice_file):
    """
    Calcula la suma de todas las componentes.
    """
    simetria_str, A_matrix, basis = leer_red_cristalina(lattice_file)
    sigma_coh = calcular_seccion_eficaz_coherente(odf, nombre_dir, lambdas, lattice_file)
    sigma_bg = calcular_background_total(lambdas, coeffs_bg)
    return sigma_coh + sigma_bg
