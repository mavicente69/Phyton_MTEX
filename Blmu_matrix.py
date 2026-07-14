# -*- coding: utf-8 -*-
"""
Evaluador de matrices B_{l\mu}^{CS}(lambda) para Tomografía de Neutrones.
Entorno: texturaPy3.10
"""
import numpy as np
import scipy.special as sp
import pandas as pd
import os

# Importamos tu motor de Fourier y módulo de simetrías
import utils_fourier
import utils_sym

# ====================================================================
# PARSER DEL ARCHIVO CRISTALOGRÁFICO
# ====================================================================
def leer_red_cristalina(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    matrix_rows = []
    basis_rows = []
    simetria_str = 'm-3m' # Default de seguridad
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        parts = line.split()
        
        # Detectamos la simetría directamente del txt
        if len(parts) == 1 and parts[0].isalpha():
            simetria_str = parts[0].lower()
            continue
            
        if len(parts) == 3:
            matrix_rows.append([float(p) for p in parts])
        elif len(parts) >= 5:
            # Asumimos formato: ID x y z b_c U_iso
            basis_rows.append([float(p) for p in parts])
            
    A_matrix = np.array(matrix_rows)
    basis = np.array(basis_rows)
    return simetria_str, A_matrix, basis

# ====================================================================
# MOTOR PRINCIPAL
# ====================================================================
def calcular_B_lmu(lattice_file, output_file, L_max, lambda_min, lambda_max, lambda_steps):
    print(f"Leyendo archivo: {lattice_file}")
    simetria_str, A_matrix, basis = leer_red_cristalina(lattice_file)
    
    # 1. Geometría del espacio Real y Recíproco
    v0 = np.abs(np.linalg.det(A_matrix))
    A_inv = np.linalg.inv(A_matrix)
    B_matrix = 2 * np.pi * A_inv.T # Matriz recíproca
    
    print(f"Volumen de la celda v0: {v0:.4f} A^3")
    
    # 2. Generar malla de vectores G (Limitada por el lambda mínimo)
    k_max = 2 * np.pi / lambda_min
    G_max_posible = 2 * k_max
    
    # Estimamos el índice de Miller máximo necesario
    b_norm_min = np.min(np.linalg.norm(B_matrix, axis=1))
    Hmax = int(G_max_posible / b_norm_min) + 2
    
    h_arr = np.arange(-Hmax, Hmax + 1)
    H, K, L = np.meshgrid(h_arr, h_arr, h_arr, indexing='ij')
    H, K, L = H.flatten(), K.flatten(), L.flatten()
    
    # Remover el origen (0,0,0)
    mask_0 = (H == 0) & (K == 0) & (L == 0)
    H, K, L = H[~mask_0], K[~mask_0], L[~mask_0]
    
    G_vecs = H[:, None] * B_matrix[0] + K[:, None] * B_matrix[1] + L[:, None] * B_matrix[2]
    G_mags = np.linalg.norm(G_vecs, axis=1)
    
    # Filtrar estrictamente a la Esfera de Ewald máxima
    mask_G = G_mags <= G_max_posible
    G_vecs = G_vecs[mask_G]
    G_mags = G_mags[mask_G]
    
    # 3. Calcular Factores de Estructura F(G)
    positions = basis[:, 1:4]
    b_c = basis[:, 4]
    U_iso = basis[:, 5] # Parámetro de Debye-Waller
    
    F_G = np.zeros(len(G_mags), dtype=complex)
    for j in range(len(b_c)):
        phase = np.dot(G_vecs, positions[j])
        dw = np.exp(-0.5 * U_iso[j] * G_mags**2)
        F_G += b_c[j] * np.exp(1j * phase) * dw
        
    F2 = np.abs(F_G)**2
    
    # Limpiar ausencias sistemáticas (e.g., redes FCC/BCC)
    mask_F2 = F2 > 1e-5
    G_vecs = G_vecs[mask_F2]
    G_mags = G_mags[mask_F2]
    F2 = F2[mask_F2]
    
    print(f"Vectores G activos encontrados: {len(G_mags)}")
    
    # 4. Obtener coordenadas esféricas de G
    Gz = G_vecs[:, 2]
    theta_G = np.arccos(Gz / G_mags) # Polar
    phi_G = np.arctan2(G_vecs[:, 1], G_vecs[:, 0]) # Azimutal
    
    # 5. Obtener los proyectores cristalográficos
    mapa_simetrias = {
        'cubic': 'm-3m',       
        'hexagonal': '6/mmm', 
        'tetragonal': '4/mmm',
        'orthorhombic': 'mmm'
    }
    grupo_puntual = mapa_simetrias.get(simetria_str, 'm-3m')
    print(f"Calculando proyectores de simetría '{grupo_puntual}' hasta L_max={L_max}...")
    
    crystal_sym = utils_sym.obtener_simetria(grupo_puntual)
    proj_C, proj_S = utils_fourier.calc_symmetry_projectors(L_max, crystal_sym, sample_sym=None)
    A_cryst, _ = utils_fourier.calc_symmetry_coefficients(L_max, proj_C, proj_S)
    
    # --- IDENTIFICAMOS LAS LLAVES PARA PRE-LLENARLAS ---
    l_mu_activos = []
    for l in range(0, L_max + 1, 2):
        if l in A_cryst and A_cryst[l].shape[1] > 0:
            for mu in range(A_cryst[l].shape[1]):
                l_mu_activos.append((l, mu))
    
    # 6. Evaluar la sumatoria sobre Lambda
    lambda_arr = np.linspace(lambda_min, lambda_max, lambda_steps)
    resultados = []
    
    print("Iniciando escaneo en lambda...")
    for lmd in lambda_arr:
        k = 2 * np.pi / lmd
        fila_resultado = {'lambda_A': lmd}
        
        # EL TRUCO ESTÁ AQUÍ: Pre-llenamos todas las columnas con 0.0
        # Así, garantizamos que las filas "vacías" existan en el DataFrame.
        for l, mu in l_mu_activos:
            fila_resultado[f'B_{l}_{mu}_Real'] = 0.0
            fila_resultado[f'B_{l}_{mu}_Imag'] = 0.0
            
        # Filtro de la función Heaviside: G <= 2k
        activos = G_mags <= 2 * k
        if not np.any(activos):
            # Si no hay difracción, agregamos la fila de ceros y pasamos al siguiente lambda.
            resultados.append(fila_resultado)
            continue
            
        G_act = G_mags[activos]
        F2_act = F2[activos]
        th_act = theta_G[activos]
        ph_act = phi_G[activos]
        
        n_densidad = 1.0 
        prefactor = (2 * n_densidad * (2 * np.pi)**4) / (v0 * k**3)
        
        for l in range(0, L_max + 1, 2):
            if l not in A_cryst or A_cryst[l].shape[1] == 0:
                continue
                
            P_val = sp.eval_legendre(l, G_act / (2 * k))
            Y_sym = utils_fourier.eval_sym_sph_harm(l, th_act, ph_act, A_cryst[l], is_sample=False)
            Y_c_mu = np.conj(Y_sym)
            
            W_G = (k / (2 * G_act)) * F2_act * P_val / (2 * l + 1)
            
            B_mu_complex = np.sum(Y_c_mu * W_G, axis=1) * prefactor * 0.01
            
            # Actualizamos los ceros con los valores reales calculados
            for mu, valor in enumerate(B_mu_complex):
                fila_resultado[f'B_{l}_{mu}_Real'] = np.real(valor)
                fila_resultado[f'B_{l}_{mu}_Imag'] = np.imag(valor)
                
        resultados.append(fila_resultado)
        
    # 7. Guardar y exportar
    df = pd.DataFrame(resultados)
    df.to_csv(output_file, index=False)
    print(f"¡Cálculo finalizado! Datos guardados en: {output_file}")

if __name__ == '__main__':
    pass