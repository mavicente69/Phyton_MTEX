# -*- coding: utf-8 -*-
"""
Módulo para el cálculo de la Sección Eficaz Macroscópica Coherente.
Entorno: texturaPy3.10
"""
import numpy as np
import pandas as pd
import scipy.special as sp
import utils_fourier

def leer_red_cristalina(filepath):
    """
    Parser del archivo cristalográfico para extraer la matriz, la base, 
    la simetría y los coeficientes de fondo.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    matrix_rows = []
    basis_rows = []
    simetria_str = None
    background_coeffs = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        parts = line.split()
        
        # 1. Detectar Simetría
        if len(parts) == 1 and parts[0].isalpha():
            simetria_str = parts[0].lower()
            continue
        
        # 2. Detectar Coeficientes de Background
        if parts[0].upper() == 'BACKGROUND':
            background_coeffs = [float(p) for p in parts[1:]]
            continue
            
        # 3. Detectar Matriz (3 columnas)
        if len(parts) == 3:
            matrix_rows.append([float(p) for p in parts])
            
        # 4. Detectar Base (5+ columnas)
        elif len(parts) >= 5:
            basis_rows.append([float(p) for p in parts])
            
    A_matrix = np.array(matrix_rows)
    basis = np.array(basis_rows)
    
    return simetria_str, A_matrix, basis, np.array(background_coeffs)

def calcular_seccion_eficaz_coherente(odf, direcciones_haz, lattice_file, lambdas, L_max=20, unidades='barns/celda'):
    """
    Evalúa la sección eficaz macroscópica coherente de forma totalmente embebida.
    
    Inputs:
    - odf: Objeto ODF (Component, Discreta, etc.)
    - direcciones_haz: Dict de nombre_direccion -> (polar, azim)
    - lattice_file: Ruta al archivo TXT con la estructura cristalina
    - lambdas: Array de longitudes de onda a evaluar
    - L_max: L máximo de truncación armónica
    - unidades: 'barns/celda' (microscópica) o 'cm-1' (macroscópica)
    """
    # 1. Extraer simetrías de la ODF provista
    crystal_sym = odf.crystal_sym
    sample_sym = odf.sample_sym
    
    # 2. Chequear o calcular coeficientes de Fourier de la ODF
    if hasattr(odf, 'C_bunge') and odf.C_bunge is not None:
        print("  [INFO] Coeficientes de Fourier (C_bunge) detectados en la ODF. Reutilizando...")
        C_bunge = odf.C_bunge
    else:
        print(f"  [INFO] Calculando coeficientes de Fourier de la ODF (L_max={L_max})...")
        C_bunge = odf.calc_bunge_coeffs(L_max)
        odf.C_bunge = C_bunge 
        
    # 3. Calcular proyectores del cristal y la muestra (A_cryst y B_samp)
    proj_C, proj_S = utils_fourier.calc_symmetry_projectors(L_max, crystal_sym, sample_sym)
    A_cryst, B_samp = utils_fourier.calc_symmetry_coefficients(L_max, proj_C, proj_S)
    
    # 4. Cálculo pre-optimizado de Coeficientes Direccionales (A_lmu)
    print("  [INFO] Evaluando coeficientes direccionales A_lmu...")
    dict_A_lmu = {nd: {} for nd in direcciones_haz}
    a_lmu_list = []
    
    for nombre_dir, (polar, azim) in direcciones_haz.items():
        for l in range(0, L_max + 1, 2):
            if l not in B_samp or B_samp[l].shape[1] == 0: continue
            
            k_lnu = utils_fourier.eval_sym_sph_harm(l, polar, azim, B_samp[l], is_sample=True).flatten()
            mu_max = max([k[1] for k in C_bunge.keys() if k[0] == l] + [-1])
            if mu_max == -1: continue
            
            for mu in range(mu_max + 1):
                A_lmu = 0.0j
                for nu in range(len(k_lnu)):
                    c_val = C_bunge.get((l, mu, nu), 0.0j)
                    A_lmu += c_val * k_lnu[nu]
                    
                if abs(A_lmu) > 1e-10:
                    dict_A_lmu[nombre_dir][(l, mu)] = A_lmu
                    a_lmu_list.append({
                        'Direccion': nombre_dir, 'l': l, 'mu': mu,
                        'Real': A_lmu.real, 'Imag': A_lmu.imag, 'Magnitud': abs(A_lmu)
                    })

    # 5. Configurar la Geometría de Difracción desde el lattice_file
    print(f"  [INFO] Procesando red cristalina desde: {lattice_file}")
    simetria_str, A_matrix, basis, background_coeffs = leer_red_cristalina(lattice_file)
    v0 = np.abs(np.linalg.det(A_matrix))
    A_inv = np.linalg.inv(A_matrix)
    B_matrix = 2 * np.pi * A_inv.T
    
    k_max = 2 * np.pi / np.min(lambdas)
    G_max_posible = 2 * k_max
    b_norm_min = np.min(np.linalg.norm(B_matrix, axis=1))
    Hmax = int(G_max_posible / b_norm_min) + 2
    
    h_arr = np.arange(-Hmax, Hmax + 1)
    H, K, L = np.meshgrid(h_arr, h_arr, h_arr, indexing='ij')
    H, K, L = H.flatten(), K.flatten(), L.flatten()
    mask_0 = (H == 0) & (K == 0) & (L == 0)
    H, K, L = H[~mask_0], K[~mask_0], L[~mask_0]
    
    G_vecs = H[:, None] * B_matrix[0] + K[:, None] * B_matrix[1] + L[:, None] * B_matrix[2]
    G_mags = np.linalg.norm(G_vecs, axis=1)
    
    mask_G = G_mags <= G_max_posible
    G_vecs, G_mags = G_vecs[mask_G], G_mags[mask_G]
    
    positions, b_c, U_iso = basis[:, 1:4], basis[:, 4], basis[:, 5]
    F_G = np.zeros(len(G_mags), dtype=complex)
    for j in range(len(b_c)):
        phase = np.dot(G_vecs, positions[j])
        dw = np.exp(-0.5 * U_iso[j] * G_mags**2)
        F_G += b_c[j] * np.exp(1j * phase) * dw
        
    F2 = np.abs(F_G)**2
    mask_F2 = F2 > 1e-5
    G_vecs, G_mags, F2 = G_vecs[mask_F2], G_mags[mask_F2], F2[mask_F2]
    
    Gz = G_vecs[:, 2]
    theta_G = np.arccos(Gz / G_mags)
    phi_G = np.arctan2(G_vecs[:, 1], G_vecs[:, 0])

    # 6. Escaneo sobre lambdas: Ensamblando B_lmu (Dinámico) x A_lmu
    print(f"  [INFO] Calculando perfiles en unidades de: {unidades}")
    resultados_sigma = {nd: np.zeros(len(lambdas), dtype=np.float64) for nd in direcciones_haz}
    
    for idx_lmd, lmd in enumerate(lambdas):
        k = 2 * np.pi / lmd
        activos = G_mags <= 2 * k
        if not np.any(activos): continue
        
        G_act = G_mags[activos]
        F2_act = F2[activos]
        th_act = theta_G[activos]
        ph_act = phi_G[activos]
        
        # --- MAGIA DE UNIDADES ---
        # 1 Barn = 10^-24 cm^2. El volumen v0 está típicamente en Angstrom^3 (10^-24 cm^3)
        # Factor 0.01 es la conversión natural de las secciones bc dadas en femtometros.
        # Al dividir Barns por v0[A^3], la dimensionalidad resultante es exactamente cm^-1.
        factor_conv = 0.01 
        if unidades == 'cm-1':
            factor_conv = 0.01 / v0
            
        prefactor = (2 * 1.0 * (2 * np.pi)**4) / (v0 * k**3) * factor_conv
        
        for l in range(0, L_max + 1, 2):
            if l not in A_cryst or A_cryst[l].shape[1] == 0: continue
            
            # Evaluación del componente B_lmu(λ)
            P_val = sp.eval_legendre(l, G_act / (2 * k))
            Y_sym = utils_fourier.eval_sym_sph_harm(l, th_act, ph_act, A_cryst[l], is_sample=False)
            Y_c_mu = np.conj(Y_sym)
            W_G = (k / (2 * G_act)) * F2_act * P_val / (2 * l + 1)
            
            B_mu_complex = np.sum(Y_c_mu * W_G, axis=1) * prefactor
            
            # Multiplicación directa en RAM 
            for mu, B_val in enumerate(B_mu_complex):
                for nombre_dir in direcciones_haz:
                    A_val = dict_A_lmu[nombre_dir].get((l, mu), 0.0j)
                    resultados_sigma[nombre_dir][idx_lmd] += np.real(B_val * A_val)
                    
    return resultados_sigma, a_lmu_list