# -*- coding: utf-8 -*-
import numpy as np
from orix.vector import Vector3d
from utils_general_odf import GeneralizedPoleFigure
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt
from orix.quaternion import Orientation

def voigt_a_tensor(v):
    v = np.asarray(v, dtype=float)
    if v.shape == (3, 3): return v
    if v.shape != (6,):
        raise ValueError("El input debe ser una matriz 3x3 o un vector Voigt de 6 elementos.")
    return np.array([
        [v[0],     v[5]/2.0, v[4]/2.0],
        [v[5]/2.0, v[1],     v[3]/2.0],
        [v[4]/2.0, v[3]/2.0, v[2]    ]
    ])

def calc_strain_pole_figures(odf_gen, hkl_list, simetria_cristal, umbral_peso=0.01):
    pesos = odf_gen.odf_base.pesos
    oris = odf_gen.odf_base.orientaciones
    strains = getattr(odf_gen, 'strain_base', np.zeros((oris.size, 6)))
    
    mask_v = pesos > (np.max(pesos) * umbral_peso)
    oris_v, strains_v = oris[mask_v], strains[mask_v]
    E_matrices = np.array([voigt_a_tensor(s) for s in strains_v])
    
    pg = simetria_cristal.point_group if hasattr(simetria_cristal, 'point_group') else simetria_cristal
    pfs_strain = []

    for plano in hkl_list:
        hkl_sym = (pg * plano).unique()
        n_cristal = Vector3d(hkl_sym).data.astype(float)
        n_cristal /= np.linalg.norm(n_cristal, axis=-1, keepdims=True)
        y_all, s_all = [], []

        for i in range(oris_v.size):
            n_lab = ~oris_v[i] * Vector3d(n_cristal)
            v_lab = n_lab.data
            mask_sur = v_lab[:, 2] < 0
            v_lab[mask_sur] = -v_lab[mask_sur]
            
            rho = np.arccos(np.clip(v_lab[:, 2], -1.0, 1.0))
            phi = np.arctan2(v_lab[:, 1], v_lab[:, 0])
            
            E = E_matrices[i]
            e11, e22, e33 = E[0,0], E[1,1], E[2,2]
            e12, e13, e23 = E[0,1], E[0,2], E[1,2]
            
            e_val = (
                e11 * (np.cos(phi)**2) * (np.sin(rho)**2) +
                e22 * (np.sin(phi)**2) * (np.sin(rho)**2) +
                e33 * (np.cos(rho)**2) +
                e12 * np.sin(2*phi) * (np.sin(rho)**2) +
                e23 * np.sin(phi) * np.sin(2*rho) +
                e13 * np.cos(phi) * np.sin(2*rho)
            ) * 1e6
            
            y_all.append(v_lab)
            s_all.append(e_val)

        pfs_strain.append(GeneralizedPoleFigure(Vector3d(np.vstack(y_all)), np.concatenate(s_all), plano))
    return pfs_strain

def extraer_strain_experimental(odf_gen, pfs_strain_exp, simetria_cristal, tolerancia_grados=5.0, umbral_peso=0.01):
    pesos = odf_gen.odf_base.pesos
    oris = odf_gen.odf_base.orientaciones
    mask_v = pesos > (np.max(pesos) * umbral_peso)
    oris_v = oris[mask_v]
    
    pg = simetria_cristal.point_group if hasattr(simetria_cristal, 'point_group') else simetria_cristal
    radio_busqueda = np.sqrt(2.0 - 2.0 * np.cos(np.radians(tolerancia_grados)))
    resultados = []

    for pf_exp in pfs_strain_exp:
        pts_exp = pf_exp.direcciones.data.copy()
        pts_exp[pts_exp[:, 2] < 0] *= -1
        arbol_exp = cKDTree(pts_exp)
        hkl_sym = (pg * pf_exp.hkl).unique()
        n_cristal = Vector3d(hkl_sym).data / np.linalg.norm(Vector3d(hkl_sym).data, axis=-1, keepdims=True)

        for i in range(oris_v.size):
            v_lab = (~oris_v[i] * Vector3d(n_cristal)).data
            v_lab[v_lab[:, 2] < 0] *= -1
            
            for polo in v_lab:
                rho_rad = np.arccos(np.clip(polo[2], -1.0, 1.0))
                phi_rad = np.arctan2(polo[1], polo[0])
                
                indices = arbol_exp.query_ball_point(polo, r=radio_busqueda)
                if len(indices) > 0:
                    resultados.append({
                        'id_ori': i, 
                        'hkl': str(pf_exp.hkl.hkl[0].tolist()),
                        'rho_rad': rho_rad, 
                        'phi_rad': phi_rad, 
                        'strain_exp': np.mean(pf_exp.intensidades[indices])
                    })
    return pd.DataFrame(resultados)

def invertir_tensores_por_orientacion(df_exp, num_orientaciones):
    tensores_ajustados = np.zeros((num_orientaciones, 6))
    granos_evaluados = num_orientaciones
    granos_con_datos = 0
    granos_exitosos = 0
    
    for i in range(num_orientaciones):
        sub = df_exp[df_exp['id_ori'] == i]
        if len(sub) > 0: granos_con_datos += 1
        if len(sub) < 6: continue 
        
        rho, phi = sub['rho_rad'].values, sub['phi_rad'].values
        A = np.column_stack([
            (np.cos(phi)**2) * (np.sin(rho)**2),    
            (np.sin(phi)**2) * (np.sin(rho)**2),    
            np.cos(rho)**2,                         
            np.sin(2*phi) * (np.sin(rho)**2),       
            np.sin(phi) * np.sin(2*rho),            
            np.cos(phi) * np.sin(2*rho)             
        ])
        
        b = sub['strain_exp'].values / 1e6 
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        voigt_tensor = np.array([sol[0], sol[1], sol[2], 2*sol[4], 2*sol[5], 2*sol[3]])
        tensores_ajustados[i] = voigt_tensor
        granos_exitosos += 1
        
    porc_total = (granos_exitosos / granos_evaluados) * 100 if granos_evaluados > 0 else 0
    porc_parcial = (granos_exitosos / granos_con_datos) * 100 if granos_con_datos > 0 else 0
    
    print("\n" + "="*55)
    print("📊 REPORTE DE INVERSIÓN TENSORIAL")
    print("="*55)
    print(f" -> Total orientaciones activas (evaluadas): {granos_evaluados}")
    print(f" -> Orientaciones con al menos 1 polo medido: {granos_con_datos}")
    print(f" -> Tensores invertidos CON ÉXITO (>= 6 polos): {granos_exitosos}")
    print("-" * 55)
    print(f" ⭐ ÉXITO GLOBAL: {porc_total:.1f}% (sobre el total)")
    print(f" ⭐ ÉXITO RELATIVO: {porc_parcial:.1f}% (sobre las que tenían datos)")
    print("="*55 + "\n")
        
    return tensores_ajustados

def plot_tensor_odf_section(odf_gen, section_val=0.0, axis='phi2', res_grados=2.5, tipo='absoluto'):
    """
    Grafica las 6 componentes tensoriales usando ESTRICTAMENTE la misma lógica de
    interpolación IDW y contornos de plot_sections de la ODF nativa.
    """
    if not hasattr(odf_gen, 'strain_base') or odf_gen.strain_base is None:
        print("Error: La ODF no tiene un tensor 'strain_base' asignado.")
        return None, None

    if hasattr(odf_gen, '_expand_tensors_for_idw'):
        odf_gen._expand_tensors_for_idw()

    print(f"\n--- Graficando Strain ODF Continua (Sección {axis}={section_val}°) ---")

    phi1_max_plot = 90 if odf_gen.lims['phi1'] <= 90 else 360
    Phi_max_plot = 90
    phi2_max_plot = 90 

    if axis == 'phi2':
        x, y = np.arange(0, phi1_max_plot + res_grados, res_grados), np.arange(0, Phi_max_plot + res_grados, res_grados)
        X, Y = np.meshgrid(x, y)
        eulers = np.stack([X.ravel(), Y.ravel(), np.full(X.size, section_val)], axis=-1)
        xlabel, ylabel = rf"$\varphi_1$", rf"$\Phi$"
    elif axis == 'phi1':
        x, y = np.arange(0, phi2_max_plot + res_grados, res_grados), np.arange(0, Phi_max_plot + res_grados, res_grados)
        X, Y = np.meshgrid(x, y)
        eulers = np.stack([np.full(X.size, section_val), Y.ravel(), X.ravel()], axis=-1)
        xlabel, ylabel = rf"$\varphi_2$", rf"$\Phi$"
    elif axis == 'Phi':
        x, y = np.arange(0, phi1_max_plot + res_grados, res_grados), np.arange(0, phi2_max_plot + res_grados, res_grados)
        X, Y = np.meshgrid(x, y)
        eulers = np.stack([X.ravel(), np.full(X.size, section_val), Y.ravel()], axis=-1)
        xlabel, ylabel = rf"$\varphi_1$", rf"$\varphi_2$"

    pg = odf_gen.crystal_sym.point_group if hasattr(odf_gen.crystal_sym, 'point_group') else odf_gen.crystal_sym
    ori_targets = Orientation.from_euler(np.radians(eulers), symmetry=pg)
    
    _, e_pred, _ = odf_gen.evaluate_microstructure(ori_targets)
    
    strains_micro = e_pred * 1e6

    nombres_comp = [r"$\epsilon_{11}$", r"$\epsilon_{22}$", r"$\epsilon_{33}$", 
                    r"$2\epsilon_{23}$", r"$2\epsilon_{13}$", r"$2\epsilon_{12}$"]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    max_val = np.percentile(np.abs(strains_micro), 98)
    if max_val < 1e-6: max_val = 1.0 
    niveles = np.linspace(-max_val, max_val, 15)

    for i in range(6):
        ax = axes_flat[i]
        Z = strains_micro[:, i].reshape(X.shape)
        
        cp = ax.contourf(X, Y, Z, levels=niveles, cmap='coolwarm', extend='both')
        
        ax.set_title(nombres_comp[i], fontsize=16, pad=10)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(np.arange(0, X.max()+1, 30))
        ax.set_yticks(np.arange(0, Y.max()+1, 30))
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cp, cax=cbar_ax, label=r'Strain Absoluto [$\mu\epsilon$]')

    plt.suptitle(rf"Strain ODF - Sección {axis} = {section_val}$^\circ$", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.94])
    
    return fig, axes

def plot_hydrostatic_odf_section(odf_gen, section_val=0.0, axis='phi2', res_grados=2.5):
    """
    Grafica la componente hidrostática de la Strain ODF.
    """
    if not hasattr(odf_gen, 'strain_base') or odf_gen.strain_base is None:
        print("Error: La ODF no tiene un tensor 'strain_base' asignado.")
        return None, None

    if hasattr(odf_gen, '_expand_tensors_for_idw'):
        odf_gen._expand_tensors_for_idw()

    print(f"\n--- Graficando Strain ODF Hidrostática (Sección {axis}={section_val}°) ---")

    phi1_max_plot = 90 if odf_gen.lims['phi1'] <= 90 else 360
    Phi_max_plot = 90
    phi2_max_plot = 90 

    if axis == 'phi2':
        x, y = np.arange(0, phi1_max_plot + res_grados, res_grados), np.arange(0, Phi_max_plot + res_grados, res_grados)
        X, Y = np.meshgrid(x, y)
        eulers = np.stack([X.ravel(), Y.ravel(), np.full(X.size, section_val)], axis=-1)
        xlabel, ylabel = rf"$\varphi_1$", rf"$\Phi$"
    elif axis == 'phi1':
        x, y = np.arange(0, phi2_max_plot + res_grados, res_grados), np.arange(0, Phi_max_plot + res_grados, res_grados)
        X, Y = np.meshgrid(x, y)
        eulers = np.stack([np.full(X.size, section_val), Y.ravel(), X.ravel()], axis=-1)
        xlabel, ylabel = rf"$\varphi_2$", rf"$\Phi$"
    elif axis == 'Phi':
        x, y = np.arange(0, phi1_max_plot + res_grados, res_grados), np.arange(0, phi2_max_plot + res_grados, res_grados)
        X, Y = np.meshgrid(x, y)
        eulers = np.stack([X.ravel(), np.full(X.size, section_val), Y.ravel()], axis=-1)
        xlabel, ylabel = rf"$\varphi_1$", rf"$\varphi_2$"

    pg = odf_gen.crystal_sym.point_group if hasattr(odf_gen.crystal_sym, 'point_group') else odf_gen.crystal_sym
    ori_targets = Orientation.from_euler(np.radians(eulers), symmetry=pg)
    
    _, e_pred, _ = odf_gen.evaluate_microstructure(ori_targets)
    
    hydro_strain = ((e_pred[:, 0] + e_pred[:, 1] + e_pred[:, 2]) / 3.0) * 1e6

    fig, ax = plt.subplots(figsize=(7, 6))

    max_val = np.percentile(np.abs(hydro_strain), 98)
    if max_val < 1e-6: max_val = 1.0 
    niveles = np.linspace(-max_val, max_val, 15)

    Z = hydro_strain.reshape(X.shape)
    
    cp = ax.contourf(X, Y, Z, levels=niveles, cmap='coolwarm', extend='both')
    
    ax.set_title(r"$\epsilon_h = (\epsilon_{11} + \epsilon_{22} + \epsilon_{33}) / 3$", fontsize=15, pad=10)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(np.arange(0, X.max()+1, 30))
    ax.set_yticks(np.arange(0, Y.max()+1, 30))
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    fig.colorbar(cp, ax=ax, label=r'Strain Hidrostático [$\mu\epsilon$]')

    plt.suptitle(rf"Strain ODF: Componente Hidrostática (en {axis}={section_val}$^\circ$)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, ax

# =========================================================================================
# INVERSIÓN ARMÓNICA GLOBAL (FOURIER)
# =========================================================================================

def evaluar_base_simetrizada(orientaciones, L_max, crystal_sym, sample_sym=None):
    from utils_fourier import _wigner_small_d, calc_symmetry_projectors, calc_symmetry_coefficients
    
    euler = orientaciones.to_euler().reshape(-1, 3)
    alpha, beta, gamma = euler[:, 0] - np.pi/2, euler[:, 1], euler[:, 2] + np.pi/2
    N_oris = orientaciones.size
    
    proj_C, proj_S = calc_symmetry_projectors(L_max, crystal_sym, sample_sym)
    A_cryst, B_samp = calc_symmetry_coefficients(L_max, proj_C, proj_S)
    
    base_cols = []
    for l in range(L_max + 1):
        if l not in A_cryst or l not in B_samp: continue
        M_l, N_l = A_cryst[l].shape[1], B_samp[l].shape[1]
        if M_l == 0 or N_l == 0: continue
        
        m_act = np.where(np.sum(np.abs(A_cryst[l]), axis=1) > 1e-8)[0] - l
        n_act = np.where(np.sum(np.abs(B_samp[l]), axis=1) > 1e-8)[0] - l
        
        T_tri = {}
        for m in m_act:
            for n in n_act:
                T_tri[(m, n)] = _wigner_small_d(l, m, n, beta) * np.exp(-1j * (m * gamma + n * alpha))
                
        for mu in range(M_l):
            for nu in range(N_l):
                T_sym = np.zeros(N_oris, dtype=complex)
                for m in m_act:
                    Av = A_cryst[l][m+l, mu]
                    if abs(Av) < 1e-10: continue
                    for n in n_act:
                        Bv = np.conj(B_samp[l][n+l, nu])
                        T_sym += Av * T_tri[(m, n)] * Bv
                base_cols.append(T_sym)
                
    return np.column_stack(base_cols) if base_cols else np.empty((N_oris, 0))

def invertir_strain_fourier(df_exp, odf_gen, L_max=4):
    print(f"\n--- Iniciando Inversión Armónica Global (L_max = {L_max}) ---")
    
    df_validos = df_exp.dropna(subset=['strain_exp']).copy()
    rho, phi = df_validos['rho_rad'].values, df_validos['phi_rad'].values
    strain_exp = df_validos['strain_exp'].values / 1e6
    ids_ori = df_validos['id_ori'].values.astype(int)
    
    oris_grilla = odf_gen.odf_base.orientaciones
    oris_exp = oris_grilla[ids_ori]
    pesos_exp = odf_gen.odf_base.pesos[ids_ori] # ODF weights
    
    M = np.column_stack([
        (np.cos(phi)**2) * (np.sin(rho)**2),    # e11
        (np.sin(phi)**2) * (np.sin(rho)**2),    # e22
        np.cos(rho)**2,                         # e33
        np.sin(2*phi) * (np.sin(rho)**2) / 2.0, # e12
        np.sin(phi) * np.sin(2*rho) / 2.0,      # e23
        np.cos(phi) * np.sin(2*rho) / 2.0       # e13
    ])
    
    Psi_exp = evaluar_base_simetrizada(oris_exp, L_max, odf_gen.crystal_sym, odf_gen.sample_sym)
    N_coefs = Psi_exp.shape[1]
    
    # Ensamblaje de Matriz de Diseño A (Complex)
    A = np.zeros((len(df_validos), 6 * N_coefs), dtype=complex)
    for c in range(6):
        A[:, c * N_coefs : (c + 1) * N_coefs] = M[:, c:c+1] * Psi_exp
            
    # Ponderación por Fracción de Volumen ODF (Weighted Least Squares)
    W = np.sqrt(pesos_exp)
    A_pesada = A * W[:, np.newaxis]
    b_pesado = strain_exp * W
    
    print(f" -> Fiteando {6 * N_coefs} coeficientes con {len(df_validos)} puntos...")
    C_opt, _, _, _ = np.linalg.lstsq(A_pesada, b_pesado, rcond=None)
    
    # Reconstrucción
    Psi_grid = evaluar_base_simetrizada(oris_grilla, L_max, odf_gen.crystal_sym, odf_gen.sample_sym)
    tensores_fit = np.zeros((oris_grilla.size, 6))
    for c in range(6):
        tensores_fit[:, c] = np.real(Psi_grid @ C_opt[c * N_coefs : (c + 1) * N_coefs])
        
    print(" ✅ Inversión Armónica Ponderada finalizada.")
    return tensores_fit