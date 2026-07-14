# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
from orix.crystal_map import Phase
from orix.vector import Vector3d, Miller
from orix.quaternion.symmetry import Oh, C1
from diffpy.structure import Lattice, Structure

from utils_so3 import SO3Grid
from utils_general_odf import GeneralizedPoleFigure, reconstruir_odf_nnls
from utils_strain import (
    calc_strain_pole_figures, 
    extraer_strain_experimental, 
    invertir_tensores_por_orientacion, 
    plot_tensor_odf_section, 
    plot_hydrostatic_odf_section,
    invertir_strain_fourier
)

def main():
    print("=========================================================")
    print("🚀 PIPELINE COMPLETO: TEXTURA + INVERSIÓN (DIRECTA Y ARMÓNICA)")
    print("=========================================================")

    # 1. PARÁMETROS DE RED Y FASE
    a_ferrita = 2.866
    lat = Lattice(a_ferrita, a_ferrita, a_ferrita, 90, 90, 90)
    fase_bcc = Phase(name="Ferrita_BCC", space_group=229, structure=Structure(lattice=lat))
    mapeo_planos = {0.91:[3,1,0], 1.017:[2,2,0], 1.175:[2,1,1], 1.44:[2,0,0], 2.03:[1,1,0]}

    ruta = os.path.join("Exp_Data", "SPF_Malamud", "center", "peak*_mar27.txt")
    archivos = glob.glob(ruta)
    pfs_intensidad_raw, pfs_strain_raw = [], []

    print("\n--- Procesando Archivos ---")
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        d_aprox = float(nombre.replace("peak", "").split("_")[0])
        hkl_v = mapeo_planos.get(d_aprox)
        if not hkl_v: continue
            
        plano = Miller(hkl=hkl_v, phase=fase_bcc)
        d_nom = a_ferrita *(1+3050e-6) / np.sqrt(sum([h**2 for h in hkl_v]))
        
        datos = np.loadtxt(archivo, comments="#")
        a_raw = datos[:, 0]          
        b_raw = datos[:, 1]          
        intens_raw = datos[:, 2]
        de_raw = datos[:, 3]
        err_de_raw = datos[:, 4]     
        err_intens_raw = datos[:, 5]
        
        strain_raw = ((de_raw - d_nom) / d_nom) * 1e6
        err_strain_raw = (err_de_raw / d_nom) * 1e6
        
        mask = (~np.isnan(intens_raw)) & (intens_raw > 0) & \
               (np.abs(strain_raw) <= 7000) & (err_strain_raw < 2000)
        
        a, b = a_raw[mask], b_raw[mask]
        intens, strain = intens_raw[mask], strain_raw[mask]
        
        x = np.sin(b) * np.cos(a)
        y = np.sin(b) * np.sin(a)
        z = np.cos(b)
        pts_3d = np.column_stack((x, y, z))
        
        pfs_intensidad_raw.append(GeneralizedPoleFigure(Vector3d(pts_3d), intens, plano))
        pfs_strain_raw.append(GeneralizedPoleFigure(Vector3d(pts_3d), strain, plano))
        print(f" -> {nombre} OK: {hkl_v} ({len(intens)} puntos tras filtro)")

    orden = np.argsort([sum(pf.hkl.hkl[0]**2) for pf in pfs_intensidad_raw])
    pfs_exp_int = [pfs_intensidad_raw[i] for i in orden]
    pfs_exp_str = [pfs_strain_raw[i] for i in orden]
    planos_ord = [pf.hkl for pf in pfs_exp_int]

    # 2. RESOLUCIÓN DE TEXTURA
    grilla = SO3Grid(resolucion_grados=5.0, simetria_cristal=Oh, simetria_muestra=C1)
    odf_nnls = reconstruir_odf_nnls(pfs_exp_int, grilla, Oh, C1, fwhm_grados=10.0)
    pfs_recalc = odf_nnls.calc_pole_figures(planos_ord, res_grados=2.0)

    # ====================================================================
    # 3. EXTRACCIÓN E INVERSIÓN DIRECTA (MÉTODO 1)
    # ====================================================================
    print("\n--- Extrayendo e Invirtiendo Strain Experimental (Directa) ---")
    df_strain = extraer_strain_experimental(odf_nnls, pfs_exp_str, Oh, tolerancia_grados=10.0, umbral_peso=0.01)
    
    if df_strain.empty:
        print("❌ ERROR: El DataFrame está vacío.")
        return

    col_id, col_strain = 'id_ori', 'strain_exp'
    df_validos = df_strain.dropna(subset=[col_strain])
    
    id_max = int(df_validos[col_id].max() + 1)
    tensores_fit = invertir_tensores_por_orientacion(df_validos, id_max)
    
    pesos = odf_nnls.odf_base.pesos
    mask_v = pesos > (np.max(pesos) * 0.01)
    indices_activos = np.where(mask_v)[0]
    
    # Asignamos los tensores de la Inversión Directa a la ODF
    odf_nnls.strain_base = np.zeros((len(pesos), 6))
    for i, idx_grilla in enumerate(indices_activos):
        if i < len(tensores_fit): odf_nnls.strain_base[idx_grilla] = tensores_fit[i]

    pfs_strain_est_dir = calc_strain_pole_figures(odf_nnls, planos_ord, Oh, umbral_peso=0.01)

    # ====================================================================
    # 4. GRÁFICOS INVERSIÓN DIRECTA
    # ====================================================================
    print("\n--- Generando Gráficos de Inversión Directa ---")
    
    # 4.1 Figuras de Polos
    n = len(pfs_exp_int)
    fig_pf, axes_pf = plt.subplots(4, n, figsize=(4*n, 17))
    if n == 1: axes_pf = axes_pf.reshape(4, 1)

    for i in range(n):
        hkl_txt = f"{planos_ord[i].hkl[0]}"
        vmax_exp = np.max(pfs_exp_int[i].intensidades)
        vlim_s = np.percentile(np.abs(pfs_exp_str[i].intensidades), 98)
        
        pfs_exp_int[i].plot(ax=axes_pf[0, i], titulo=f"EXP Int {hkl_txt}", modo='scatter', s=15, max_val=vmax_exp, direccion_x='vertical')
        pfs_recalc[i].plot(ax=axes_pf[1, i], titulo=f"RECALC NNLS", modo='contour', cmap='jet', direccion_x='vertical')
        pfs_exp_str[i].plot(ax=axes_pf[2, i], titulo=f"EXP Strain {hkl_txt}", modo='scatter', s=15, cmap='coolwarm', max_val=vlim_s, min_val=-vlim_s, direccion_x='vertical')
        pfs_strain_est_dir[i].plot(ax=axes_pf[3, i], titulo=f"FIT Strain {hkl_txt}\n(Inv. Directa)", modo='scatter', s=10, cmap='coolwarm', max_val=vlim_s, min_val=-vlim_s, direccion_x='vertical')

    fig_pf.suptitle("Comparación de Figuras de Polos (Inversión Directa)", fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()

    # 4.2 Scatter de Calidad
    strains_exp_vals, strains_calc_vals = [], []
    for _, row in df_validos.iterrows():
        idx_ori = int(row[col_id])
        if idx_ori >= len(tensores_fit): continue
        E = tensores_fit[idx_ori]
        if np.all(E == 0): continue
        
        r, p = row['rho_rad'], row['phi_rad']
        e_calc = (E[0]*(np.cos(p)**2)*(np.sin(r)**2) + E[1]*(np.sin(p)**2)*(np.sin(r)**2) + E[2]*(np.cos(r)**2) +
                  (E[5]/2.0)*np.sin(2*p)*(np.sin(r)**2) + (E[3]/2.0)*np.sin(p)*np.sin(2*r) + (E[4]/2.0)*np.cos(p)*np.sin(2*r)) * 1e6
        strains_exp_vals.append(row[col_strain])
        strains_calc_vals.append(e_calc)

    fig_scat = plt.figure(figsize=(7, 6))
    plt.scatter(strains_exp_vals, strains_calc_vals, color='royalblue', alpha=0.4, s=8, edgecolors='none')
    mn, mx = min(min(strains_exp_vals), min(strains_calc_vals)), max(max(strains_exp_vals), max(strains_calc_vals))
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ajuste Ideal')
    plt.xlabel(r"$\epsilon_{exp}$ [$\mu\epsilon$]"); plt.ylabel(r"$\epsilon_{calc}$ [$\mu\epsilon$]")
    plt.title("Validación (Inversión Directa)", fontsize=14, fontweight='bold'); plt.legend(); plt.grid(True, linestyle=':', alpha=0.7)

    # 4.3 Strain ODFs Directas (Capturamos la figura para re-titular)
    print(" -> Graficando SODF (Directa)...")
    odf_nnls.plot_sections(sections=[0], axis='phi2') # Textura Base
    
    fig_td_dir, _ = plot_tensor_odf_section(odf_gen=odf_nnls, section_val=0.0, axis='phi2')
    fig_td_dir.suptitle("Strain ODF - Componentes (Inversión Directa)", fontsize=16, fontweight='bold', y=0.98)
    
    fig_hd_dir, _ = plot_hydrostatic_odf_section(odf_nnls, section_val=0.0, axis='phi2')
    fig_hd_dir.suptitle("Strain Hidrostático (Inversión Directa)", fontsize=14, fontweight='bold', y=0.96)


    # ====================================================================
    # 5. INVERSIÓN ARMÓNICA GLOBAL (FOURIER - MÉTODO 2)
    # ====================================================================
    print("\n" + "="*55)
    print("--- Extrayendo e Invirtiendo Strain (Global Armónico) ---")
    
    L_max_ajuste = 4 
    tensores_globales = invertir_strain_fourier(df_strain, odf_nnls, L_max=L_max_ajuste)

    # Reemplazamos los tensores viejos en la ODF por los armónicos nuevos
    odf_nnls.strain_base = tensores_globales

    # Recalculamos las figuras de polos con los tensores armónicos
    pfs_strain_est_harm = calc_strain_pole_figures(odf_nnls, planos_ord, Oh, umbral_peso=0.0)

    # ====================================================================
    # 6. GRÁFICOS INVERSIÓN ARMÓNICA
    # ====================================================================
    print("\n--- Generando Gráficos de Inversión Armónica ---")
    
    # 6.1 Figuras de Polos (Comparativa Experimental vs Armónica)
    fig_harm, axes_harm = plt.subplots(2, n, figsize=(4*n, 9))
    if n == 1: axes_harm = axes_harm.reshape(2, 1)

    for i in range(n):
        hkl_txt = f"{planos_ord[i].hkl[0]}"
        vlim_s = np.percentile(np.abs(pfs_exp_str[i].intensidades), 98)
        
        pfs_exp_str[i].plot(ax=axes_harm[0, i], titulo=f"EXP Strain {hkl_txt}", modo='scatter', s=15, cmap='coolwarm', max_val=vlim_s, min_val=-vlim_s, direccion_x='vertical')
        pfs_strain_est_harm[i].plot(ax=axes_harm[1, i], titulo=f"FIT Strain {hkl_txt}\n(Inv. Armónica L={L_max_ajuste})", modo='scatter', s=10, cmap='coolwarm', max_val=vlim_s, min_val=-vlim_s, direccion_x='vertical')

    fig_harm.suptitle("Comparación de Figuras de Polos (Inversión Armónica WLS)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    # 6.2 Strain ODFs Armónicas (Como actualizamos la odf_nnls, tomará los datos globales automáticamente)
    print(" -> Graficando SODF (Armónica)...")
    
    fig_td_harm, _ = plot_tensor_odf_section(odf_gen=odf_nnls, section_val=0.0, axis='phi2')
    fig_td_harm.suptitle(f"Strain ODF - Componentes (Inversión Armónica L={L_max_ajuste})", fontsize=16, fontweight='bold', y=0.98)
    
    fig_hd_harm, _ = plot_hydrostatic_odf_section(odf_nnls, section_val=0.0, axis='phi2')
    fig_hd_harm.suptitle(f"Strain Hidrostático (Inversión Armónica L={L_max_ajuste})", fontsize=14, fontweight='bold', y=0.96)

    # 7. Despliegue de TODAS las ventanas juntas
    print("\n✅ Proceso Finalizado. Mostrando todas las gráficas generadas...")
    plt.show()

if __name__ == "__main__":
    main()