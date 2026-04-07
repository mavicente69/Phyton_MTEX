# -*- coding: utf-8 -*-
"""
Prueba Integral de Fourier: Equivalencia de Bases de Simetría (Bunge)
Entorno: texturaPy3.10
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from orix.quaternion import Orientation, symmetry
from orix.vector import Miller
from orix.crystal_map import Phase
from utils_kernels import OrientationKernel
from utils_odf import ODFComponent, ODFFourier, ODFBunge
from utils_fourier import calc_symmetry_projectors, calc_symmetry_coefficients, get_bunge_coefs
from utils_pf import PoleFigure, plot_pfs

def test_fourier_bases():
    print("=========================================================")
    print("🧪 VALIDACIÓN DE TEOREMA DE BUNGE: BASES EQUIVALENTES")
    print("=========================================================")

    # Restauramos tu simetría Hexagonal original
    cs = symmetry.D6 
    ss = None            
    
    phi1, Phi, phi2 =17,68.0, -17.0
    print(f" -> Orientación central : Euler ({phi1}°, {Phi}°, {phi2}°)")
    ori = Orientation.from_euler(np.radians([[phi1, Phi, phi2]]), symmetry=cs)

    FWHM = 30.0
    TIPO_KERNEL = 'poussin'
    print(f" -> Kernel              : {TIPO_KERNEL.capitalize()} (FWHM = {FWHM}°)")
    kernel = OrientationKernel(tipo=TIPO_KERNEL, fwhm_grados=FWHM)

    odf_sintetica = ODFComponent(orientaciones=ori, pesos=[1.0], kernels=kernel, crystal_sym=cs, sample_sym=ss)

    L_MAX = 20  
    print(f"\n -> 1. Extrayendo Fourier Triclínico y Simetrizado clásico (L_max={L_MAX})...")
    coefs_sym_mn, coefs_tri = odf_sintetica.calc_fourier_coeffs(L_MAX, return_triclinic=True)

    print(" -> 2. Extrayendo Base Irreducible de Bunge (Mu, Nu)...")
    proj_C, proj_S = calc_symmetry_projectors(L_MAX, cs, ss)
    A_cryst, B_samp = calc_symmetry_coefficients(L_MAX, proj_C, proj_S)
    coefs_bunge = get_bunge_coefs(coefs_tri, L_MAX, A_cryst, B_samp)

    # =========================================================
    # REPORTE DE COEFICIENTES POR PANTALLA
    # =========================================================
    columnas = ['L', 'm', 'n', 'Real', 'Imag', 'Modulo']
    df_tri = pd.DataFrame(coefs_tri, columns=columnas)
    df_sym = pd.DataFrame(coefs_sym_mn, columns=columnas)
    for df in [df_tri, df_sym]:
        for col in ['L', 'm', 'n']: df[col] = df[col].astype(int)

    print("\n=========================================================")
    print(f"🔬 TAMAÑO DE LA INFORMACIÓN:")
    print(f" -> Espacio Triclínico (L,m,n)   : {len(df_tri)} coeficientes")
    print(f" -> Espacio Simetrizado (L,m,n)  : {len(df_sym)} coeficientes activos")
    print(f" -> Base Irreducible (L,Mu,Nu)   : {len(coefs_bunge)} coeficientes independientes")
    print("=========================================================")
    
    print("\n--- BASE IRREDUCIBLE DE BUNGE (Mu, Nu) ---")
    print("  L | Mu | Nu |      Real      |    Imaginario  |     Módulo   ")
    print("---------------------------------------------------------------")
    contador = 0
    for (l, mu, nu), val in coefs_bunge.items():
        if abs(val) > 1e-4:
            modulo = abs(val)
            print(f" {l:2d} | {mu:2d} | {nu:2d} | {val.real:12.6f} | {val.imag:12.6f} | {modulo:12.6f}")
            contador += 1
            if contador >= 15: break 
    print("=========================================================\n")

    # =========================================================
    # INSTANCIAMOS LOS 3 MODELOS
    # =========================================================
    odf_original = odf_sintetica
    odf_fourier_mn = ODFFourier(coefs_sym_mn, crystal_sym=cs, sample_sym=ss)
    odf_fourier_munu = ODFBunge(coefs_bunge, crystal_sym=cs, sample_sym=ss)

    # =========================================================
    # PERFIL 1D A LO LARGO DE PHI (0 a 90)
    # =========================================================
    print(" -> Evaluando perfil 1D a lo largo de Phi (0° a 90°)...")
    Phi_grados = np.linspace(0, 90, 200)
    eulers_linea = np.radians(np.column_stack((np.full_like(Phi_grados, phi1), Phi_grados, np.full_like(Phi_grados, phi2))))
    oris_linea = Orientation.from_euler(eulers_linea, symmetry=cs)

    perfil_1 = odf_original.evaluate(oris_linea)
    perfil_2 = odf_fourier_mn.evaluate(oris_linea)
    perfil_3 = odf_fourier_munu.evaluate(oris_linea)

    print(f"\n  Intensidad Máxima (Pico Central en {Phi}°):")
    print(f"  * 1. ODF Original            : {np.max(perfil_1):.4f} MUD")
    print(f"  * 2. ODF Fourier (m, n)      : {np.max(perfil_2):.4f} MUD")
    print(f"  * 3. ODF Bunge   (Mu, Nu)    : {np.max(perfil_3):.4f} MUD")

    # Gráfico del Perfil 1D
    plt.figure(figsize=(10, 6))
    plt.plot(Phi_grados, perfil_1, 'b-', linewidth=4, alpha=0.5, label=f'Original Exacta ({TIPO_KERNEL.capitalize()})')
    plt.plot(Phi_grados, perfil_2, 'r--', linewidth=2.5, label=f'Fourier Clásico (m, n)')
    plt.plot(Phi_grados, perfil_3, 'g:', linewidth=2.5, label=f'Fourier Bunge (Mu, Nu)')
    
    plt.axvline(Phi, color='gray', linestyle=':', label=f'Centro de Componente ({Phi}°)')
    plt.title(f"Perfil 1D de Textura - Comparativa de Bases Analíticas (L_max={L_MAX})", fontsize=14)
    plt.xlabel("Ángulo $\Phi$ (grados)", fontsize=12)
    plt.ylabel("Intensidad ODF (MUD)", fontsize=12)
    plt.xlim(0, 90)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show(block=False) 

    # =========================================================
    # VISUALIZACIÓN GRÁFICA COMPARATIVA (SECCIONES)
    # =========================================================
    print("\n📊 GRAFICANDO LOS TRES MÉTODOS...")
    secciones = [0, 15, 30, 45, 60]
    
    print(" [1/3] Graficando Método 1: Original Exacta")
    odf_original.plot_sections(sections=secciones, res_grados=2.5)
    
    print(" [2/3] Graficando Método 2: Base Wigner Clásica (m,n)")
    odf_fourier_mn.plot_sections(sections=secciones, res_grados=2.5)
    
    print(" [3/3] Graficando Método 3: Base Irreducible de Bunge (Mu,Nu)")
    odf_fourier_munu.plot_sections(sections=secciones, res_grados=2.5)

    # =========================================================
    # FIGURAS DE POLOS COMPARATIVAS (LOS 3 MÉTODOS)
    # =========================================================
    print("\n🌍 [4/4] GENERANDO FIGURAS DE POLOS COMPARATIVAS...")
    fase_hex = Phase(name="Hexagonal", space_group=194)
    basal = Miller(hkil=[0, 0, 0, 1], phase=fase_hex)
    prisma = Miller(hkil=[1, 0, -1, 0], phase=fase_hex)

    res_pf = 5  # Resolución en grados para orix (5 es un buen balance velocidad/calidad)

    print(" -> Proyectando Polo Basal (0001)...")
    pf_basal_orig = odf_original.calc_pole_figures([basal], res_grados=res_pf)[0]
    pf_basal_mn   = odf_fourier_mn.calc_pole_figures([basal], res_grados=res_pf)[0]
    pf_basal_munu = odf_fourier_munu.calc_pole_figures([basal], res_grados=res_pf)[0]

    print(" -> Proyectando Polo Prismático (10-10)...")
    pf_prisma_orig = odf_original.calc_pole_figures([prisma], res_grados=res_pf)[0]
    pf_prisma_mn   = odf_fourier_mn.calc_pole_figures([prisma], res_grados=res_pf)[0]
    pf_prisma_munu = odf_fourier_munu.calc_pole_figures([prisma], res_grados=res_pf)[0]

    # Graficamos la comparativa del Polo Basal
    plot_pfs(
        [pf_basal_orig, pf_basal_mn, pf_basal_munu], 
        titulos=["Basal (0001) - Original", "Basal (0001) - Wigner (m,n)", "Basal (0001) - Bunge (\u03bc,\u03bd)"], 
        direccion_x='vertical'
    )
    
    # Graficamos la comparativa del Polo Prismático
    plot_pfs(
        [pf_prisma_orig, pf_prisma_mn, pf_prisma_munu], 
        titulos=["Prisma (10-10) - Original", "Prisma (10-10) - Wigner", "Prisma (10-10) - Bunge"], 
        direccion_x='vertical'
    )



if __name__ == "__main__":
    test_fourier_bases()