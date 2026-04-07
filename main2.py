# -*- coding: utf-8 -*-
"""
Script Principal: Simulación de Texturas Complejas (Aditivas).
Componentes: Isotrópica (Azar) + Ideales (Laminación) + Fibras.
Entorno: texturaPy3.10 [2026-03-06]
"""

import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Rotation, Orientation
from orix.crystal_map import Phase
from orix.vector import Miller, Vector3d
from diffpy.structure import Lattice, Structure

# IMPORTACIÓN DE MÓDULOS PROPIOS
import utils_sym
import utils_kernels
import utils_orient
from utils_odf import ODFIsotropic, ODFComponent, ODFFiber
from utils_pf import plot_pfs  # Importamos la función de ploteo múltiple

def main():
    # 1. CONFIGURACIÓN DE FASE Y SIMETRÍAS
    print("--- Configurando Fase: Titanio (HCP) ---")
    samp_sym = utils_sym.obtener_simetria('-1')
    lat = Lattice(2.95, 2.95, 4.68, 90, 90, 120)
    fase_hcp = Phase(name="Titanium", space_group=194, structure=Structure(lattice=lat))
    cryst_sym = fase_hcp.point_group

    # 2. DEFINICIÓN DE LOS KERNELS
    # Kernel agudo para componentes, más ancho para la fibra
    k_comp = utils_kernels.OrientationKernel(fwhm_grados=30, tipo='poussin')
    k_fibra = utils_kernels.OrientationKernel(fwhm_grados=30, tipo='poussin')

    # 3. CONSTRUCCIÓN DE LA ODF POR PARTES
    print("--- Construyendo Textura Mixta ---")
    
    # A. Componente Isotrópica (Fondo de Azar - 20%)
    odf_azar = ODFIsotropic(peso=0.2, crystal_sym=cryst_sym, sample_sym=samp_sym)

    # B. Componentes Ideales (Laminación - 50%)
    ori_basal = Orientation(Rotation.from_euler(np.deg2rad([0, 0, 0])))
    odf_laminacion = ODFComponent(ori_basal, pesos=[0.1], kernels=k_comp, 
                                  crystal_sym=cryst_sym, sample_sym=samp_sym)

    # C. Textura de Fibra 1 (Deformación - 30%)
    eje_c = Miller(hkil=[0, 0, 0, 1], phase=fase_hcp)
    nd = Vector3d([0, 0, 1])
    odf_fibra = ODFFiber(eje_c, nd, peso=0.3, kernel=k_fibra, 
                         crystal_sym=cryst_sym, sample_sym=samp_sym)
                         
    # D. Textura de Fibra 2 (Eje prismático paralelo a RD)
    eje_pris = Miller(hkil=[1, 0, -1, 0], phase=fase_hcp)
    rd = Vector3d([1, 0, 0])
    odf_fibra_pris = ODFFiber(eje_pris, rd, peso=0.4, kernel=k_fibra, 
                              crystal_sym=cryst_sym, sample_sym=samp_sym)

    # E. SUMA TOTAL (Arquitectura Aditiva)
    odf_total = odf_azar + odf_laminacion + odf_fibra + odf_fibra_pris

    # 4. ANÁLISIS CUANTITATIVO
    print("\n--- Resultados del Análisis Mixto ---")
    j_index = odf_total.calc_texture_index(res_grados=5)
    val_basal = odf_total.get_value_at(0, 0, 0)
    vol_basal = odf_total.calc_component_volume(center_euler=[0,0,0], radius_degrees=15)

    print("="*45)
    print(f"1. Índice J (Textura Total): {j_index:.3f}")
    print(f"2. Intensidad en Basal:      {val_basal:.2f} MUD")
    print(f"3. Vol. en esfera basal:     {vol_basal*100:.1f} %")
    print("="*45)

    # 5. VISUALIZACIÓN ODF y PFs
    print("\n--- Generando Gráficos Finales ---")
    odf_total.plot_sections(sections=[0, 30, 60], axis='phi2', res_grados=5)

    planos = [
        Miller(hkil=[0, 0, 0, 1], phase=fase_hcp),
        Miller(hkil=[1, 0, -1, 0], phase=fase_hcp)
    ]
    
    print("\n--- Renderizando Figuras de Polos ---")
    mis_pfs = odf_total.calc_pole_figures(planos, res_grados=5)
    plot_pfs(mis_pfs, titulos=["PF (0001)", "PF (10-10)"], direccion_x='vertical')

    # ==========================================================
    # 6. PERFIL LINEAL DE LA ODF (1D)
    # ==========================================================
    print("\n--- Extrayendo Perfil Lineal de la ODF ---")
    limite_phi1 = odf_total.lims['phi1']
    phi1_vals = np.linspace(0, limite_phi1, 200) 
    
    Phi_fijo = 60
    phi2_fijo = 0
    
    eulers_linea = np.column_stack((
        phi1_vals, 
        np.full_like(phi1_vals, Phi_fijo), 
        np.full_like(phi1_vals, phi2_fijo)
    ))
    
    densidades_linea = odf_total.get_density(eulers_linea, degrees=True)
    
    fig_perfil, ax_perfil = plt.subplots(figsize=(8, 4))
    ax_perfil.plot(phi1_vals, densidades_linea, 'b-', linewidth=2, 
                   label=f'$\Phi={Phi_fijo}^\circ, \phi_2={phi2_fijo}^\circ$')
    
    ax_perfil.set_xlim(0, limite_phi1)
    ax_perfil.set_ylim(bottom=0) 
    
    ax_perfil.set_xlabel('$\phi_1$ (grados)', fontsize=12)
    ax_perfil.set_ylabel('Intensidad (MUD)', fontsize=12)
    ax_perfil.set_title('Perfil 1D de la ODF', fontsize=14)
    
    ax_perfil.grid(True, linestyle='--', alpha=0.7)
    ax_perfil.legend()
    fig_perfil.tight_layout()

    plt.show()
    
    # ==========================================================
    # 7. TEST FOURIER (Matriz Vectorizada)
    # ==========================================================
    print("\n--- TEST DE FOURIER ---")
    coefs = odf_total.calc_fourier_coeffs(L_max=4)
    print(f"Total de armónicos calculados: {len(coefs)}")
    
    # Buscamos la fila donde L=0, m=0, n=0 en la matriz (N, 6)
    mask_000 = (coefs[:, 0] == 0) & (coefs[:, 1] == 0) & (coefs[:, 2] == 0)
    
    if np.any(mask_000):
        # La columna 3 tiene la parte Real
        c000_val = coefs[mask_000][0, 3] 
        print(f"C(0,0,0) (Integral total) = {c000_val:.4f}")
    else:
        print("C(0,0,0) (Integral total) no encontrado o anulado.")

if __name__ == "__main__":
    main()