# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:13:15 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:13:15 2026

@author: mavic
"""

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
    # Esto asegura que la intensidad mínima en toda la ODF sea 0.2 MUD
    odf_azar = ODFIsotropic(peso=0.2, crystal_sym=cryst_sym, sample_sym=samp_sym)

    # B. Componentes Ideales (Laminación - 50%)
    # Definimos la componente Basal [0, 0, 0]
    ori_basal = Orientation(Rotation.from_euler(np.deg2rad([0, 0, 0])))
    odf_laminacion = ODFComponent(ori_basal, pesos=[0.0], kernels=k_comp, 
                                  crystal_sym=cryst_sym, sample_sym=samp_sym)

    # C. Textura de Fibra (Deformación - 30%)
    # Ejemplo: Eje c [0001] paralelo a la dirección normal ND [0, 0, 1]
    eje_c = Miller(hkil=[0, 0, 0, 1], phase=fase_hcp)
    nd = Vector3d([0, 0, 1])
    odf_fibra = ODFFiber(eje_c, nd, peso=0.4, kernel=k_fibra, 
                         crystal_sym=cryst_sym, sample_sym=samp_sym)
    # 3. Fibra 2: Eje prismático paralelo a RD (50%)
    eje_pris = Miller(hkil=[1, 0, -1, 0], phase=fase_hcp)
    rd = Vector3d([1, 0, 0])
    odf_fibra_pris = ODFFiber(eje_pris, rd, peso=0.4, kernel=k_fibra, 
                      crystal_sym=cryst_sym, sample_sym=samp_sym)

    # D. SUMA TOTAL (Arquitectura Aditiva)
    # La suma de pesos (0.2 + 0.5 + 0.3) es 1.0 para mantener la normalización
    odf_total = odf_azar + odf_laminacion + odf_fibra + odf_fibra_pris

    # 4. ANÁLISIS CUANTITATIVO
    print("\n--- Resultados del Análisis Mixto ---")
    j_index = odf_total.calc_texture_index(res_grados=5)
    val_basal = odf_total.get_value_at(0, 0, 0)
    # Calculamos el volumen de la componente basal "pura" (excluyendo azar y fibra lejana)
    vol_basal = odf_total.calc_component_volume(center_euler=[0,0,0], radius_degrees=15)

    print("="*45)
    print(f"1. Índice J (Textura Total): {j_index:.3f}")
    print(f"2. Intensidad en Basal:      {val_basal:.2f} MUD")
    print(f"3. Vol. en esfera basal:     {vol_basal*100:.1f} %")
    print("="*45)

# 5. VISUALIZACIÓN
    print("\n--- Generando Gráficos Finales ---")
    # Secciones de Euler
    odf_total.plot_sections(sections=[0, 30, 60], axis='phi2',res_grados=5)

    # Figuras de Polos (0001) y (10-10)
    planos = [
        Miller(hkil=[0, 0, 0, 1], phase=fase_hcp),
        Miller(hkil=[1, 0, -1, 0], phase=fase_hcp)
    ]
    
    # 1. Calculamos las figuras de polos
    mis_pfs = odf_total.calc_pole_figures(planos, resolution=120)
    
    # --- LA MAGIA DE UNIRLAS EN UNA SOLA FIGURA ---
    # Creamos un "lienzo" con 1 fila y tantas columnas como PFs tengamos
    n_pfs = len(mis_pfs)
    fig, axes = plt.subplots(1, n_pfs, figsize=(6 * n_pfs, 6))
    
    # Si por casualidad calculamos solo 1 plano, lo convertimos en lista para que el loop no falle
    if n_pfs == 1:
        axes = [axes]
        
    # Le pasamos a cada Figura de Polos el recuadro (ax) donde tiene que dibujarse
    for i, pf in enumerate(mis_pfs):
        pf.plot(ax=axes[i])

    plt.tight_layout()
    plt.show()

# ... (código anterior de la sección 5 de visualización de PFs)
    for pf in mis_pfs:
        pf.plot(ax=axes[i])
    plt.tight_layout()

    # ==========================================================
    # 6. PERFIL LINEAL DE LA ODF (1D)
    # ==========================================================
    print("\n--- Extrayendo Perfil Lineal de la ODF ---")
    
    # 1. Definimos el rango de phi1 (usamos el límite dinámico de la ODF)
    limite_phi1 = odf_total.lims['phi1']
    phi1_vals = np.linspace(0, limite_phi1, 200) # 200 puntos para que la curva sea suave
    
    # 2. Fijamos los otros dos ángulos
    Phi_fijo = 60
    phi2_fijo = 0
    
    # 3. Armamos la matriz (N, 3) con los ángulos de Euler
    eulers_linea = np.column_stack((
        phi1_vals, 
        np.full_like(phi1_vals, Phi_fijo), 
        np.full_like(phi1_vals, phi2_fijo)
    ))
    
    # 4. Calculamos las intensidades al instante con tu nueva función
    densidades_linea = odf_total.get_density(eulers_linea, degrees=True)
    
    # 5. Graficamos el resultado
    fig_perfil, ax_perfil = plt.subplots(figsize=(8, 4))
    
    ax_perfil.plot(phi1_vals, densidades_linea, 'b-', linewidth=2, 
                   label=f'$\Phi={Phi_fijo}^\circ, \phi_2={phi2_fijo}^\circ$')
    
    ax_perfil.set_xlim(0, limite_phi1)
    ax_perfil.set_ylim(bottom=0) # La intensidad ODF nunca es negativa
    
    ax_perfil.set_xlabel('$\phi_1$ (grados)', fontsize=12)
    ax_perfil.set_ylabel('Intensidad (MUD)', fontsize=12)
    ax_perfil.set_title('Perfil 1D de la ODF', fontsize=14)
    
    ax_perfil.grid(True, linestyle='--', alpha=0.7)
    ax_perfil.legend()
    fig_perfil.tight_layout()

    # Mostrar todos los gráficos juntos
    plt.show()

if __name__ == "__main__":
    main()
