# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:01:00 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:01:00 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.vector import Miller, Vector3d
from diffpy.structure import Lattice, Structure

# ACÁ ESTÁ EL CAMBIO 1: Sacamos Symmetry de acá
from orix.quaternion import Rotation, Orientation

# ACÁ ESTÁ EL CAMBIO 2: Importamos D2h para la simetría ortorrómbica
from orix.quaternion.symmetry import D2h, C1

from utils_wimv import reconstruir_odf_wimv
from utils_pf import PoleFigure
from utils_odf import ODFDiscreta

def main():
    print("--- Configurando Fase: Zirconio (HCP) ---")
    lat_zr = Lattice(3.232, 3.232, 5.147, 90, 90, 120)
    fase_zr = Phase(name="Zirconium", space_group=194, structure=Structure(lattice=lat_zr))

    directorio_base = r"C:\Users\mavic\MiguelAngel\KOWARI-HIDRUROS\Viaje a ANSTO\Muestras textura Chapa Zircaloy4\Texturas\Zr_tex_prisms\N0"

    archivos_exp = [
        {"file": "d1_part1_p1_Zr(100).txt", "hkil": [1, 0, -1, 0]},
        {"file": "d1_part2_p1_Zr(002).txt", "hkil": [0, 0, 0, 2]},
        {"file": "d1_part2_p2_Zr(101).txt", "hkil": [1, 0, -1, 1]},
        {"file": "d2_part1_p1_Zr(102).txt", "hkil": [1, 0, -1, 2]},
        {"file": "d2_part3_p1_Zr(110).txt", "hkil": [1, 1, -2, 0]}
    ]

    mis_pfs_originales = []

    print("--- Cargando y procesando datos ASCII ---")
    for info in archivos_exp:
        try:
            ruta = os.path.join(directorio_base, info["file"])
            datos = np.loadtxt(ruta, skiprows=4)
        except Exception as e:
            continue
            
        psi, phi, intensidades = datos[:, 0], datos[:, 1], datos[:, 2] 
        angulos_euler_rad = np.radians(np.column_stack((phi, 90 - psi, np.zeros_like(phi))))
        g = ~Rotation.from_euler(angulos_euler_rad)

        mis_pfs_originales.append(PoleFigure(
            direcciones=g * Vector3d.zvector(), 
            intensidades=intensidades, 
            hkl=Miller(hkil=info["hkil"], phase=fase_zr)
        ))

    print("\n--- Aplicando Rotación de Alineación ---")
    rot_correccion = Orientation.from_axes_angles(Vector3d([0, 1, 0]), np.radians(90))
    mis_pfs_corregidas = [pf.rotate(rot_correccion) for pf in mis_pfs_originales]

    # 1. WIMV
    # ACÁ ESTÁ EL CAMBIO 3: Asignamos directamente D2h sin usar el texto 'mmm'
    simetria_chapa = D2h
    simetria_triclinic = C1
    ss=C1
    oris_wimv, pesos_wimv = reconstruir_odf_wimv(
        lista_pfs_exp=mis_pfs_corregidas, 
        fase_cristal=fase_zr,
        simetria_muestra=ss, # ¡Acá le pasamos la simetría!
        resolucion_grados=10,
        iteraciones=10
    )
    
    # 2. INSTANCIACIÓN DE TU CLASE
    print("\n--- Recalculando Figuras de Polos desde tu clase ODFDiscreta ---")
    mi_odf = ODFDiscreta(orientaciones=oris_wimv, pesos=pesos_wimv, crystal_sym=fase_zr, sample_sym=ss)
    

    # =========================================================
    # 8. VISUALIZACIÓN DE LA ODF (SECCIONES DE EULER)
    # =========================================================
    print("\n--- Generando Secciones de la ODF ---")
    
    # Para el Zirconio (Hexagonal), la región fundamental de phi2 llega hasta 60°.
    # Los cortes clásicos para ver las fibras basales y prismáticas son 0° y 30°.
    # Llamamos a TU función nativa con la nueva sintaxis:
    mi_odf.plot_sections(sections=[0, 30, 60 ,90, 120, 150, 180], axis='phi1', res_grados=5)

    plt.tight_layout()
    plt.show() # <- Este plt.show() final mostrará todas las ventanas juntas
    
 
    # 3. LLAMADA A TU FUNCIÓN NATIVA
    planos_medidos = [pf.hkl for pf in mis_pfs_corregidas]
    pfs_recalculadas = mi_odf.calc_pole_figures(lista_hkl=planos_medidos)

    # 4. GRÁFICOS
    print("\n--- Generando Comparativa EXP vs CALC ---")
    n_pfs = len(mis_pfs_corregidas)
    fig2, axes2 = plt.subplots(2, n_pfs, figsize=(4.5 * n_pfs, 9))
    
    for i in range(n_pfs):
        hkl_str = archivos_exp[i]["file"].split('_')[-1].replace('.txt', '')
        mis_pfs_corregidas[i].plot(ax=axes2[0, i], cmap='jet')
        axes2[0, i].set_title(f"EXP {hkl_str}", fontsize=12, fontweight='bold')
        
        pfs_recalculadas[i].plot(ax=axes2[1, i], cmap='jet')
        axes2[1, i].set_title(f"CALC {hkl_str}", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    main()