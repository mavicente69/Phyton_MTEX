# -*- coding: utf-8 -*-
"""
Script Principal - Reconstrucción de Texturas desde PFs Experimentales
Entorno: texturaPy3.10
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.vector import Miller, Vector3d
from diffpy.structure import Lattice, Structure
from orix.quaternion import Rotation, Orientation
from orix.quaternion.symmetry import D2h, C1

from utils_wimv import reconstruir_odf_wimv
# ¡IMPORTAMOS TU FUNCIÓN GLOBAL ACÁ!
from utils_pf import PoleFigure, plot_pf_comparison 
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

    # =========================================================
    # VISUALIZACIÓN 1: PFs EXPERIMENTALES RAW (COMO VIENEN)
    # =========================================================
    print("\n--- Graficando PFs experimentales originales (RAW) ---")
    n_pfs = len(mis_pfs_corregidas)
    fig0, axes0 = plt.subplots(1, n_pfs, figsize=(4.5 * n_pfs, 4.5))
    if n_pfs == 1: axes0 = [axes0]
    
    for i, pf in enumerate(mis_pfs_corregidas):
        hkl_str = ''.join([str(int(x)) for x in pf.hkl.hkil.flatten()])
        pf.plot(ax=axes0[i], cmap='jet', direccion_x='vertical')
        axes0[i].set_title(f"RAW EXP {{{hkl_str}}}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show(block=False) 
    plt.pause(0.1)

    # =========================================================
    # RECONSTRUCCIÓN WIMV
    # =========================================================
    simetria_chapa = D2h
    simetria_triclinic = C1
    ss = C1
    
    oris_wimv, pesos_wimv = reconstruir_odf_wimv(
        lista_pfs_exp=mis_pfs_corregidas, 
        fase_cristal=fase_zr,
        simetria_muestra=ss, 
        resolucion_grados=5.0,  # Resolución ajustada a 5 grados para evitar aplastar picos
        iteraciones=25          # Más iteraciones para subir la aguja
    )
    
    print("\n--- Instanciando clase ODFDiscreta ---")
    mi_odf = ODFDiscreta(orientaciones=oris_wimv, pesos=pesos_wimv, crystal_sym=fase_zr.point_group, sample_sym=ss)    

    # =========================================================
    # VISUALIZACIÓN 2: SECCIONES DE LA ODF (Eje phi2)
    # =========================================================
    print("\n--- Generando Secciones de la ODF ---")
    cortes_hexagonales = [0, 15, 30, 45, 60]
    
    mi_odf.plot_sections(
        sections=cortes_hexagonales, 
        axis='phi2',          
        res_grados=5.0,       
        cmap='jet'
    )

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    
    # =========================================================
    # RECALCULAR Y COMPARAR PFs USANDO TU FUNCIÓN NATIVA
    # =========================================================
    print("\n--- Generando Comparativa EXP (Escalada) vs CALC ---")
    planos_medidos = [pf.hkl for pf in mis_pfs_corregidas]
    
    # Recalculamos con buena resolución para evitar bordes blancos
    pfs_recalculadas = mi_odf.calc_pole_figures(lista_hkl=planos_medidos, resolution=20)

    # Armamos unos títulos limpios para la función comparativa
    nombres_planos = [f"Plano {{{''.join([str(int(x)) for x in pf.hkl.hkil.flatten()])}}}" for pf in mis_pfs_corregidas]

    # Llamamos a tu función, que automáticamente gestiona los pares "Entrada/Salida" y unifica la escala por hkl
    plot_pf_comparison(
        pfs_in=mis_pfs_corregidas,
        pfs_out=pfs_recalculadas,
        titulos=nombres_planos,
        suptitle="Comparativa de Figuras de Polos (WIMV)",
        unificar_escala='hkl',
        direccion_x='vertical'
    )
    
    # Este plt.show final frena el script para que veas todas las ventanas
    plt.show() 

    # =========================================================
    # CUANTIFICACIÓN AUTOMÁTICA DE LA TEXTURA (ZIRCALOY)
    # =========================================================
    print("\n=========================================================")
    print("📈 REPORTE CUANTITATIVO DE LA TEXTURA (MÉTODO WIMV)")
    print("=========================================================")
    
    # 1. J-Index (Severidad Global de la Textura)
    # Un material isotrópico tiene J=1. Texturas de laminación fuertes suelen dar J=4 a 15+
    j_index = mi_odf.calc_texture_index(res_grados=5.0)
    print(f" -> J-Index (Fuerza global de la textura) : {j_index:.2f}")
    
    # 2. Fracciones de Volumen de Componentes Claves
    print("\n -> Fracciones de Volumen (Radio de integración esférica: ±15.0°):")
    radio_busqueda = 15.0
    
    # A. Fibra Basal Inclinada (Típica de deformación por laminación, ~30° hacia la dirección transversal)
    centro_basal_inclinado = [0, 30, 0]
    vol_basal_inc = mi_odf.calc_component_volume(
        center_euler=centro_basal_inclinado, radius_degrees=radio_busqueda, res_grados=5.0
    )
    print(f"    * Fibra Basal Inclinada {centro_basal_inclinado} : {vol_basal_inc * 100:>6.2f} %")
    
    # B. Polo Basal Normal (Eje C apuntando exactamente al plano de la normal, ND)
    # En chapas muy deformadas en frío, este número suele ser bajísimo.
    centro_basal_normal = [0, 0, 0]
    vol_basal_norm = mi_odf.calc_component_volume(
        center_euler=centro_basal_normal, radius_degrees=radio_busqueda, res_grados=5.0
    )
    print(f"    * Polo Basal Normal     {centro_basal_normal} : {vol_basal_norm * 100:>6.2f} %")
    
    # C. Fibra Prismática (Ejes C acostados en el plano de la chapa, Phi=90°)
    centro_prismatica = [0, 90, 30] 
    vol_prism = mi_odf.calc_component_volume(
        center_euler=centro_prismatica, radius_degrees=radio_busqueda, res_grados=5.0
    )
    print(f"    * Fibra Prismática      {centro_prismatica} : {vol_prism * 100:>6.2f} %")
    
    print("=========================================================\n")
if __name__ == "__main__":
    main()