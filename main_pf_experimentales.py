# -*- coding: utf-8 -*-
"""
Script Principal - Reconstrucción de Texturas desde PFs Experimentales
Soporte Multi-Fase: Zirconio (HCP) e Hidruro de Zirconio Delta (FCC)
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
from utils_pf import PoleFigure, plot_pf_comparison 
from utils_odf import ODFDiscreta, ODFFourier, ODFBunge

def main():
    # =========================================================
    # 1. SELECCIÓN DE MATERIAL A EVALUAR 
    # =========================================================
    MATERIAL = "Zr"  # Cambiá a "Zr" para evaluar la chapa de Zircaloy
    
    if MATERIAL == "Zr":
        print("--- Configurando Fase: Zirconio (HCP) (Convención X || a*) ---")
        lat_fase = Lattice(3.232, 3.232, 5.147, 90, 90, 120)
        
        # Parche de alineación geométrica X || a*
        ang = np.radians(-30)
        R_z = np.array([
            [np.cos(ang), -np.sin(ang), 0],
            [np.sin(ang),  np.cos(ang), 0],
            [0,            0,           1]
        ])
        lat_fase.base = lat_fase.base @ R_z.T
        fase_activa = Phase(name="Zirconium", space_group=194, structure=Structure(lattice=lat_fase))
        
        directorio_base = r"C:\Users\mavic\MiguelAngel\Texture Phyton\Exp_Data\Zr\N2"
        directorio_base = r"C:\Users\mavic\MiguelAngel\Texture Phyton\Exp_Data\Zr\N0"
        archivos_exp = [
            {"file": "d1_part1_p1_Zr(100).txt", "hkil": [1, 0, -1, 0]},
            {"file": "d1_part2_p1_Zr(002).txt", "hkil": [0, 0, 0, 2]},
            {"file": "d1_part2_p2_Zr(101).txt", "hkil": [1, 0, -1, 1]},
            {"file": "d2_part1_p1_Zr(102).txt", "hkil": [1, 0, -1, 2]},
            {"file": "d2_part3_p1_Zr(110).txt", "hkil": [1, 1, -2, 0]}
        ]
        
    elif MATERIAL == "ZrH":
        print("--- Configurando Fase: Hidruro de Zirconio Delta (FCC) ---")
        # Parámetro de red para ZrH delta (~4.77 Å). Grupo 225 = FCC (Fm-3m)
        lat_fase = Lattice(4.77, 4.77, 4.77, 90, 90, 90)
        fase_activa = Phase(name="ZrH_Delta", space_group=225, structure=Structure(lattice=lat_fase))
        
        directorio_base = r"C:\Users\mavic\MiguelAngel\Texture Phyton\Exp_Data\ZrH\AI1"
        archivos_exp = [
            {"file": "d1_part2_p2_ZrH(111).txt", "hkl": [1, 1, 1]},
            {"file": "d1_part3_p3_ZrH(200).txt", "hkl": [2, 0, 0]}
        ]

    mis_pfs_originales = []

    # =========================================================
    # 2. CARGA DE DATOS EXPERIMENTALES
    # =========================================================
    print("--- Cargando y procesando datos ASCII ---")
    for info in archivos_exp:
        try:
            ruta = os.path.join(directorio_base, info["file"])
            datos = np.loadtxt(ruta, skiprows=4)
        except Exception as e:
            print(f"Error cargando {info['file']}: {e}")
            continue
            
        psi, phi, intensidades = datos[:, 0], datos[:, 1], datos[:, 2] 
        angulos_euler_rad = np.radians(np.column_stack((phi, 90 - psi, np.zeros_like(phi))))
        g = ~Rotation.from_euler(angulos_euler_rad)

        # Soporte dinámico para notación de 3 o 4 índices
        if "hkil" in info:
            miller_idx = Miller(hkil=info["hkil"], phase=fase_activa)
        else:
            miller_idx = Miller(hkl=info["hkl"], phase=fase_activa)

        mis_pfs_originales.append(PoleFigure(
            direcciones=g * Vector3d.zvector(), 
            intensidades=intensidades, 
            hkl=miller_idx
        ))

    print("\n--- Aplicando Rotación de Alineación ---")
    rot_correccion = Orientation.from_axes_angles(Vector3d([0, 1, 0]), np.radians(90))
    mis_pfs_corregidas = [pf.rotate(rot_correccion) for pf in mis_pfs_originales]

    # =========================================================
    # 3. RECONSTRUCCIÓN WIMV
    # =========================================================
    simetria_chapa = D2h
    ss = D2h
    
    oris_wimv, pesos_wimv = reconstruir_odf_wimv(
        lista_pfs_exp=mis_pfs_corregidas, 
        fase_cristal=fase_activa,
        simetria_muestra=ss, 
        resolucion_grados=5.0,  
        iteraciones=25          
    )
    
    print("\n--- Instanciando clase ODFDiscreta ---")
    mi_odf = ODFDiscreta(orientaciones=oris_wimv, pesos=pesos_wimv, crystal_sym=fase_activa.point_group, sample_sym=ss)    

    # =========================================================
    # 4. VISUALIZACIÓN: SECCIONES DE LA ODF
    # =========================================================
    print("\n--- Generando Secciones de la ODF ---")
    
    if MATERIAL == "Zr":
        # Para HCP, lo clásico es cortar a lo largo de phi2 (0 a 60)
        cortes = [0, 15, 30, 45, 60]
        eje_corte = 'phi2'
    elif MATERIAL == "ZrH":
        # Para Cúbicos (BCC/FCC), mapeamos de 0 a 90
        cortes = [0, 15, 30, 45, 60, 75, 90]
        eje_corte = 'phi2'

    print(f" -> Eje de corte: {eje_corte} en grados: {cortes}")
    mi_odf.plot_sections(
        sections=cortes, 
        axis=eje_corte,          
        res_grados=5.0,       
        cmap='jet'
    )

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # =========================================================
    # 5. RECALCULAR Y COMPARAR PFs
    # =========================================================
    print("\n--- Generando Comparativa EXP vs CALC (Numérica) ---")
    planos_medidos = [pf.hkl for pf in mis_pfs_corregidas]
    
    pfs_recalculadas = mi_odf.calc_pole_figures(lista_hkl=planos_medidos, res_grados=2.0)
    
    nombres_planos = []
    for pf in mis_pfs_corregidas:
        idx_format = pf.hkl.hkil.flatten() if MATERIAL == "Zr" else pf.hkl.hkl.flatten()
        nombres_planos.append(f"Plano {{{''.join([str(int(x)) for x in idx_format])}}}")

    plot_pf_comparison(
        pfs_in=mis_pfs_corregidas,
        pfs_out=pfs_recalculadas,
        titulos=nombres_planos,
        suptitle=f"Comparativa de PFs: Experimental vs Integración WIMV ({MATERIAL})",
        unificar_escala='hkl',
        direccion_x='vertical'
    )
    plt.show(block=False)
    plt.pause(0.1)

    # =========================================================
    # 6. CUANTIFICACIÓN AUTOMÁTICA DE LA TEXTURA
    # =========================================================
    print("\n=========================================================")
    print(f"📈 REPORTE CUANTITATIVO DE LA TEXTURA ({MATERIAL})")
    print("=========================================================")
    
    j_index = mi_odf.calc_texture_index(res_grados=5.0)
    print(f" -> J-Index (Fuerza global de la textura) : {j_index:.2f}")
    
    print("\n -> Fracciones de Volumen (Radio de integración esférica: ±15.0°):")
    radio_busqueda = 15.0
    
    if MATERIAL == "Zr":
        centro_basal_inclinado = [0, 30, 0]
        vol_basal_inc = mi_odf.calc_component_volume(center_euler=centro_basal_inclinado, radius_degrees=radio_busqueda, res_grados=5.0)
        print(f"    * Fibra Basal Inclinada {centro_basal_inclinado} : {vol_basal_inc * 100:>6.2f} %")
        
        centro_basal_normal = [0, 0, 0]
        vol_basal_norm = mi_odf.calc_component_volume(center_euler=centro_basal_normal, radius_degrees=radio_busqueda, res_grados=5.0)
        print(f"    * Polo Basal Normal     {centro_basal_normal} : {vol_basal_norm * 100:>6.2f} %")
        
        centro_prismatica = [0, 90, 30] 
        vol_prism = mi_odf.calc_component_volume(center_euler=centro_prismatica, radius_degrees=radio_busqueda, res_grados=5.0)
        print(f"    * Fibra Prismática      {centro_prismatica} : {vol_prism * 100:>6.2f} %")

    elif MATERIAL == "ZrH":
        centro_cube = [0, 0, 0]
        vol_cube = mi_odf.calc_component_volume(center_euler=centro_cube, radius_degrees=radio_busqueda, res_grados=5.0)
        print(f"    * Componente Cúbica        {centro_cube} : {vol_cube * 100:>6.2f} %")
        
        centro_goss = [0, 45, 0]
        vol_goss = mi_odf.calc_component_volume(center_euler=centro_goss, radius_degrees=radio_busqueda, res_grados=5.0)
        print(f"    * Componente Goss          {centro_goss} : {vol_goss * 100:>6.2f} %")
        
        centro_rot_cube = [45, 0, 0]
        vol_rot_cube = mi_odf.calc_component_volume(center_euler=centro_rot_cube, radius_degrees=radio_busqueda, res_grados=5.0)
        print(f"    * Componente Cúbica Rotada {centro_rot_cube} : {vol_rot_cube * 100:>6.2f} %")

    print("=========================================================\n")

    # =========================================================
    # 7. VALIDACIÓN DE BASES: SIMETRIZADA VS BUNGE
    # =========================================================
    print("\n=========================================================")
    print("🌀 VALIDACIÓN DE BASES: SIMETRIZADA VS IRREDUCIBLE DE BUNGE")
    print("=========================================================")
    L_MAX = 20  
    
    coefs_sym_mn = mi_odf.calc_fourier_coeffs(L_MAX)
    mi_odf_sym = ODFFourier(coefs_array=coefs_sym_mn, crystal_sym=fase_activa.point_group, sample_sym=ss)

    coefs_bunge = mi_odf.calc_bunge_coeffs(L_MAX)
    mi_odf_bunge = ODFBunge(C_bunge=coefs_bunge, crystal_sym=fase_activa.point_group, sample_sym=ss)

    print("\n--- Generando PFs Analíticas para comparación cruzada ---")
    pfs_fourier_sym = mi_odf_sym.calc_pole_figures(lista_hkl=planos_medidos, res_grados=2.0)
    pfs_fourier_bunge = mi_odf_bunge.calc_pole_figures(lista_hkl=planos_medidos, res_grados=2.0)

    plot_pf_comparison(
        pfs_in=pfs_fourier_sym,
        pfs_out=pfs_fourier_bunge,
        titulos=nombres_planos,
        suptitle=f"Comparativa Analítica: Simetrizada (Arriba) vs Bunge Irreducible (Abajo) (L_max={L_MAX})",
        unificar_escala='hkl',
        direccion_x='vertical'
    )
    plt.show(block=False)
    plt.pause(0.1)

    # =========================================================
    # 8. COMPARATIVA DIRECTA: WIMV DISCRETA VS FOURIER (SIMETRIZADA)
    # =========================================================
    print("\n=========================================================")
    print("🔍 COMPARATIVA DE PROYECCIONES: WIMV (DISCRETO) VS FOURIER (ANALÍTICO)")
    print("=========================================================")
    
    plot_pf_comparison(
        pfs_in=pfs_recalculadas,
        pfs_out=pfs_fourier_sym,
        titulos=nombres_planos,
        suptitle=f"Comparativa de Métodos: WIMV Discreta (Arriba) vs Fourier (Abajo)",
        unificar_escala='hkl',
        direccion_x='vertical'
    )

    print("\n✅ PROCESO COMPLETADO.")
    plt.show()

if __name__ == "__main__":
    main()