# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:53:44 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Motor de Inversión (NNLS) - Unificado para Datos Sintéticos y Experimentales
Entorno: texturaPy3.10
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.vector import Miller, Vector3d
from diffpy.structure import Lattice, Structure
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import D2h,C1

# Importamos TODAS tus herramientas nativas
import utils_orient
from utils_inversion import reconstruir_odf_nnls
from utils_odf import ODFComponent
from utils_kernels import OrientationKernel
from utils_pf import PoleFigure, plot_pf_comparison, plot_pfs  # <-- Agregamos plot_pfs

def main():
    # ====================================================================
    # 0. CONTROLADOR DE MODO ("sintetico" o "experimental")
    # ====================================================================
    MODO_ANALISIS = "experimental"  # <-- Cambiá esto para switchear

    # ====================================================================
    # 1. CONFIGURACIÓN DEL MATERIAL
    # ====================================================================
    print("--- Configurando Fase: Zirconio (HCP) ---")
    lat_zr = Lattice(3.232, 3.232, 5.147, 90, 90, 120)
    fase_zr = Phase(name="Zirconium", space_group=194, structure=Structure(lattice=lat_zr))
    cs = fase_zr.point_group
    ss = D2h
    ss = C1
    
    pfs_entrada = []
    lista_planos = []
    titulos_pf = []

    if MODO_ANALISIS == "sintetico":
        print("--- Generando ODF Sintética Unimodal ---")
        ori_central = Orientation.from_euler(np.radians([[0, 30, 0]]), symmetry=cs)
        odf_ideal = ODFComponent(
            orientaciones=ori_central,
            pesos=[1.0],
            kernels=OrientationKernel(fwhm_grados=15.0, tipo='gaussian'), 
            crystal_sym=cs,
            sample_sym=ss
        )

        lista_planos = [Miller(hkil=[0,0,0,2], phase=fase_zr),
                        Miller(hkil=[1,0,-1,0], phase=fase_zr),
                        Miller(hkil=[1,0,-1,1], phase=fase_zr)]
        
        pfs_entrada = odf_ideal.calc_pole_figures(lista_hkl=lista_planos, resolution=30)
        titulos_pf = ['Basal {0002}', 'Prismático {10-10}', 'Piramidal {10-11}']

    elif MODO_ANALISIS == "experimental":
        print("--- Cargando y procesando datos ASCII experimentales ---")
        directorio_base = r"C:\Users\mavic\MiguelAngel\KOWARI-HIDRUROS\Viaje a ANSTO\Muestras textura Chapa Zircaloy4\Texturas\Zr_tex_prisms\N0"

        archivos_exp = [
            {"file": "d1_part1_p1_Zr(100).txt", "hkil": [1, 0, -1, 0]},
            {"file": "d1_part2_p1_Zr(002).txt", "hkil": [0, 0, 0, 2]},
            {"file": "d1_part2_p2_Zr(101).txt", "hkil": [1, 0, -1, 1]},
            {"file": "d2_part1_p1_Zr(102).txt", "hkil": [1, 0, -1, 2]},
            {"file": "d2_part3_p1_Zr(110).txt", "hkil": [1, 1, -2, 0]}
        ]

        mis_pfs_raw = []
        mis_pfs_mud = []

        for info in archivos_exp:
            try:
                ruta = os.path.join(directorio_base, info["file"])
                datos = np.loadtxt(ruta, skiprows=4)
            except Exception as e:
                print(f"⚠️ Ignorando archivo no encontrado: {info['file']}")
                continue
                
            psi, phi, intensidades_raw = datos[:, 0], datos[:, 1], datos[:, 2] 
            angulos_euler_rad = np.radians(np.column_stack((phi, 90 - psi, np.zeros_like(phi))))
            g = ~Rotation.from_euler(angulos_euler_rad)
            direcciones = g * Vector3d.zvector()
            hkl_actual = Miller(hkil=info["hkil"], phase=fase_zr)

            # 1. Guardamos la versión cruda (RAW Counts)
            mis_pfs_raw.append(PoleFigure(direcciones, intensidades_raw, hkl_actual))
            
            # 2. Calculamos y guardamos la versión normalizada (MUD)
            pesos_vol = np.sin(np.radians(psi))
            pesos_vol[pesos_vol == 0] = 1e-6 
            intensidad_random = np.sum(intensidades_raw * pesos_vol) / np.sum(pesos_vol)
            intensidades_mud = intensidades_raw / intensidad_random
            
            mis_pfs_mud.append(PoleFigure(direcciones, intensidades_mud, hkl_actual))
            
            lista_planos.append(hkl_actual)
            titulos_pf.append(f"Polo {info['hkil']}")

        print("\n--- Aplicando Rotación de Alineación ---")
        rot_correccion = Orientation.from_axes_angles(Vector3d([0, 1, 0]), np.radians(90))
        pfs_raw_rotadas = [pf.rotate(rot_correccion) for pf in mis_pfs_raw]
        pfs_entrada = [pf.rotate(rot_correccion) for pf in mis_pfs_mud]

        print("\n[PASO 1.5] Dibujando PFs Experimentales (Conteos Brutos)...")
        # Mostramos los datos crudos antes de que empiece la inversión
        plot_pfs(pfs_raw_rotadas, titulos=[f"RAW {t}" for t in titulos_pf])

    # ====================================================================
    # 3. INVERSIÓN NNLS
    # ====================================================================
    print("\n[PASO 2] Alimentando Inversión NNLS...")
    mi_odf = reconstruir_odf_nnls(
        lista_pfs_exp=pfs_entrada, 
        fase_cristal=fase_zr, 
        simetria_muestra=ss,
        resolucion_grados=5
    )

    # ====================================================================
    # 4. RECALCULAR Y COMPARAR PFs
    # ====================================================================
    print("\n[PASO 3] Recalculando PFs desde la ODF invertida...")
    pfs_recalc = mi_odf.calc_pole_figures(lista_hkl=lista_planos, resolution=30)

    print("\n[PASO 4] Dibujando Comparativa de Figuras de Polos (MUD)...")
    # Como ambas (pfs_entrada y pfs_recalc) están en MUD, volvemos a 'global'
    plot_pf_comparison(
        pfs_in=pfs_entrada, 
        pfs_out=pfs_recalc, 
        titulos=titulos_pf, 
        suptitle=f"Inversión NNLS - {MODO_ANALISIS.upper()} (Escala MUD)",
        unificar_escala='global' 
    )

    # ====================================================================
    # 5. DIBUJAR ODF
    # ====================================================================
    print("\n[PASO 5] Dibujando ODF (Secciones Phi2)...")
    mi_odf.plot_sections(sections=[0, 15, 30, 45, 60], axis='phi2', res_grados=2)

    plt.show()

if __name__ == "__main__":
    main()