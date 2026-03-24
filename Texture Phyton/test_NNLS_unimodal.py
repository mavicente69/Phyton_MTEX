# -*- coding: utf-8 -*-
"""
Motor de Inversión (M-ART) - Unificado para Datos Sintéticos y Experimentales
Soporte Multi-Material (HCP, BCC, FCC)
Entorno: texturaPy3.10
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.vector import Miller, Vector3d
from diffpy.structure import Lattice, Structure
from orix.quaternion import Orientation, Rotation
from orix.quaternion.symmetry import D2h, C1

# Importamos TODAS tus herramientas nativas
import utils_orient
from utils_inversion import reconstruir_odf_nnls
# ACÁ ESTÁ EL ARREGLO: Importamos el nombre exacto de tu clase
from utils_odf import ODFComponent, ODFIsotropic 
from utils_kernels import OrientationKernel
from utils_pf import PoleFigure, plot_pf_comparison, plot_pfs

def main():
    # ====================================================================
    # 0. CENTRO DE CONTROL: MODO Y MATERIAL
    # ====================================================================
    MODO_ANALISIS = "experimental"   # Opciones: "sintetico" o "experimental"
    MATERIAL = "zirconio"         # Opciones: "zirconio", "acero_bcc", "acero_fcc"
    
    # Simetría de muestra (C1 triclinica o D2h ortorrómbica para laminación)
    ss = C1  
    
    # ====================================================================
    # 1. CONFIGURACIÓN DEL MATERIAL
    # ====================================================================
    if MATERIAL == "zirconio":
        print("--- Configurando Fase: Zirconio (HCP) ---")
        lat = Lattice(3.232, 3.232, 5.147, 90, 90, 120)
        fase_cristal = Phase(name="Zirconium", space_group=194, structure=Structure(lattice=lat))
        
        planos_sinteticos = [
            Miller(hkil=[0,0,0,2], phase=fase_cristal),
            Miller(hkil=[1,0,-1,0], phase=fase_cristal),
            Miller(hkil=[1,0,-1,1], phase=fase_cristal)
        ]
        titulos_pf = ['Basal {0002}', 'Prismático {10-10}', 'Piramidal {10-11}']
        
        ori_central_grados = [20, 30, 0] 
        secciones_plot = [0, 15, 30, 45, 60]

    elif MATERIAL == "acero_bcc":
        print("--- Configurando Fase: Acero Ferrítico (BCC) ---")
        lat = Lattice(2.866, 2.866, 2.866, 90, 90, 90)
        fase_cristal = Phase(name="Steel_BCC", space_group=229, structure=Structure(lattice=lat))
        
        planos_sinteticos = [
            Miller(hkl=[1,1,0], phase=fase_cristal),
            Miller(hkl=[2,0,0], phase=fase_cristal),
            Miller(hkl=[2,1,1], phase=fase_cristal)
        ]
        titulos_pf = ['{110}', '{200}', '{211}']
        
        ori_central_grados = [0, 45, 0]
        secciones_plot = [0, 45]

    elif MATERIAL == "acero_fcc":
        print("--- Configurando Fase: Acero Inoxidable (FCC) ---")
        lat = Lattice(3.60, 3.60, 3.60, 90, 90, 90)
        fase_cristal = Phase(name="Steel_FCC", space_group=225, structure=Structure(lattice=lat))
        
        planos_sinteticos = [
            Miller(hkl=[1,1,1], phase=fase_cristal),
            Miller(hkl=[2,0,0], phase=fase_cristal),
            Miller(hkl=[2,2,0], phase=fase_cristal)
        ]
        titulos_pf = ['{111}', '{200}', '{220}']
        
        ori_central_grados = [35, 45, 90] 
        secciones_plot = [0, 45]
        
    else:
        raise ValueError("Material no reconocido.")

    cs = fase_cristal.point_group
    pfs_entrada = []
    lista_planos = []

    # ====================================================================
    # 2. GENERACIÓN / CARGA DE DATOS
    # ====================================================================
    if MODO_ANALISIS == "sintetico":
        print(f"--- Generando Mezcla de Textura: Unimodal + Isotrópica ---")
        lista_planos = planos_sinteticos
        
        # 1. Componente Unimodal (Fuerte - 70% de la masa)
        ori_central = Orientation.from_euler(np.radians([ori_central_grados]), symmetry=cs)
        odf_unimodal = ODFComponent(
            orientaciones=ori_central,
            pesos=[0.01],  # Directamente le asignamos el 70%
            kernels=OrientationKernel(fwhm_grados=15.0, tipo='gaussian'), 
            crystal_sym=cs,
            sample_sym=ss
        )
        
        # 2. Componente Isotrópica usando TU clase (Fondo - 30% de la masa)
        odf_iso = ODFIsotropic(crystal_sym=cs, sample_sym=ss, peso=0.99
                               )
        
        # 3. MAGIA OOP: Tu método __add__ crea automáticamente una ODFMixed
        odf_mezcla = odf_unimodal + odf_iso
        
        # 4. Calculamos las PFs directamente desde el objeto mixto
        pfs_perfectas_mezcladas = odf_mezcla.calc_pole_figures(lista_hkl=lista_planos, resolution=30)

        # 5. SABOTAJE: Inyectamos factores de escala arbitrarios a la mezcla
        print("--- Saboteando escalas MUD de las PFs sintéticas mezcladas ---")
        factores_sabotaje = [1500, 750, 300
                             ] 
        
        pfs_entrada = []
        titulos_nuevos = [] 
        
        for i, pf_mezcla in enumerate(pfs_perfectas_mezcladas):
            factor = factores_sabotaje[i] if i < len(factores_sabotaje) else 1.0
            print(f"    * Multiplicando {titulos_pf[i]} por factor espurio de {factor}")
            
            intensidades_arruinadas = pf_mezcla.intensidades * factor
            pf_saboteada = PoleFigure(
                direcciones=pf_mezcla.direcciones, 
                intensidades=intensidades_arruinadas, 
                hkl=pf_mezcla.hkl
            )
            pfs_entrada.append(pf_saboteada)
            
            titulos_nuevos.append(f"{titulos_pf[i]} (x {factor})")

        titulos_pf = titulos_nuevos

        print("\n[PASO 1.5] Dibujando PFs Sintéticas (Mezcla Saboteada)...")
        plot_pfs(pfs_entrada, titulos=titulos_pf, direccion_x='vertical')
        
    elif MODO_ANALISIS == "experimental":
        print("--- Cargando y procesando datos ASCII experimentales ---")
        
        if MATERIAL == "zirconio":
            directorio_base = r"C:\Users\mavic\MiguelAngel\KOWARI-HIDRUROS\Viaje a ANSTO\Muestras textura Chapa Zircaloy4\Texturas\Zr_tex_prisms\N0"
            archivos_exp = [
                {"file": "d1_part1_p1_Zr(100).txt", "hkil": [1, 0, -1, 0]},
                {"file": "d1_part2_p1_Zr(002).txt", "hkil": [0, 0, 0, 2]},
                {"file": "d1_part2_p2_Zr(101).txt", "hkil": [1, 0, -1, 1]},
                {"file": "d2_part1_p1_Zr(102).txt", "hkil": [1, 0, -1, 2]},
                {"file": "d2_part3_p1_Zr(110).txt", "hkil": [1, 1, -2, 0]}
            ]
        else:
            raise NotImplementedError("Rutas de archivos experimentales para Aceros no configuradas.")

        mis_pfs_raw = []
        mis_pfs_mud = []
        titulos_pf_exp = []

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
            
            if MATERIAL == "zirconio":
                hkl_actual = Miller(hkil=info["hkil"], phase=fase_cristal)
                titulos_pf_exp.append(f"Polo {info['hkil']}")
            else:
                hkl_actual = Miller(hkl=info["hkl"], phase=fase_cristal)
                titulos_pf_exp.append(f"Polo {info['hkl']}")

            mis_pfs_raw.append(PoleFigure(direcciones, intensidades_raw, hkl_actual))

            pesos_vol = np.sin(np.radians(psi))
            pesos_vol[pesos_vol == 0] = 1e-6
            intensidad_random = np.sum(intensidades_raw * pesos_vol) / np.sum(pesos_vol)
            intensidades_mud = intensidades_raw / intensidad_random

            mis_pfs_mud.append(PoleFigure(direcciones, intensidades_mud, hkl_actual))
            lista_planos.append(hkl_actual)

        titulos_pf = titulos_pf_exp

        print("\n--- Aplicando Rotación de Alineación ---")
        rot_correccion = Orientation.from_axes_angles(Vector3d([0, 1, 0]), np.radians(90))
        pfs_raw_rotadas = [pf.rotate(rot_correccion) for pf in mis_pfs_raw]
        pfs_entrada = [pf.rotate(rot_correccion) for pf in mis_pfs_mud]

        print("\n[PASO 1.5] Dibujando PFs Experimentales (Conteos Brutos)...")
        plot_pfs(pfs_raw_rotadas, titulos=[f"RAW {t}" for t in titulos_pf], direccion_x='vertical')

    # ====================================================================
    # 3. INVERSIÓN M-ART (ESTADÍSTICA ROBUSTA)
    # ====================================================================
    print("\n[PASO 2] Alimentando Inversión M-ART...")
    
    mi_odf = reconstruir_odf_nnls(
        lista_pfs_exp=pfs_entrada, 
        fase_cristal=fase_cristal, 
        simetria_muestra=ss,
        tipo_kernel='gaussian',
        resolucion_grados=3 
    )

    # ====================================================================
    # 4. RECALCULAR Y COMPARAR PFs
    # ====================================================================
    print("\n[PASO 3] Recalculando PFs desde la ODF invertida...")
    pfs_recalc = mi_odf.calc_pole_figures(lista_hkl=lista_planos, resolution=30)

    print("\n[PASO 4] Dibujando Comparativa de Figuras de Polos (MUD)...")
    plot_pf_comparison(
        pfs_in=pfs_entrada, 
        pfs_out=pfs_recalc, 
        titulos=titulos_pf, 
        suptitle=f"Inversión M-ART - {MODO_ANALISIS.upper()} (Escala MUD)",
        unificar_escala='hkl',  
        direccion_x='vertical' 
    )

    # ====================================================================
    # 5. DIBUJAR ODF
    # ====================================================================
    print("\n[PASO 5] Dibujando ODF (Secciones Phi2)...")
    mi_odf.plot_sections(sections=secciones_plot, axis='phi2', res_grados=5)

    plt.show()

if __name__ == "__main__":
    main()