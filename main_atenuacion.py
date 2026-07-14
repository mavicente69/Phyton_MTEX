# -*- coding: utf-8 -*-
"""
Script Principal: Evaluación Integral de Atenuación Total y Transmisión.
Arquitectura final de texturaPy3.10
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from orix.vector import Miller
from orix.crystal_map import Phase
from orix.quaternion import Orientation
import utils_sym
import utils_pf

# Importaciones de los módulos de cálculo
from calc_sigma_elastic_coherent import calcular_seccion_eficaz_coherente, leer_red_cristalina
from calc_secciones_background import calcular_background_total

# ====================================================================
# 1. CARGADOR DE DATOS DE TEXTURA
# ====================================================================
def cargar_odf_experimental(filepath, crystal_sym, sample_sym=None):
    from utils_odf import ODFComponent
    from utils_kernels import OrientationKernel
    
    kernel_mtex = OrientationKernel(fwhm_grados=16.0, tipo='poussin')
    euler, pesos = None, None
    
    if filepath.endswith('.txt'):
        data = np.loadtxt(filepath)
        if data.shape[1] >= 4:
            euler = data[:, 0:3]
            pesos = data[:, 3]
    elif filepath.endswith('.mat'):
        mat = sio.loadmat(filepath)
        for k, v in mat.items():
            if isinstance(v, np.ndarray):
                if len(v.shape) == 2 and v.shape[1] == 3: euler = v
                elif len(v.shape) == 2 and (v.shape[1] == 1 or v.shape[0] == 1): pesos = v.flatten()
                
    if euler is None or pesos is None:
        raise ValueError(f"No se pudieron extraer Euler y Pesos del archivo: {filepath}")

    if np.max(euler) > 7.0:
        euler = np.radians(euler)
        
    phi1 = np.mod(euler[:, 0], 2 * np.pi)
    Phi  = euler[:, 1]
    phi2 = np.mod(euler[:, 2], 2 * np.pi)
    
    euler_reordenado = np.column_stack((phi1, Phi, phi2))
    oris = Orientation.from_euler(euler_reordenado, symmetry=crystal_sym)
    
    return ODFComponent(orientaciones=oris, pesos=pesos, crystal_sym=crystal_sym, sample_sym=sample_sym, kernels=kernel_mtex)

# ====================================================================
# 2. FUNCIÓN DE ORQUESTACIÓN: ATENUACIÓN TOTAL
# ====================================================================
def evaluar_atenuacion_integral(odf, direcciones_haz, lambdas, lattice_file, L_max=20, espesor_cm=1.0):
    """
    Coordina el cálculo de la sección eficaz coherente y el background térmico/absorción.
    Retorna todo en unidades macroscópicas (cm^-1) y la transmisión (adimensional).
    """
    print(f"\n[MOTOR] Procesando material desde: {lattice_file}")
    simetria_str, A_matrix, basis, background_coeffs = leer_red_cristalina(lattice_file)
    v0 = np.abs(np.linalg.det(A_matrix))

    # Fallback seguro por si el archivo TXT no tiene los coeficientes definidos
    if background_coeffs is None or len(background_coeffs) < 3:
        print("  [AVISO] No se detectó línea BACKGROUND válida en el txt. Usando valores genéricos.")
        background_coeffs = [0.1, 0.01, 0.05]

    # A) Calcular Sección Eficaz Coherente Direccional (Devuelve en cm^-1)
    print("  -> Evaluando componente Elástica Coherente (Textura)...")
    sigma_coh_dict, _ = calcular_seccion_eficaz_coherente(
        odf=odf, 
        direcciones_haz=direcciones_haz, 
        lattice_file=lattice_file, 
        lambdas=lambdas, 
        L_max=L_max, 
        unidades='cm-1'
    )

    # B) Calcular Background Isotrópico (Absorción + Incoherente + Inelástico)
    print("  -> Evaluando componente Isotrópica de Fondo...")
    sigma_bg_barns = calcular_background_total(lambdas, background_coeffs)
    # Convertimos los Barns a cm^-1 dividiendo por el volumen de la celda (en Angstroms^3)
    sigma_bg_cm1 = sigma_bg_barns / v0

    # C) Sumar y obtener Transmisión
    resultados_totales_cm1 = {}
    resultados_transmision = {}

    for dir_name in direcciones_haz.keys():
        sigma_tot = sigma_coh_dict[dir_name] + sigma_bg_cm1
        resultados_totales_cm1[dir_name] = sigma_tot
        
        # Ley de Beer-Lambert: T = I/I_0 = exp(-Sigma_macroscopica * espesor)
        resultados_transmision[dir_name] = np.exp(-sigma_tot * espesor_cm)

    return resultados_totales_cm1, resultados_transmision, sigma_bg_cm1

# ====================================================================
# 3. CONTROLADOR PRINCIPAL
# ====================================================================
def main():
    print("="*65)
    print(" TEXTURAPY 3.10 - EVALUADOR MACROSCÓPICO INTEGRAL")
    print("="*65)

    # ---------------------------------------------------------
    # PARÁMETROS DE CONFIGURACIÓN DEL USUARIO
    # ---------------------------------------------------------
    # 1. Archivos y Directorios dinámicos
    materiales = {
        '1': ('Aluminio', os.path.join('Exp_Data', 'Al', 'Al_lattice.txt'), 'odf_Bolmaro_ascii.txt'),
        '2': ('Cobre', os.path.join('Exp_Data', 'Cu', 'Cu_lattice.txt'), 'odf_Cu.txt'),
        '3': ('Zirconio', os.path.join('Exp_Data', 'Zr', 'Zr_lattice.txt'), 'odf_Zr4.mat')
    }

    print("Materiales disponibles:")
    for key, (nombre, f_red, _) in materiales.items():
        print(f"  [{key}] {nombre}")
    
    try:
        seleccion = input("\nElige un material ingresando el número (1/2/3) [1 por defecto]: ").strip()
    except EOFError:
        seleccion = '1'
        
    if seleccion not in materiales:
        seleccion = '1'

    nombre_mat, lattice_file, tex_file = materiales[seleccion]
    
    # Construcción dinámica de la ruta de la textura basada en la carpeta del material
    base_dir = os.path.dirname(lattice_file)
    tex_path = os.path.join(base_dir, tex_file)
    
    # 2. Configuración del haz de neutrones
    lambdas = np.linspace(1.0, 6.0, 500)
    direcciones_haz = {
        'Z (Normal)': (0.0, 0.0),
        'X (Transversal)': (np.pi/2, 0.0)
    }
    
    # 3. Parámetros físicos
    L_max_calculo = 20
    espesor_muestra_cm = 1.0
    # ---------------------------------------------------------

    if not os.path.exists(lattice_file):
        print(f"\n[ERROR CRÍTICO] No se encuentra el archivo de red: {lattice_file}")
        print(f"Asegúrate de que la carpeta '{base_dir}' exista en tu directorio de trabajo.")
        return
        
    if not os.path.exists(tex_path):
        print(f"\n[ERROR CRÍTICO] No se encuentra el archivo de textura: {tex_path}")
        return

    # --- PASO 1: CARGA DE TEXTURA Y CRISTALOGRAFÍA ---
    print(f"\n[FASE 1] Cargando ODF de {nombre_mat} y calculando proyecciones...")
    
    # Asignación de simetría basada en el material
    if nombre_mat in ['Aluminio', 'Cobre']:
        crystal_sym = utils_sym.obtener_simetria('m-3m')
        fase_mat = Phase(point_group=crystal_sym)
        hkls = [
            Miller(hkl=[1,1,1], phase=fase_mat), 
            Miller(hkl=[2,0,0], phase=fase_mat), 
            Miller(hkl=[2,2,0], phase=fase_mat)
        ]
    else:
        crystal_sym = utils_sym.obtener_simetria('6/mmm')
        fase_mat = Phase(point_group=crystal_sym)
        hkls = [
            Miller(hkil=[0,0,0,2], phase=fase_mat), 
            Miller(hkil=[1,0,-1,0], phase=fase_mat), 
            Miller(hkil=[1,0,-1,1], phase=fase_mat)
        ]
        
    sample_sym = utils_sym.obtener_simetria('-1')
    
    odf_exp = cargar_odf_experimental(tex_path, crystal_sym, sample_sym)
    pfs_exp = odf_exp.calc_pole_figures(hkls, res_grados=5.0)

    # --- PASO 2: CÁLCULO FÍSICO ---
    print("\n[FASE 2] Ejecutando motor de cálculo macroscópico...")
    sigma_total, transmision, sigma_fondo = evaluar_atenuacion_integral(
        odf=odf_exp, 
        direcciones_haz=direcciones_haz, 
        lambdas=lambdas, 
        lattice_file=lattice_file, 
        L_max=L_max_calculo,
        espesor_cm=espesor_muestra_cm
    )

    # --- PASO 3: EXPORTACIÓN Y GRÁFICOS ---
    print("\n[FASE 3] Generando reportes visuales...")
    
    # Gráfico A: Textura Original
    odf_exp.plot_sections(sections=[0, 45, 60], axis='phi2', res_grados=5)
    utils_pf.plot_pfs(pfs_exp, titulos=[str(h.hkl) for h in hkls], cmap='jet', direccion_x='vertical', max_val_global=False)

    # Gráfico B: Atenuación y Transmisión
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colores = ['blue', 'red', 'green', 'orange']

    # Panel 1: Sección Eficaz (Atenuación)
    for (nombre_dir, col) in zip(direcciones_haz.keys(), colores):
        ax1.plot(lambdas, sigma_total[nombre_dir], label=f'Haz || {nombre_dir}', linewidth=2, color=col)
    ax1.plot(lambdas, sigma_fondo, label='Fondo Isotrópico', color='k', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax1.set_title('Sección Eficaz Macroscópica Total', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitud de Onda $\lambda$ [$\AA$]', fontsize=12)
    ax1.set_ylabel('$\Sigma_{tot}$ [cm$^{-1}$]', fontsize=12)
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.set_xlim(lambdas.min(), lambdas.max())

    # Panel 2: Transmisión
    for (nombre_dir, col) in zip(direcciones_haz.keys(), colores):
        ax2.plot(lambdas, transmision[nombre_dir], label=f'Haz || {nombre_dir}', linewidth=2, color=col)
        
    ax2.set_title(f'Transmisión (Espesor: {espesor_muestra_cm} cm)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitud de Onda $\lambda$ [$\AA$]', fontsize=12)
    ax2.set_ylabel('Transmisión $T = I/I_0$', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.set_xlim(lambdas.min(), lambdas.max())

    plt.tight_layout()
    output_png = os.path.join(base_dir, f'Atenuacion_Transmision_{nombre_mat}.png')
    plt.savefig(output_png, dpi=300)
    print(f"\n[ÉXITO] Ejecución completada. Resultados guardados en '{output_png}'.")
    plt.show()

if __name__ == '__main__':
    main()