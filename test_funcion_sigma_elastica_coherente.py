# -*- coding: utf-8 -*-
"""
Script principal interactivo: Flujo Completo de Textura y Sección Eficaz.
1. Carga ODF experimental (ODFComponent).
2. Invierte la ODF por NNLS.
3. Evalúa Sección Eficaz Macroscópica embebida (A_lmu x B_lmu dinámicos).
4. Exporta y grafica los resultados.
Entorno: texturaPy3.10
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

from orix.vector import Miller
from orix.crystal_map import Phase
from orix.quaternion import Orientation
import utils_sym
import utils_fourier
import utils_inversion
import utils_pf

# Importamos tu módulo modular (Ya embebido)
from calc_sigma_elastic_coherent import calcular_seccion_eficaz_coherente

# ====================================================================
# CARGADOR DE DATOS EXPERIMENTALES (ODF COMPONENT)
# ====================================================================
def cargar_odf_experimental(filepath, crystal_sym, sample_sym=None):
    """
    Lee el archivo ASCII o MAT. 
    - Convención: Bunge (Z-X'-Z'') directo.
    - Kernel: De la Vallee Poussin (16°)
    Arma y devuelve directamente un objeto ODFComponent.
    """
    from utils_odf import ODFComponent
    from utils_kernels import OrientationKernel
    
    kernel_mtex = OrientationKernel(fwhm_grados=16.0, tipo='poussin')
    
    
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
        raise ValueError(f"No se pudieron extraer las columnas de Euler y Pesos del archivo: {filepath}")

    if np.max(euler) > 7.0:
        euler = np.radians(euler)
        
    phi1 = np.mod(euler[:, 0] , 2 * np.pi)
    Phi  = euler[:, 1]
    phi2 = np.mod(euler[:, 2] , 2 * np.pi)
    
    euler_reordenado = np.column_stack((phi1, Phi, phi2))
    
    oris = Orientation.from_euler(euler_reordenado, symmetry=crystal_sym)
    
    print(f"  [OK] ODF cargada exitosamente con {len(pesos)} nodos.")
    return ODFComponent(orientaciones=oris, pesos=pesos, crystal_sym=crystal_sym, sample_sym=sample_sym, kernels=kernel_mtex)

# ====================================================================
# EVALUADOR DE DISTANCIAS INTERPLANARES (d-spacing)
# ====================================================================
def calcular_d_spacings(lattice_file, hkls):
    """Calcula el d-spacing extrayendo la matriz de la red del archivo .txt"""
    with open(lattice_file, 'r') as f:
        lines = f.readlines()
    
    matrix_rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%') or line.isalpha(): 
            continue
        parts = line.split()
        if len(parts) == 3:
            matrix_rows.append([float(p) for p in parts])
        if len(matrix_rows) == 3: 
            break
            
    A_matrix = np.array(matrix_rows)
    A_inv = np.linalg.inv(A_matrix)
    B_matrix = 2 * np.pi * A_inv.T
    
    d_spacings = []
    for h_obj in hkls:
        h, k, l = h_obj.hkl[0]
        G_vec = h * B_matrix[0] + k * B_matrix[1] + l * B_matrix[2]
        d = 2 * np.pi / np.linalg.norm(G_vec)
        
        pg_name = str(getattr(getattr(h_obj, 'phase', None), 'point_group', '')).lower()
        if '6' in pg_name or 'hex' in pg_name:
            arr = h_obj.hkil[0]
            label = f"{int(arr[0])}{int(arr[1])}{int(arr[2])}{int(arr[3])}"
        else:
            label = f"{int(h)}{int(k)}{int(l)}"
            
        d_spacings.append((label, d))
        
    return d_spacings

# ====================================================================
# MOTOR PRINCIPAL
# ====================================================================
def main():
    print("="*65)
    print(" TEXTURAPY 3.10 - FLUJO COMPLETO: TEXTURA Y SECCIÓN EFICAZ")
    print("="*65)

    # --- 1. Definición de Materiales ---
    materiales = {
        '1': ('Aluminio', os.path.join('Exp_Data', 'Al', 'Al_lattice.txt'), 'odf_Bolmaro_ascii.txt'),
        '2': ('Cobre', os.path.join('Exp_Data', 'Cu', 'Cu_lattice.txt'), 'odf_Cu.txt'),
        '3': ('Zirconio', os.path.join('Exp_Data', 'Zr', 'Zr_lattice.txt'), 'odf_Zr4.mat')
    }

    print("Materiales disponibles:")
    for key, (nombre, f_red, _) in materiales.items():
        print(f"  [{key}] {nombre}")
    
    try:
        seleccion = input("\nElige un material ingresando el número (1/2/3): ").strip()
    except EOFError:
        seleccion = '1'
        
    if seleccion not in materiales:
        print("\n[AVISO] Selección inválida. Usando Aluminio [1] por defecto.")
        seleccion = '1'

    # --- 2. Definición de Unidades ---
    print("\nOpciones de unidades para la Sección Eficaz:")
    print("  [1] Barns / Celda (Microscópica por celda unidad)")
    print("  [2] cm^-1 (Macroscópica de atenuación)")
    try:
        sel_unidades = input("\nElige la unidad (1/2): ").strip()
    except EOFError:
        sel_unidades = '1'
        
    unidades_str = 'cm-1' if sel_unidades == '2' else 'barns/celda'

    nombre_mat, lattice_file, tex_file = materiales[seleccion]
    output_dir = os.path.dirname(lattice_file)
    
    print(f"\n[INFO] Has seleccionado: {nombre_mat} | Unidades: {unidades_str}")

    if not os.path.exists(lattice_file):
        print(f"\n[ERROR] No se encontró el archivo '{lattice_file}'.")
        return

    # L_max global para la ejecución
    L_max = 20

    # ====================================================================
    # FASE 1: EXTRACCIÓN DE ODF Y FIGURAS DE POLOS EXPERIMENTALES
    # ====================================================================
    print("\n--- FASE 1: Figuras de Polos Experimentales ---")
    
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
    
    # Simetría de la muestra
    sample_sym = utils_sym.obtener_simetria('-1') 
    
    tex_path = os.path.join(output_dir, tex_file)
    print(f"Leyendo ODF desde: {tex_path}")
    odf_exp = cargar_odf_experimental(tex_path, crystal_sym, sample_sym)
    
    odf_exp.plot_sections(sections=[0], axis='phi2', res_grados=5)
    
    print("Proyectando Figuras de Polos desde la ODF base...")
    pfs_exp = odf_exp.calc_pole_figures(hkls, res_grados=5.0)
    
    print("Graficando Figuras de Polos (X hacia arriba)...")
    utils_pf.plot_pfs(pfs_exp, titulos=[str(h.hkl) for h in hkls], cmap='jet', direccion_x='vertical', max_val_global=False)

    # ====================================================================
    # FASE 2: INVERSIÓN ODF (NNLS)
    # ====================================================================
    print("\n--- FASE 2: Inversión ODF (Método NNLS) ---")
    odf_rec = utils_inversion.reconstruir_odf_nnls(pfs_exp, fase_mat, simetria_muestra=sample_sym, resolucion_grados=5.0)

    print("\nRecalculando Figuras de Polos desde la ODF invertida...")
    pfs_rec = odf_rec.calc_pole_figures(hkls, res_grados=5.0)
    
    print("Graficando Comparativa: Experimental vs Recalculada...")
    utils_pf.plot_pf_comparison(
        pfs_exp, 
        pfs_rec, 
        titulos=[str(h.hkl) for h in hkls],
        suptitle=f"Verificación de Inversión NNLS - {nombre_mat}",
        direccion_x='vertical'
    )

    # ====================================================================
    # FASE 3: CÁLCULO DE SECCIÓN EFICAZ MACROSCÓPICA (EMBEBIDO)
    # ====================================================================
    print(f"\n--- FASE 3: Sección Eficaz y Coeficientes Direccionales (L_max={L_max}) ---")
    
    direcciones_haz = {
        'z': (0.0, 0.0),
        'x': (np.pi/2, 0.0),
        'y': (np.pi/2, np.pi/2)
    }
    
    # Generamos el vector de longitudes de onda a evaluar (Ya no dependemos del CSV pre-calculado)
    lambdas = np.linspace(1.0, 6.0, 500)
    
    # Llamamos a la función con el nuevo parámetro 'unidades' y le pasamos 'lattice_file'
    resultados_sigma, a_lmu_list = calcular_seccion_eficaz_coherente(
        odf=odf_rec, 
        direcciones_haz=direcciones_haz, 
        lattice_file=lattice_file, 
        lambdas=lambdas, 
        L_max=L_max,
        unidades=unidades_str
    )

    # ====================================================================
    # FASE 4: EXPORTACIONES Y VALIDACIÓN FOURIER
    # ====================================================================
    print("\n--- FASE 4: Exportación de Coeficientes y Resultados ---")
    
    C_bunge = odf_rec.C_bunge
    
    print("Exportando coeficientes de Fourier (C_l^munu)...")
    c_bunge_list = []
    for (l, mu, nu), valor in C_bunge.items():
        if abs(valor) > 1e-10:  
            c_bunge_list.append({
                'l': l,
                'mu': mu,
                'nu': nu,
                'Real': valor.real,
                'Imag': valor.imag,
                'Magnitud': abs(valor)
            })
    df_cbunge = pd.DataFrame(c_bunge_list)
    out_cbunge_file = os.path.join(output_dir, f'C_bunge_{nombre_mat}_L{L_max}.csv')
    df_cbunge.to_csv(out_cbunge_file, index=False)
    
    print("Calculando Figuras de Polos desde coeficientes de Fourier para validación...")
    from utils_odf import ODFBunge
    odf_fourier = ODFBunge(C_bunge, crystal_sym, sample_sym)
    pfs_fourier = odf_fourier.calc_pole_figures(hkls, res_grados=5.0)
    
    print("Graficando Comparativa: Recalculada (NNLS) vs Fourier...")
    utils_pf.plot_pf_comparison(
        pfs_rec, 
        pfs_fourier, 
        titulos=[str(h.hkl) for h in hkls],
        suptitle=f"Comparativa NNLS vs Fourier (L_max={L_max}) - {nombre_mat}",
        direccion_x='vertical'
    )
    
    print("Exportando coeficientes direccionales (A_lmu)...")
    df_almu = pd.DataFrame(a_lmu_list)
    out_almu_file = os.path.join(output_dir, f'A_lmu_{nombre_mat}_L{L_max}.csv')
    df_almu.to_csv(out_almu_file, index=False)
        
    print("Exportando secciones eficaces calculadas a Excel/CSV...")
    df_sigma = pd.DataFrame({'lambda_A': lambdas})
    for dir_name in direcciones_haz.keys():
        df_sigma[f'Sigma_{dir_name}'] = resultados_sigma[dir_name]
    
    out_sigma_csv = os.path.join(output_dir, f'Sigma_Macroscopica_{nombre_mat}.csv')
    df_sigma.to_csv(out_sigma_csv, index=False)
    
    try:
        out_sigma_excel = os.path.join(output_dir, f'Sigma_Macroscopica_{nombre_mat}.xlsx')
        df_sigma.to_excel(out_sigma_excel, index=False)
        print(f"[ÉXITO] Secciones eficaces exportadas nativamente a '{out_sigma_excel}'.")
    except ImportError:
        print(f"[ÉXITO] Secciones eficaces exportadas a '{out_sigma_csv}' (Ábrelo directamente con Excel).")
        
    # ====================================================================
    # FASE 5: GRAFICAR COMPARATIVA DIRECCIONAL
    # ====================================================================
    print("\n--- FASE 5: Generando gráfico comparativo de Sección Eficaz ---")
    plt.figure(figsize=(10, 6))
    colores = ['blue', 'red', 'green']
    
    for (nombre_dir, col) in zip(direcciones_haz.keys(), colores):
        plt.plot(lambdas, resultados_sigma[nombre_dir], label=f'Haz || {nombre_dir}', linewidth=2, color=col)
        
    d_spacings = calcular_d_spacings(lattice_file, hkls)
    y_max_actual = plt.ylim()[1]
    
    for label, d in d_spacings:
        lambda_bragg = 2 * d
        if min(lambdas) <= lambda_bragg <= max(lambdas):
            plt.axvline(x=lambda_bragg, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.8)
            plt.text(lambda_bragg - 0.05, y_max_actual * 0.95, f'$\lambda=2d_{{{label}}}$', 
                     rotation=90, color='dimgray', verticalalignment='top', horizontalalignment='right', 
                     fontsize=11, fontweight='bold')

    plt.title(f'Sección Eficaz Direccional - {nombre_mat}', fontsize=15, fontweight='bold')
    plt.xlabel('Longitud de Onda $\lambda$ [$\AA$]', fontsize=12)
    
    # Etiqueta Y Dinámica según las unidades seleccionadas
    if unidades_str == 'cm-1':
        lbl_y = r'Sección Eficaz Macroscópica $\Sigma_{coh}$ [cm$^{-1}$]'
    else:
        lbl_y = r'Sección Eficaz $\sigma_{coh}$ [Barns / Celda]'
    plt.ylabel(lbl_y, fontsize=12)
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Forzamos límites
    plt.xlim(lambdas.min(), lambdas.max())
    y_max_plot = plt.ylim()[1]
    plt.ylim(-0.05 * y_max_plot, y_max_plot * 1.05)
    
    plt.tight_layout()
    
    plot_sigma_file = os.path.join(output_dir, f'Sigma_Direccional_{nombre_mat}.png')
    plt.savefig(plot_sigma_file, dpi=300)
    print(f"[ÉXITO] Gráfico de atenuación direccional guardado en '{plot_sigma_file}'.")
    plt.show()

if __name__ == '__main__':
    main()