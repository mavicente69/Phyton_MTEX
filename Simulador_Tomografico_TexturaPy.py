# -*- coding: utf-8 -*-
"""
Simulador de Tomografía Tensorial de Neutrones (Multi-Material).
Entorno: texturaPy3.10
Estructurado en Bloques para uso interactivo en Spyder (# %%)

NUEVO: Soporte Multi-Material (Vóxel a Vóxel). Permite definir múltiples 
composiciones químicas y/o texturas en diferentes regiones espaciales del volumen.
Incluye exportación de la topología 3D real para el Visualizador.

CORREGIDO: El tilteo ahora aplica la rotación 3D alrededor del centro de masa
de la muestra, preservando la física de la tomografía.
REFACTORIZADO: El rango y resolución de lambda se definen en UN solo lugar (CONFIG).
"""
# %% [BLOQUE 1] IMPORTACIONES Y FUNCIONES BASE (Correr una vez al inicio)
import os
import time
import numpy as np
import pandas as pd
import tomopy
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from main_atenuacion import cargar_odf_experimental
from calc_secciones_background import calcular_background_total
from Blmu_matrix import calcular_B_lmu

import utils_sym
import utils_fourier

def leer_red_cristalina(filepath):
    """Lee el archivo de red cristalográfica (Sin valores por defecto)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    matrix_rows, basis_rows = [], []
    simetria_str, background_coeffs = None, None 
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'): continue
            
        if line.upper().startswith('BACKGROUND'):
            background_coeffs = [float(x) for x in line.split()[1:]]
            continue
            
        parts = line.split()
        if len(parts) == 1 and parts[0].isalpha():
            simetria_str = parts[0].lower()
            continue
            
        if len(parts) == 3: matrix_rows.append([float(p) for p in parts])
        elif len(parts) >= 5: basis_rows.append([float(p) for p in parts])
            
    if simetria_str is None or background_coeffs is None:
        raise ValueError(f"\n[ERROR CRÍTICO] Faltan datos (Simetría o BACKGROUND) en: {filepath}")
        
    return simetria_str, np.array(matrix_rows), np.array(basis_rows), background_coeffs

def crear_volumen_multimaterial(N, radio_cilindro, altura_cilindro):
    """
    Crea un mapa de ID de materiales vóxel a vóxel.
    0 = Vacío/Aire
    1 = Material 1 (Ej: Núcleo interno)
    2 = Material 2 (Ej: Coraza externa)
    """
    volumen = np.zeros((N, N, N), dtype=np.int8)
    centro = N // 2
    
    z_start = max(0, centro - altura_cilindro // 2)
    z_end = min(N, centro + altura_cilindro // 2)
    
    radio_nucleo = radio_cilindro * 0.5 # El núcleo ocupa la mitad del radio
    
    for z in range(z_start, z_end):
        for y in range(N):
            for x in range(N):
                r2 = (x - centro)**2 + (y - centro)**2
                if r2 <= radio_nucleo**2:
                    volumen[z, y, x] = 1 # ID Material 1
                elif r2 <= radio_cilindro**2:
                    volumen[z, y, x] = 2 # ID Material 2
                    
    return volumen

def proyectar_coeficientes_textura(mascara_mat, C_bunge, angulos_rad, voxel_size_cm):
    """Proyecta la textura multiplicada por la densidad de SU material específico."""
    proyecciones_C = {}
    volumen_bg = mascara_mat * voxel_size_cm
    proyecciones_C['background'] = tomopy.project(volumen_bg, angulos_rad)
    
    coefs_activos = {k: v for k, v in C_bunge.items() if np.max(np.abs(np.asarray(v))) >= 1e-10}
    
    for (l, mu, nu), c_val in coefs_activos.items():
        c_array = np.asarray(c_val)
        if np.max(np.abs(c_array.real)) > 1e-10:
            proyecciones_C[(l, mu, nu, 'real')] = tomopy.project(mascara_mat * c_array.real * voxel_size_cm, angulos_rad)
        if np.max(np.abs(c_array.imag)) > 1e-10:
            proyecciones_C[(l, mu, nu, 'imag')] = tomopy.project(mascara_mat * c_array.imag * voxel_size_cm, angulos_rad)
    return proyecciones_C

def extraer_B_lmu_desde_df(df_B, lambda_val, L_max):
    idx_lam = (np.abs(df_B['lambda_A'] - lambda_val)).argmin()
    row = df_B.iloc[idx_lam]
    B_lmu = {}
    for l in range(0, L_max + 1, 2):
        mu = 0
        while f'B_{l}_{mu}_Real' in df_B.columns:
            v_real, v_imag = row[f'B_{l}_{mu}_Real'], row[f'B_{l}_{mu}_Imag']
            if abs(v_real) > 1e-10 or abs(v_imag) > 1e-10:
                B_lmu[(l, mu)] = v_real + 1j * v_imag
            mu += 1
    return B_lmu, row['lambda_A']

def calcular_atenuacion_material(lambda_val, proyecciones_C, B_lmu, bg_coeffs, v0, angulos_rad, L_max, sample_sym, tilt_deg=0.0):
    """Calcula la atenuación pura (Sigma * x) para un solo material."""
    num_angulos, resolucion_z, resolucion_x = proyecciones_C['background'].shape
    
    sigma_bg_barns = calcular_background_total(np.array([lambda_val]), bg_coeffs)[0]
    atenuacion_mat = proyecciones_C['background'] * (sigma_bg_barns / v0)
    
    _, proj_S = utils_fourier.calc_symmetry_projectors(L_max, sample_sym, sample_sym)
    _, B_samp = utils_fourier.calc_symmetry_coefficients(L_max, proj_S, proj_S)
    
    for i, ang_rad in enumerate(angulos_rad):
        polar_haz = np.pi / 2 - np.radians(tilt_deg)
        azim_haz = -ang_rad
        
        for l in range(0, L_max + 1, 2):
            if l not in B_samp or B_samp[l].shape[1] == 0: continue
            k_lnu = utils_fourier.eval_sym_sph_harm(l, polar_haz, azim_haz, B_samp[l], is_sample=True).flatten()
            
            mu_max = max([k[1] for k in B_lmu.keys() if k[0] == l] + [-1])
            if mu_max == -1: continue
            
            for mu in range(mu_max + 1):
                b_val_cm1 = B_lmu.get((l, mu), 0.0j) / v0
                if abs(b_val_cm1) < 1e-10: continue
                
                for nu in range(len(k_lnu)):
                    k_val = k_lnu[nu]
                    if (l, mu, nu, 'real') in proyecciones_C:
                        atenuacion_mat[i] += np.real(b_val_cm1 * k_val * proyecciones_C[(l, mu, nu, 'real')][i])
                    if (l, mu, nu, 'imag') in proyecciones_C:
                        atenuacion_mat[i] += np.real(b_val_cm1 * k_val * 1j * proyecciones_C[(l, mu, nu, 'imag')][i])
                        
    return atenuacion_mat

def ensamblar_radiografias_multimaterial(lambda_test, ws_data, tilt_deg):
    """Suma las atenuaciones de todos los materiales y aplica Beer-Lambert."""
    atenuacion_total = None
    lam_real_evaluado = lambda_test
    
    for id_mat, info_mat in ws_data['materiales'].items():
        B_lmu, lam_real = extraer_B_lmu_desde_df(info_mat['df_B'], lambda_test, ws_data['L_max'])
        lam_real_evaluado = lam_real # Actualiza al lambda real de la grilla
        
        proy_C_mat = ws_data['proyecciones_C_tilts'][tilt_deg][id_mat]
        
        atenuacion_mat = calcular_atenuacion_material(
            lam_real, proy_C_mat, B_lmu, info_mat['bg_coeffs'], info_mat['v0'], 
            ws_data['angulos_rad'], ws_data['L_max'], ws_data['sample_sym'], tilt_deg
        )
        
        if atenuacion_total is None:
            atenuacion_total = np.zeros_like(atenuacion_mat)
            
        atenuacion_total += atenuacion_mat # La densidad óptica es puramente aditiva
        
    radiografias = np.exp(-atenuacion_total)
    return radiografias, lam_real_evaluado

# =========================================================================================
# EXTRACCIÓN Y GUARDADO DE PROYECCIONES DIRECTAS (DETECTOR) DE A_lmu (POR MATERIAL)
# =========================================================================================
def extraer_proyecciones_Almu(proyecciones_C_mat, angulos_rad, L_max, sample_sym, tilt_deg=0.0):
    _, proj_S = utils_fourier.calc_symmetry_projectors(L_max, sample_sym, sample_sym)
    _, B_samp = utils_fourier.calc_symmetry_coefficients(L_max, proj_S, proj_S)
    
    l_mu_keys = set((k[0], k[1]) for k in proyecciones_C_mat.keys() if isinstance(k, tuple) and len(k) == 4)
    proyecciones_Almu = {}
    n_ang, res_z, res_x = proyecciones_C_mat['background'].shape
    
    for (l, mu) in sorted(list(l_mu_keys)):
        proj_real = np.zeros((n_ang, res_z, res_x), dtype=np.float32)
        proj_imag = np.zeros((n_ang, res_z, res_x), dtype=np.float32)
        
        for i, ang_rad in enumerate(angulos_rad):
            polar_haz = np.pi / 2 - np.radians(tilt_deg)
            k_lnu = utils_fourier.eval_sym_sph_harm(l, polar_haz, -ang_rad, B_samp[l], is_sample=True).flatten()
            
            for nu in range(len(k_lnu)):
                k_val = k_lnu[nu]
                p_real = proyecciones_C_mat.get((l, mu, nu, 'real'))
                p_imag = proyecciones_C_mat.get((l, mu, nu, 'imag'))
                
                if p_real is not None:
                    proj_real[i] += p_real[i] * k_val.real
                    proj_imag[i] += p_real[i] * k_val.imag
                if p_imag is not None:
                    proj_real[i] -= p_imag[i] * k_val.imag
                    proj_imag[i] += p_imag[i] * k_val.real
                    
        proyecciones_Almu[(l, mu)] = (proj_real, proj_imag)
    return proyecciones_Almu

def guardar_proyecciones_Almu(proyecciones_Almu, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for (l, mu), (proj_real, proj_imag) in proyecciones_Almu.items():
        proj_final = proj_real + 1j * proj_imag if np.max(np.abs(proj_imag)) > 1e-8 else proj_real
        np.save(os.path.join(out_dir, f"Proyecciones_Detector_Almu_L{l}_mu{mu}.npy"), proj_final)

# %% [BLOQUE 2] INICIALIZACIÓN Y CÁLCULO PESADO (MAIN CONFIGURATION)
def inicializar_workspace(config):
    print("="*65)
    print(" INICIALIZANDO ESPACIO DE TRABAJO MULTI-MATERIAL")
    print("="*65)
    
    t0 = time.time()
    L_max = config['L_max']
    N_voxels = config['N_voxels']
    angulos_theta = np.linspace(0, 180, config['num_proyecciones'], endpoint=False)
    angulos_rad = np.radians(angulos_theta)
    sample_sym = utils_sym.obtener_simetria('-1')

    # 1. Crear Mapa de Materiales Vóxel a Vóxel
    volumen_ids = crear_volumen_multimaterial(N_voxels, config['radio_cilindro'], config['altura_cilindro'])
    
    ws_materiales = {}
    mapa_simetrias = {'cubic': 'm-3m', 'hexagonal': '6/mmm', 'tetragonal': '4/mmm', 'orthorhombic': 'mmm'}
    
    # 2. Cargar Física y Textura por cada Material definido
    for id_mat, cfg_mat in config['materiales'].items():
        print(f"\n[CARGANDO MATERIAL {id_mat}: {cfg_mat['nombre']}]")
        
        simetria_str, A_matrix, _, bg_coeffs = leer_red_cristalina(cfg_mat['lattice_file'])
        grupo_puntual = mapa_simetrias[simetria_str]
        crystal_sym = utils_sym.obtener_simetria(grupo_puntual) 
        
        odf_mat = cargar_odf_experimental(cfg_mat['tex_path'], crystal_sym, sample_sym)
        C_bunge_mat = odf_mat.calc_bunge_coeffs(L_max)
        
        # GUARDAR B_lmu EN EL DIRECTORIO ESPECÍFICO DEL MATERIAL
        b_lmu_file = os.path.join(cfg_mat['base_dir'], f"B_lmu_{cfg_mat['nombre']}.csv")
        if not os.path.exists(b_lmu_file):
            print(f"Calculando B_lmu base para {cfg_mat['nombre']}...")
            # USAMOS LOS PARÁMETROS DE LAMBDA DEFINIDOS EN CONFIG
            calcular_B_lmu(
                cfg_mat['lattice_file'], 
                b_lmu_file, 
                L_max, 
                config['LAMBDA_MIN'], 
                config['LAMBDA_MAX'], 
                config['NUM_LAMBDA_PUNTOS']
            )
            
        ws_materiales[id_mat] = {
            'nombre': cfg_mat['nombre'],
            'v0': np.abs(np.linalg.det(A_matrix)),
            'bg_coeffs': bg_coeffs,
            'crystal_sym': crystal_sym,
            'C_bunge': C_bunge_mat,
            'df_B': pd.read_csv(b_lmu_file)
        }

    # 3. Calcular Proyecciones Tensoriales (Tilteo + Material Separado)
    proyecciones_C_tilts = {}
    
    for tilt in config['angulos_tilt']:
        print(f"\n--- Procesando Geometría a Tilt: {tilt}° ---")
        proyecciones_C_tilts[tilt] = {}
        
        for id_mat in ws_materiales.keys():
            # Extraemos SOLO la máscara de este material (1.0 donde está, 0.0 donde no)
            mascara_base = (volumen_ids == id_mat).astype(np.float32)
            
            # CORRECCIÓN: Aplicar rotación 3D usando el centro de masa del volumen
            if abs(tilt) > 1e-5:
                print(f"  -> Aplicando tilteo de {tilt}° a la máscara de '{ws_materiales[id_mat]['nombre']}'...")
                mascara_tilt = rotate(mascara_base, angle=tilt, axes=(0, 1), reshape=False, order=1, prefilter=True)
            else:
                mascara_tilt = mascara_base
                
            print(f"  -> Proyectando textura de '{ws_materiales[id_mat]['nombre']}' en la geometría...")
            proy_mat = proyectar_coeficientes_textura(mascara_tilt, ws_materiales[id_mat]['C_bunge'], angulos_rad, config['tamano_voxel_cm'])
            proyecciones_C_tilts[tilt][id_mat] = proy_mat

    # Obtenemos dimensiones finales del detector
    res_z = proyecciones_C_tilts[config['angulos_tilt'][0]][1]['background'].shape[1]
    res_x = proyecciones_C_tilts[config['angulos_tilt'][0]][1]['background'].shape[2]
    
    print(f"\n[INFO] Workspace Multi-Material inicializado en {time.time() - t0:.2f} segundos.")
    
    return {
        'materiales': ws_materiales,
        'proyecciones_C_tilts': proyecciones_C_tilts,
        'volumen_ids': volumen_ids,
        'angulos_rad': angulos_rad,
        'angulos_theta': angulos_theta,
        'L_max': L_max,
        'sample_sym': sample_sym,
        'N_voxels': N_voxels,
        'resolucion_z': res_z,
        'resolucion_x': res_x,
        'CONFIG': config 
    }

if __name__ == '__main__':
    # =====================================================================
    # PANEL DE CONTROL (MAIN): Define AQUÍ tus Materiales y Geometría
    # =====================================================================
    CONFIGURACION_MAIN = {
        # --- 0. DEFINICIÓN DE RESOLUCIÓN DE LAMBDA (UNIFICADA) ---
        'LAMBDA_MIN': 1.0,
        'LAMBDA_MAX': 6.0,
        'NUM_LAMBDA_PUNTOS': 100,   # <--- CAMBIA AQUÍ LA RESOLUCIÓN (500, 1000, 2000)
        
        # --- 1. DEFINICIÓN MULTI-MATERIAL ---
        'materiales': {
            1: {
                'nombre': 'Aluminio_Nucleo',
                'base_dir': os.path.join('Exp_Data', 'Al'),
                'lattice_file': os.path.join('Exp_Data', 'Al', 'Al_lattice.txt'),
                'tex_path': os.path.join('Exp_Data', 'Al', 'odf_Bolmaro_ascii.txt')
            },
            2: {
                'nombre': 'Cobre_Coraza',
                'base_dir': os.path.join('Exp_Data', 'Cu'),
                'lattice_file': os.path.join('Exp_Data', 'Cu', 'Cu_lattice.txt'),
                'tex_path': os.path.join('Exp_Data', 'Cu', 'odf_Cu.txt')
            }
        },
        
        # --- 2. Parámetros Globales ---
        'L_max': 16,               
        'N_voxels': 70,           
        'radio_cilindro': 25,     
        'altura_cilindro': 40,    
        'tamano_voxel_cm': 0.02,  
        'num_proyecciones': 180,  
        
        'angulos_tilt': [0.0, 15.0, 30.0, 45.0],
        'tilt_interactivo': 0.0,                  
        
        # --- 3. Parámetros para Bloques de Análisis ---
        'lam_img_A': 2.5,  'ang_img_A': 0.0,
        'lam_img_B': 4.0,  'ang_img_B': 0.0,
        'lam_perfil': 4.0,
        'ang_espectro': 45.0,                         
        'angulos_espectro_multi': [0.0, 45.0, 90.0],  
        # LOS ESPECTROS INTERACTIVOS DEL VISUALIZADOR USARÁN EL RANGO DEFINIDO ARRIBA
        'lambdas_espectro': np.linspace(1.0, 6.0, 80), 
        'lam_detector': 2.5,
        'angulos_detector': [0.0, 45.0, 90.0],
        
        # --- 4. Exportación ---
        # ESTAS GRÁFICAS SE GENERAN CON LA MISMA RESOLUCIÓN QUE LOS B_LMU
        'lambdas_export_csv': np.linspace(1.0, 6.0, 1000),
        'lambdas_export_volumen': np.linspace(1.0, 6.0, 1000), 
        'exportar_csv': True,
        'exportar_npy': True,
        'exportar_proyecciones_almu': True   
    }
    
    ws = inicializar_workspace(CONFIGURACION_MAIN)

# %% [BLOQUE 3] INTERACTIVO: COMPARAR DOS IMÁGENES (Distintos Lambdas y/o Ángulos)
if __name__ == '__main__':
    try:
        tilt_int = ws['CONFIG']['tilt_interactivo']
        config_1 = {'lam': ws['CONFIG']['lam_img_A'], 'angulo': ws['CONFIG']['ang_img_A']}   
        config_2 = {'lam': ws['CONFIG']['lam_img_B'], 'angulo': ws['CONFIG']['ang_img_B']}
        
        print(f"\nCalculando imágenes Multi-Material (Tilteo: {tilt_int}°)...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, (nombre_cfg, cfg) in enumerate([('Imagen A', config_1), ('Imagen B', config_2)]):
            idx_angulo = (np.abs(ws['angulos_theta'] - cfg['angulo'])).argmin()
            ang_real = ws['angulos_theta'][idx_angulo]
            
            radiografias, lam_real = ensamblar_radiografias_multimaterial(cfg['lam'], ws, tilt_int)
            
            im = axes[i].imshow(radiografias[idx_angulo, :, :], cmap='viridis', vmin=0, vmax=1)
            axes[i].set_title(rf'{nombre_cfg}: $\lambda={lam_real:.2f}\AA$, $\theta={ang_real}^\circ$')
            plt.colorbar(im, ax=axes[i], label='$I/I_0$')
            axes[i].axis('off')
            
        plt.suptitle(f'Tomografía Multi-Material (Core-Shell) - Tilt={tilt_int}°', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    except NameError:
        print("\n[AVISO] Ejecuta primero el BLOQUE 2.")

# %% [BLOQUE 4] INTERACTIVO: PERFILES DEL DETECTOR 1D (Corte Transversal)
if __name__ == '__main__':
    try:
        lam_test_det = ws['CONFIG']['lam_detector']
        angulos_det = ws['CONFIG']['angulos_detector']
        tilt_int = ws['CONFIG']['tilt_interactivo']
        corte_z = ws['resolucion_z'] // 2
        
        print(f"\nCalculando perfiles 1D en detector a \\lambda={lam_test_det} A (Tilt: {tilt_int}°)...")
        radiografias_det, _ = ensamblar_radiografias_multimaterial(lam_test_det, ws, tilt_int)
        
        plt.figure(figsize=(9, 5))
        colores_det = ['darkblue', 'darkred', 'darkgreen', 'purple']
        
        for i, ang in enumerate(angulos_det):
            idx_ang = (np.abs(ws['angulos_theta'] - ang)).argmin()
            ang_real = ws['angulos_theta'][idx_ang]
            
            perfil_x = radiografias_det[idx_ang, corte_z, :]
            plt.plot(np.arange(ws['resolucion_x']), perfil_x, linewidth=2, color=colores_det[i % len(colores_det)], label=rf'$\theta={ang_real}^\circ$')
        
        plt.title(rf"Perfil de Transmisión Multi-Material (Corte Z={corte_z}) a $\lambda={lam_test_det}\AA$", fontsize=13)
        plt.xlabel("Posición X (Píxel)", fontsize=12)
        plt.ylabel("Transmisión ($I/I_0$)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except NameError:
        print("\n[AVISO] Ejecuta primero el BLOQUE 2.")

# %% [BLOQUE 5] INTERACTIVO: ESPECTROS (INTENSIDAD VS LAMBDA) PARA MÚLTIPLES PÍXELES
if __name__ == '__main__':
    try:
        ang_espectro = ws['CONFIG']['ang_espectro']
        lambdas_espectro = ws['CONFIG']['lambdas_espectro']
        tilt_int = ws['CONFIG']['tilt_interactivo']
        
        centro_z, centro_x = ws['resolucion_z'] // 2, ws['resolucion_x'] // 2
        pixeles = [
            {'x': centro_x,      'z': centro_z, 'label': 'Centro (Mat 1)'},
            {'x': centro_x + 15, 'z': centro_z, 'label': 'Borde Intermedio (Mat 2)'},
            {'x': centro_x + 30, 'z': centro_z, 'label': 'Fuera de Muestra'}
        ]
        
        idx_ang = (np.abs(ws['angulos_theta'] - ang_espectro)).argmin()
        
        print(f"\nCalculando espectros Multi-Material en $\\theta={ws['angulos_theta'][idx_ang]}^\\circ$...")
        for pix in pixeles: pix['espectro'] = []
        
        for lam in lambdas_espectro:
            rads_lam, _ = ensamblar_radiografias_multimaterial(lam, ws, tilt_int)
            for pix in pixeles:
                pix['espectro'].append(rads_lam[idx_ang, pix['z'], pix['x']])
                
        plt.figure(figsize=(10, 5))
        for i, pix in enumerate(pixeles):
            plt.plot(lambdas_espectro, pix['espectro'], linewidth=2, label=rf"{pix['label']} (X={pix['x']})")
        
        plt.title(rf"Espectros de Transmisión por Zona del Material ($\theta={ws['angulos_theta'][idx_ang]}^\circ$)", fontsize=14)
        plt.xlabel(r"Longitud de Onda $\lambda$ [$\AA$]")
        plt.ylabel(r"Transmisión ($I/I_0$)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except NameError:
        print("\n[AVISO] Ejecuta primero el BLOQUE 2.")

# %% [BLOQUE 8] EXPORTACIÓN DE DATOS 4D Y TOPOLOGÍA 3D
if __name__ == '__main__':
    try:
        out_dir_base = "Resultados_Tomografia"
        os.makedirs(out_dir_base, exist_ok=True)
        
        # ======================================================================
        # NUEVO: GUARDAR LA MATRIZ 3D DE TOPOLOGÍA PARA EL VISUALIZADOR
        # ======================================================================
        ruta_volumen_ids = os.path.join(out_dir_base, "Volumen_Materiales_IDs.npy")
        np.save(ruta_volumen_ids, ws['volumen_ids'])
        print(f"\n[ÉXITO] Matriz de Vóxeles 3D guardada en: {ruta_volumen_ids}")
        # ======================================================================

        if ws['CONFIG']['exportar_npy']:
            print(f"\n--- EXPORTACIÓN DE VOLÚMENES 4D (MULTI-MATERIAL) ---")
            for tilt in ws['CONFIG']['angulos_tilt']:
                out_dir_tilt = os.path.join(out_dir_base, f"Tilt_{tilt}deg")
                os.makedirs(out_dir_tilt, exist_ok=True)
                
                # AHORA USAMOS LA MISMA RESOLUCIÓN QUE LOS B_LMU
                lambdas_vol = ws['CONFIG']['lambdas_export_volumen']
                volumen_4d = np.zeros((len(lambdas_vol), len(ws['angulos_theta']), ws['resolucion_z'], ws['resolucion_x']), dtype=np.float32)
                
                print(f"Calculando volumen 4D para Tilt {tilt}°...")
                for i, lam in enumerate(lambdas_vol):
                    rads_tomo, _ = ensamblar_radiografias_multimaterial(lam, ws, tilt)
                    volumen_4d[i, :, :, :] = rads_tomo
                
                np.save(os.path.join(out_dir_tilt, 'Proyecciones_4D_Espectro.npy'), volumen_4d)
                print(f"  -> Guardado Tilt {tilt}° exitoso.")
                
    except NameError:
        print("\n[AVISO] Ejecuta primero el BLOQUE 2.")

# %% [BLOQUE 9] INTERACTIVO: EXTRACCIÓN Y GUARDADO DE PROYECCIONES EN EL DETECTOR (A_lmu) POR MATERIAL
if __name__ == '__main__':
    try:
        if ws['CONFIG'].get('exportar_proyecciones_almu', True):
            out_dir_almu_base = os.path.join("Resultados_Tomografia", "Proyecciones_Almu_Detector")
            print(f"\n--- EXTRACCIÓN DE PROYECCIONES A_lmu (SEGMENTADO POR MATERIAL) ---")
            
            for tilt in ws['CONFIG']['angulos_tilt']:
                print(f"\n=======================================================")
                print(f" EXTRAYENDO A_lmu PARA TILTEO: {tilt}°")
                
                for id_mat, info_mat in ws['materiales'].items():
                    print(f"  -> Extrayendo para Material {id_mat}: {info_mat['nombre']}")
                    
                    # Estructura de carpetas: Resultados/Proyecciones/Material_1_Al/Tilt_0deg/
                    out_dir_mat = os.path.join(out_dir_almu_base, f"Material_{id_mat}_{info_mat['nombre']}", f"Tilt_{tilt}deg")
                    
                    proy_C_mat = ws['proyecciones_C_tilts'][tilt][id_mat]
                    
                    proy_Almu_mat = extraer_proyecciones_Almu(
                        proy_C_mat, 
                        ws['angulos_rad'], 
                        ws['L_max'], 
                        ws['sample_sym'],
                        tilt_deg=tilt
                    )
                    
                    guardar_proyecciones_Almu(proy_Almu_mat, out_dir_mat)
            
            print("\n[ÉXITO] Proyecciones segmentadas extraídas y guardadas.")
            
    except NameError:
         print("\n[AVISO] Ejecuta primero el BLOQUE 2.")