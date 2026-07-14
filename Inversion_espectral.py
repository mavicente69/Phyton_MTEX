# -*- coding: utf-8 -*-
"""
Inversor Espectral Jerárquico (H-ALS) para Tomografía Tensorial de Neutrones.
Entorno: texturaPy3.10

MODIFICADO: Se eliminaron las matemáticas locales redundantes. Ahora utiliza estrictamente
las funciones importadas de los módulos base ('calc_secciones_background', 
'calc_sigma_elastic_coherent' y 'Blmu_matrix') asegurando paridad física con el Simulador.
"""
import os
import time
import gc
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ====================================================================
# IMPORTACIÓN DE MÓDULOS DEL USUARIO (SINGLE SOURCE OF TRUTH)
# ====================================================================
from calc_sigma_elastic_coherent import leer_red_cristalina
from calc_secciones_background import calcular_background_total
from Blmu_matrix import calcular_B_lmu

# ====================================================================
# GESTOR DE BASES TEÓRICAS
# ====================================================================
def obtener_bases_teoricas(cfg_material, lambdas, L_max):
    """
    Utiliza los módulos de texturaPy3.10 para calcular el fondo no elástico 
    y extraer las matrices B_lmu. Lee/Genera un archivo CSV para máxima velocidad.
    """
    lattice_file = cfg_material['lattice_file']
    nombre_mat = cfg_material['nombre']
    
    # 1. Extracción Cristalográfica usando el módulo oficial
    simetria_str, A_matrix, basis, bg_coeffs = leer_red_cristalina(lattice_file)
    v0 = np.abs(np.linalg.det(A_matrix))
    
    # Fallback si el archivo TXT no tiene BACKGROUND definido
    if len(bg_coeffs) < 3:
        bg_coeffs = np.array([0.1, 0.01, 0.05])
        
    # 2. Atenuación de fondo (Background No-Elástico)
    sigma_bg_barns = calcular_background_total(lambdas, bg_coeffs)
    sigma_bg_cm1 = sigma_bg_barns / v0
    
    # 3. Matrices B_lmu (Dinámicas vía CSV)
    base_dir = os.path.dirname(lattice_file)
    csv_file = os.path.join(base_dir, f"B_lmu_{nombre_mat}.csv")
    
   # Generamos la tabla con 500 pasos, luego interpolaremos a lo que requiera el Inversor
    calcular_B_lmu(lattice_file, csv_file, L_max, np.min(lambdas), np.max(lambdas), 500)
        
    # Lectura e Interpolación (Idéntico a Visualizacion_datos_de_tomografia_2.py)
    df_B = pd.read_csv(csv_file)
    lambdas_B = df_B['lambda_A'].values
    
    B_lmu_dict = {}
    for col in df_B.columns:
        if col.startswith('B_') and col.endswith('_Real'):
            parts = col.split('_')
            l_val, mu_val = int(parts[1]), int(parts[2])
            
            # Solo importamos hasta el L_max configurado en el Inversor
            if l_val > L_max: continue
            
            col_imag = f'B_{l_val}_{mu_val}_Imag'
            b_real = df_B[col].values
            b_imag = df_B[col_imag].values if col_imag in df_B.columns else np.zeros_like(lambdas_B)
            
            # Normalizamos por v0 para tener unidades en cm^-1
            B_complex = (b_real + 1j * b_imag) / v0
            
            # Extrapolamos linealmente a la grilla exacta de lambdas solicitada
            f_real = interp1d(lambdas_B, B_complex.real, kind='linear', fill_value='extrapolate')
            f_imag = interp1d(lambdas_B, B_complex.imag, kind='linear', fill_value='extrapolate')
            
            B_lmu_dict[(l_val, mu_val)] = f_real(lambdas) + 1j * f_imag(lambdas)
            
    return B_lmu_dict, sigma_bg_cm1

# ====================================================================
# MOTOR DE INVERSIÓN ESPECTRAL (H-ALS: JERÁRQUICO ITERATIVO)
# ====================================================================
def ejecutar_inversion(config):
    print("="*65)
    print(" INVERSOR ESPECTRAL JERÁRQUICO (H-ALS)")
    print("="*65)
    t0_total = time.time()
    
    dir_entrada = config['dir_entrada']
    dir_salida = config['dir_salida']
    lambdas_cfg = config['lambdas']
    L_max = config['L_max']
    materiales = config['materiales']
    plot_diagnostico = config.get('plot_diagnostico', True)
    exportar_espesor = config.get('exportar_espesor', True)
    
    max_iteraciones = config.get('iteraciones_als', 10) 
    lr = config.get('tasa_aprendizaje', 0.5) 
    
    os.makedirs(dir_salida, exist_ok=True)
    out_dir_almu_base = os.path.join(dir_salida, "Proyecciones_Almu_Invertidas")
    out_dir_espesor_base = os.path.join(dir_salida, "Proyecciones_Espesor")
    
    # --- AUTO-CORRECCIÓN DE DIMENSIÓN ---
    archivo_4d_test = next((os.path.join(dir_entrada, f"Tilt_{t}deg", "Proyecciones_4D_Espectro.npy") 
                            for t in config['angulos_tilt'] 
                            if os.path.exists(os.path.join(dir_entrada, f"Tilt_{t}deg", "Proyecciones_4D_Espectro.npy"))), None)
            
    if archivo_4d_test:
        N_lam_real = np.load(archivo_4d_test, mmap_mode='r').shape[0]
        lambdas = np.linspace(lambdas_cfg[0], lambdas_cfg[-1], N_lam_real)
        print(f"[INFO] Grilla ajustada a {N_lam_real} puntos de energía según archivo 4D.")
    else:
        lambdas = lambdas_cfg

    # 1. Generar Bases Teóricas (AHORA USANDO TUS MÓDULOS)
    print("\n[FASE 1] Cargando Bases Teóricas (vía Módulos Externos)...")
    bases_mat = {}
    factor_bunge_L0 = 1.0 / np.sqrt(4 * np.pi)
    
    for id_mat, cfg in materiales.items():
        print(f"  -> Obteniendo físicas para: {cfg['nombre']}")
        # CORREGIDO: Se removió id_mat de los argumentos, la función solo recibe 3.
        B_lmu_dict, sigma_bg = obtener_bases_teoricas(cfg, lambdas, L_max)
        bases_mat[id_mat] = {'B': B_lmu_dict, 'bg': sigma_bg}

    # 2. Separación Jerárquica de Bases (Bloques por L)
    print("\n[FASE 2] Construyendo Bloques Jerárquicos de Ajuste (H-ALS)...")
    mat_ids = list(materiales.keys())
    
    H_iso_cols = []
    H_aniso_cols = []
    aniso_keys = []
    
    for m in mat_ids:
        # Base Isótropa (A_00 / Espesor)
        B00_real = np.real(bases_mat[m]['B'].get((0,0), np.zeros_like(lambdas)))
        basis_iso = bases_mat[m]['bg'] + B00_real * factor_bunge_L0
        H_iso_cols.append(basis_iso)
        
        # Bases Anisótropas (L > 0)
        for (l, mu), B_comp in bases_mat[m]['B'].items():
            if l == 0: continue
            if np.any(np.abs(np.real(B_comp)) > 1e-8):
                H_aniso_cols.append(np.real(B_comp))
                aniso_keys.append({'mat': m, 'l': l, 'mu': mu, 'part': 'real'})
            if np.any(np.abs(np.imag(B_comp)) > 1e-8):
                H_aniso_cols.append(-np.imag(B_comp))
                aniso_keys.append({'mat': m, 'l': l, 'mu': mu, 'part': 'imag'})
                
    H_iso = np.column_stack(H_iso_cols)
    H_aniso = np.column_stack(H_aniso_cols) if H_aniso_cols else None

    # --- PRECOMPUTACIÓN DE BLOQUES POR ORDEN L ---
    L_groups = {} 
    pinvs_L = {}  
    
    if H_aniso is not None:
        alpha_reg = config.get('penalizacion_alpha', 0.001)
        tipo_reg = config.get('tipo_regularizacion', 'tikhonov_L')
        
        for k_idx, key in enumerate(aniso_keys):
            l_val = key['l']
            if l_val not in L_groups: L_groups[l_val] = []
            L_groups[l_val].append(k_idx)
            
        print(f"  -> Bloques de armónicos detectados: {list(sorted(L_groups.keys()))}")
            
        for l_val in sorted(L_groups.keys()):
            indices_l = L_groups[l_val]
            H_bloque_l = H_aniso[:, indices_l]
            
            W_diag = np.zeros(len(indices_l))
            for i in range(len(indices_l)):
                if tipo_reg == 'tikhonov_L':
                    W_diag[i] = l_val * (l_val + 1)
                elif tipo_reg == 'exponencial':
                    W_diag[i] = np.exp(l_val / 2.0) 
                else:
                    W_diag[i] = 1.0
                    
            W_matrix = np.diag(W_diag)
            H_T_H = np.dot(H_bloque_l.T, H_bloque_l)
            pinvs_L[l_val] = np.dot(np.linalg.inv(H_T_H + alpha_reg * W_matrix), H_bloque_l.T)

    # 3. Inversión Tilteo por Tilteo
    print(f"\n[FASE 3] Iniciando Inversión Iterativa Jerárquica (H-ALS - {max_iteraciones} iteraciones, Tasa_Ajuste={lr})...")
    for tilt in config['angulos_tilt']:
        archivo_4d = os.path.join(dir_entrada, f"Tilt_{tilt}deg", "Proyecciones_4D_Espectro.npy")
        if not os.path.exists(archivo_4d): continue
            
        print(f"\n  ================== TILTEO {tilt}° ==================")
        t0_tilt = time.time()
        
        data_4d = np.load(archivo_4d, mmap_mode='r')
        N_lam, N_theta, Nz, Nx = data_4d.shape
        N_pixels = N_theta * Nz * Nx
        
        Y_meas = -np.log(np.clip(data_4d, 1e-10, 1.0))
        Y_flat = Y_meas.reshape(N_lam, N_pixels) 
        
        pixeles_activos = np.where(np.max(Y_flat, axis=0) > 1e-4)[0]
        N_activos = len(pixeles_activos)
        Y_activos = Y_flat[:, pixeles_activos]
        
        print(f"  -> {N_activos} píxeles activos. Resolviendo H-ALS en cascada L...")
        
        A_iso_act = np.zeros((len(mat_ids), N_activos), dtype=np.float32)
        A_aniso_act = np.zeros((len(H_aniso_cols), N_activos), dtype=np.float32) if H_aniso is not None else []
        
        # BUCLE PRINCIPAL DE ITERACIÓN
        for it in range(max_iteraciones):
            # PASO A: Ajuste Base de Espesores (NNLS)
            if H_aniso is not None:
                Y_para_iso = Y_activos - np.dot(H_aniso, A_aniso_act)
            else:
                Y_para_iso = Y_activos
                
            for i in range(N_activos):
                A_iso_new, _ = nnls(H_iso, Y_para_iso[:, i])
                # Amortiguación (Damping) en la componente Isótropa
                A_iso_act[:, i] = (1.0 - lr) * A_iso_act[:, i] + lr * A_iso_new
                
            # PASO B: Ajuste de Textura en Cascada (L Crecientes)
            if H_aniso is not None:
                Y_residuo_iso = Y_activos - np.dot(H_iso, A_iso_act)
                
                for l_val in sorted(L_groups.keys()):
                    indices_l = L_groups[l_val]
                    H_l = H_aniso[:, indices_l]
                    
                    Y_aniso_total = np.dot(H_aniso, A_aniso_act)
                    Y_aniso_este_L = np.dot(H_l, A_aniso_act[indices_l, :])
                    Y_aniso_otros = Y_aniso_total - Y_aniso_este_L
                    
                    Y_objetivo_l = Y_residuo_iso - Y_aniso_otros
                    
                    A_l_new = np.dot(pinvs_L[l_val], Y_objetivo_l)
                    # Amortiguación (Damping)
                    A_aniso_act[indices_l, :] = (1.0 - lr) * A_aniso_act[indices_l, :] + lr * A_l_new

        # Reconstruimos la matriz final para todos los píxeles
        A_00 = np.zeros((len(mat_ids), N_pixels), dtype=np.float32)
        A_00[:, pixeles_activos] = A_iso_act
        
        A_aniso_flat = np.zeros((len(H_aniso_cols), N_pixels), dtype=np.float32)
        if H_aniso is not None:
            A_aniso_act[np.abs(A_aniso_act) < 1e-5] = 0.0
            A_aniso_flat[:, pixeles_activos] = A_aniso_act
            Y_fitted = np.dot(H_iso, A_00) + np.dot(H_aniso, A_aniso_flat)
        else:
            Y_fitted = np.dot(H_iso, A_00)

        # --- DIAGNÓSTICO VISUAL ---
        if plot_diagnostico and N_activos > 0:
            print("  -> Generando Gráficos de Diagnóstico (3 puntos clave)...")
            theta_idx = 0 
            z_c = Nz // 2
            x_c = Nx // 2
            
            idx_start = theta_idx * (Nz * Nx) + z_c * Nx
            idx_end = idx_start + Nx
            
            perfil_x = np.max(Y_flat[:, idx_start:idx_end], axis=0)
            x_activos = np.where(perfil_x > 1e-4)[0]
            
            if len(x_activos) > 0:
                x_borde_ext = x_activos[max(0, len(x_activos) - 3)] 
                x_borde_int = x_c + (x_borde_ext - x_c) // 2 
                
                pixeles_a_plotear = {
                    f"Centro (X={x_c})": idx_start + x_c,
                    f"Borde Interno / Interfaz (X={x_borde_int})": idx_start + x_borde_int,
                    f"Borde Externo (X={x_borde_ext})": idx_start + x_borde_ext
                }
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
                
                for ax, (label_punto, p_idx) in zip(axes, pixeles_a_plotear.items()):
                    if p_idx not in pixeles_activos:
                        ax.set_title(f"{label_punto}\n(Sin Muestra)")
                        continue
                        
                    ax.plot(lambdas, Y_flat[:, p_idx], 'k.', label='Verdad (Simulador)', alpha=0.5)
                    ax.plot(lambdas, Y_fitted[:, p_idx], 'r-', lw=2.5, label='Ajuste Final (H-ALS)')
                    
                    for i, m in enumerate(mat_ids):
                        bg_individual = H_iso_cols[i] * A_00[i, p_idx]
                        ax.plot(lambdas, bg_individual, ':', lw=2, label=f'Base Isótropa {materiales[m]["nombre"]}')
                    
                    ax.set_title(f"Píxel: {label_punto}")
                    ax.set_xlabel("Longitud de Onda [$\AA$]")
                    if ax == axes[0]:
                        ax.set_ylabel("Atenuación Macroscópica")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=9)
                    
                plt.suptitle(f"Diagnóstico Jerárquico H-ALS (L_max={L_max}) - Tilt {tilt}°", fontweight='bold', fontsize=15)
                plt.tight_layout()
                plt.show(block=True)
            
        # --- EMPAQUETADO Y GUARDADO ---
        print("  -> Guardando proyecciones invertidas organizadas por material...")
        
        for i, m in enumerate(mat_ids):
            nombre_mat = materiales[m]['nombre']
            out_dir_mat = os.path.join(out_dir_almu_base, f"Material_{m}_{nombre_mat}", f"Tilt_{tilt}deg")
            os.makedirs(out_dir_mat, exist_ok=True)
            
            if exportar_espesor:
                out_dir_espesor = os.path.join(out_dir_espesor_base, f"Material_{m}_{nombre_mat}", f"Tilt_{tilt}deg")
                os.makedirs(out_dir_espesor, exist_ok=True)
                espesor_map = A_00[i].reshape(N_theta, Nz, Nx).astype(np.float32)
                np.save(os.path.join(out_dir_espesor, "Sinograma_Espesor_cm.npy"), espesor_map)
                
            proy_Almu = {}
            A00_map = (A_00[i] * factor_bunge_L0).reshape(N_theta, Nz, Nx).astype(np.float32)
            proy_Almu[(0,0)] = (A00_map, np.zeros_like(A00_map))
            
            for k_idx, key in enumerate(aniso_keys):
                if key['mat'] != m: continue
                l, mu = key['l'], key['mu']
                if (l, mu) not in proy_Almu:
                    proy_Almu[(l, mu)] = [np.zeros((N_theta, Nz, Nx), dtype=np.float32),
                                          np.zeros((N_theta, Nz, Nx), dtype=np.float32)]
                                          
                mapa = A_aniso_flat[k_idx].reshape(N_theta, Nz, Nx).astype(np.float32)
                if key['part'] == 'real':
                    proy_Almu[(l,mu)][0] = mapa
                elif key['part'] == 'imag':
                    proy_Almu[(l,mu)][1] = mapa
                    
            for (l, mu), (proj_real, proj_imag) in proy_Almu.items():
                proj_final = proj_real + 1j * proj_imag if np.max(np.abs(proj_imag)) > 1e-8 else proj_real
                np.save(os.path.join(out_dir_mat, f"Proyecciones_Detector_Almu_L{l}_mu{mu}.npy"), proj_final)
                
        # Limpieza masiva
        if 'Y_fitted' in locals(): del Y_fitted
        del data_4d, Y_meas, Y_flat, Y_activos, A_iso_act, A_aniso_act
        gc.collect()
        
        print(f"  [OK] Tilteo {tilt}° resuelto y empaquetado en {time.time()-t0_tilt:.2f}s")
        
    print(f"\n[INFO] Inversión total completada en {time.time() - t0_total:.2f} segundos.")

if __name__ == '__main__':
    # =====================================================================
    # CONFIGURACIÓN DEL INVERSOR
    # =====================================================================
    CONFIG_INVERSION = {
        'dir_entrada': r'Resultados_Tomografia', 
        'dir_salida': r'Resultados_Inversion_Almu', 
        
        'plot_diagnostico': True,  
        'exportar_espesor': True,  
        
        # Iteraciones del algoritmo ALS
        'iteraciones_als': 10,
        
        'tasa_aprendizaje': 0.5,
        
        # 'exponencial' 'nada'  'tikhonov_L'
        'tipo_regularizacion': 'nada', 
        'penalizacion_alpha': 0.001,
        
        'lambdas': np.linspace(1.0, 6.0, 300), 
        'angulos_tilt': [0.0, 15.0, 30.0, 45.0],
        
        'L_max': 0, 
        
        'materiales': {
            1: {
                'nombre': 'Aluminio_Nucleo',
                'lattice_file': os.path.join('Exp_Data', 'Al', 'Al_lattice.txt')
            },
            2: {
                'nombre': 'Cobre_Coraza',
                'lattice_file': os.path.join('Exp_Data', 'Cu', 'Cu_lattice.txt')
            }
        }
    }
    
    ejecutar_inversion(CONFIG_INVERSION)