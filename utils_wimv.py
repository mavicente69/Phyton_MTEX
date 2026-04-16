# -*- coding: utf-8 -*-
"""
Motor de Reconstrucción ODF - WIMV por Binning Discreto
Arquitectura: Block Coordinate Descent (WIMV Interno + NNLS Positivo Externo)
Mejora: Mapeo Isostático (Fibonacci) + Soft-Binning (Gaussiano K=5) + Parche Geométrico
Entorno: texturaPy3.10
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree
from orix.vector import Vector3d

from utils_so3 import SO3Grid

def fibonacci_hemisphere(n_points):
    """Genera una distribución equiespaciada de puntos en el hemisferio superior."""
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - indices / n_points) 
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def reconstruir_odf_wimv(lista_pfs_exp, fase_cristal, simetria_muestra=None, resolucion_grados=None, iteraciones=25):
    print(f"\n=========================================================")
    print(f"🧩 MOTOR DE INVERSIÓN WIMV (PANAL + SOFT-BINNING K=5)")
    print(f"=========================================================")
    
    simetria_real_usuario = simetria_muestra
    
    if resolucion_grados is None:
        puntos_promedio = np.mean([pf.direcciones.size for pf in lista_pfs_exp])
        res_rad = np.sqrt(2 * np.pi / puntos_promedio)
        res_ideal_deg = np.degrees(res_rad)
        resoluciones_estandar = np.array([2.5, 3.0, 5.0, 7.5, 10.0, 15.0])
        idx_cercano = np.argmin(np.abs(resoluciones_estandar - res_ideal_deg))
        resolucion_grados = resoluciones_estandar[idx_cercano]
        print(f" -> Resolución detectada : {resolucion_grados:.1f}°")

    # ====================================================================
    # 2. GENERAR GRILLA TOPOLÓGICA SO(3)
    # ====================================================================
    print(" -> Generando grilla topológica SO(3) en la Zona Fundamental...")
    grilla_so3 = SO3Grid(
        resolucion_grados=resolucion_grados, 
        simetria_cristal=fase_cristal.point_group, 
        simetria_muestra=simetria_real_usuario
    )
    
    orientaciones = grilla_so3.orientaciones
    n_ori = grilla_so3.N
    print(f" -> Espacio SO(3) generado : {n_ori} orientaciones base")
    
    pesos_odf = np.ones(n_ori)

    # ====================================================================
    # 3. PRE-NORMALIZACIÓN EXPERIMENTAL
    # ====================================================================
    print(" -> Aplicando Pre-Normalización por MÁXIMO a datos experimentales...")
    exp_vals_por_pf = []
    factores_pre_escala = []
    
    for pf in lista_pfs_exp:
        intensidad_maxima = np.max(pf.intensidades)
        factor_pre = 1.0 / intensidad_maxima if intensidad_maxima > 0 else 1.0
        factores_pre_escala.append(factor_pre)
        interp = NearestNDInterpolator(pf.direcciones.data, pf.intensidades * factor_pre)
        exp_vals_por_pf.append(interp)

    # ====================================================================
    # 4. PRE-CÁLCULO: MAPEO DE BINS WIMV (SOFT BINNING)
    # ====================================================================
    print(" -> Pre-calculando proyecciones (Soft-Binning Gaussiano)...")
    
    resolucion_rad = np.radians(resolucion_grados)
    area_bin = resolucion_rad**2
    n_bins = int(np.round(2 * np.pi / area_bin))
    
    v_centers = fibonacci_hemisphere(n_bins)
    arbol_fibonacci = cKDTree(v_centers)
    
    print(f"    * Nodos del panal generados : {n_bins}")
    
    pesos_bins = np.ones(n_bins)
    indices_por_pf = []
    b_targets_raw = []
    ops_muestra = simetria_real_usuario if simetria_real_usuario is not None else [None]
    
    # K-vecinos para distribuir el peso (suavizado espacial)
    K_VECINOS = 5 
    sigma_kernel = resolucion_rad * 0.85 # Ancho de la campana de Gauss
    
    # --- INICIO DEL PARCHE DE COHERENCIA GEOMÉTRICA ---
    pg = fase_cristal.point_group
    es_hexagonal = '6' in getattr(pg, 'name', str(pg)) or pg.size in [12, 24]
    
    orientaciones_eval = orientaciones
    if es_hexagonal:
        print("    * Aplicando parche de consistencia Hexagonal (+30° en phi2) al WIMV...")
        euler_eval = orientaciones.to_euler()
        euler_eval[:, 2] += np.radians(30)
        orientaciones_eval = type(orientaciones).from_euler(euler_eval, symmetry=pg)
    # --- FIN DEL PARCHE ---

    for i, pf in enumerate(lista_pfs_exp):
        interp = exp_vals_por_pf[i]
        val_exp_sym = np.zeros(v_centers.shape[0])
        n_ops = 0
        
        for op in ops_muestra:
            v_rot = (~op * Vector3d(v_centers)).data if op is not None else v_centers.copy()
            v_rot = np.where(v_rot[:, 2:3] < 0, -v_rot, v_rot) 
            val_exp_sym += np.nan_to_num(interp(v_rot), nan=0.0)
            n_ops += 1
            
        val_exp = val_exp_sym / n_ops
        val_exp = np.maximum(val_exp, 1e-4)
        
        max_bin = np.max(val_exp)
        if max_bin > 0: val_exp /= max_bin
        b_targets_raw.append(val_exp)

        sym_hkl = fase_cristal.point_group * pf.hkl
        normales_cristal = sym_hkl.data.astype(float)
        normas = np.linalg.norm(normales_cristal, axis=1, keepdims=True)
        normales_cristal = np.divide(normales_cristal, normas, out=np.zeros_like(normales_cristal), where=normas!=0)
        
        normales_upper = np.where(normales_cristal[:, 2:3] < 0, -normales_cristal, normales_cristal)
        normales_unicas = np.unique(normales_upper.round(4), axis=0)
        vecs_cristal = Vector3d(normales_unicas)
        
        # ACÁ APLICAMOS EL PARCHE: Usamos orientaciones_eval en lugar de orientaciones crudas
        polos_muestra = (~orientaciones_eval)[:, np.newaxis] * vecs_cristal 
        
        if simetria_real_usuario is not None:
            polos_lista = [(op * polos_muestra).data for op in simetria_real_usuario]
            coords = np.concatenate(polos_lista, axis=1).reshape(-1, 3).astype(float)
            n_sym_actual = vecs_cristal.size * simetria_real_usuario.size
        else:
            coords = polos_muestra.data.reshape(-1, 3).astype(float)
            n_sym_actual = vecs_cristal.size
            
        normas_c = np.linalg.norm(coords, axis=1, keepdims=True)
        coords = np.divide(coords, normas_c, out=np.zeros_like(coords), where=normas_c!=0)
        coords = np.where(coords[:, 2:3] < 0, -coords, coords) 

        # MAGIA DEL SOFT-BINNING: Consultamos los K vecinos más cercanos
        dist, lin_idx = arbol_fibonacci.query(coords, k=K_VECINOS)
        
        # Calculamos los pesos gaussianos basados en la distancia
        dist = np.maximum(dist, 1e-8)
        w_dist = np.exp(-0.5 * (dist / sigma_kernel)**2)
        w_dist /= np.sum(w_dist, axis=1, keepdims=True) # Normalizamos a 1
        
        # Guardamos diccionarios con índices y sus respectivos pesos fraccionales
        indices_por_pf.append({
            'idx': lin_idx.reshape(n_ori, n_sym_actual, K_VECINOS),
            'w_dist': w_dist.reshape(n_ori, n_sym_actual, K_VECINOS)
        })

    # =========================================================
    # 5. BUCLE GLOBAL (MACRO/MICRO ARQUITECTURA)
    # =========================================================
    print(" -> Optimizando sistema       : NNLS Positivo Externo + WIMV Interno...")
    
    macro_iteraciones = 5
    micro_iteraciones = iteraciones // macro_iteraciones
    if micro_iteraciones < 2: micro_iteraciones = 2
    
    factores_escala_wimv = np.ones(len(lista_pfs_exp))
    historial_escalas = []
    
    for macro in range(macro_iteraciones):
        
        # -----------------------------------------------------------------
        # CICLO INTERNO: WIMV Puro con Soft-Binning
        # -----------------------------------------------------------------
        for micro in range(micro_iteraciones):
            for i in range(len(lista_pfs_exp)):
                data_pf = indices_por_pf[i]
                lin_idx_3d = data_pf['idx']
                w_dist_3d = data_pf['w_dist']
                val_exp_target = b_targets_raw[i] * factores_escala_wimv[i]
                
                # Proyección Teórica difuminada en los K vecinos
                pesos_expand = pesos_odf[:, None, None] * w_dist_3d
                
                p_calc_sum = np.bincount(lin_idx_3d.ravel(), weights=pesos_expand.ravel(), minlength=n_bins)
                p_calc_count = np.bincount(lin_idx_3d.ravel(), weights=w_dist_3d.ravel(), minlength=n_bins)

                valid = p_calc_count > 1e-6
                p_calc = np.zeros(n_bins)
                p_calc[valid] = p_calc_sum[valid] / p_calc_count[valid]
                
                ratio_bins = np.ones(n_bins)
                ratio_bins[valid] = np.clip(val_exp_target[valid] / (p_calc[valid] + 1e-6), 0.1, 10.0)
                
                # Back-projection sumando los ratios fraccionados
                ratios_polos_k = ratio_bins[lin_idx_3d]
                ratios_sym = np.sum(ratios_polos_k * w_dist_3d, axis=2) 
                
                factor_correccion_odf = np.exp(np.mean(np.log(ratios_sym + 1e-8), axis=1))
                pesos_odf *= (1.0 + 0.5 * (factor_correccion_odf - 1.0))
                
            media_odf = np.mean(pesos_odf)
            if media_odf > 0:
                pesos_odf /= media_odf

        # -----------------------------------------------------------------
        # CICLO EXTERNO: NNLS (Optimiza Escalas Globales)
        # -----------------------------------------------------------------
        error_abs_macro = 0.0
        masa_exp_macro = 0.0
        
        for i in range(len(lista_pfs_exp)):
            data_pf = indices_por_pf[i]
            lin_idx_3d = data_pf['idx']
            w_dist_3d = data_pf['w_dist']
            val_exp_base = b_targets_raw[i]
            
            pesos_expand = pesos_odf[:, None, None] * w_dist_3d
            
            p_calc_sum = np.bincount(lin_idx_3d.ravel(), weights=pesos_expand.ravel(), minlength=n_bins)
            p_calc_count = np.bincount(lin_idx_3d.ravel(), weights=w_dist_3d.ravel(), minlength=n_bins)
            
            valid = p_calc_count > 1e-6
            p_calc = np.zeros(n_bins)
            p_calc[valid] = p_calc_sum[valid] / p_calc_count[valid]
            
            numerador = np.sum(p_calc[valid] * val_exp_base[valid] * pesos_bins[valid])
            denominador = np.sum((val_exp_base[valid]**2) * pesos_bins[valid])
            
            if denominador > 0:
                nuevo_escala = numerador / denominador
                factores_escala_wimv[i] = nuevo_escala
                
            val_exp_escalado = val_exp_base * factores_escala_wimv[i]
            error_abs_macro += np.sum(np.abs(p_calc[valid] - val_exp_escalado[valid]) * pesos_bins[valid])
            masa_exp_macro += np.sum(val_exp_escalado[valid] * pesos_bins[valid])

        historial_escalas.append(factores_escala_wimv.copy())

        rp_error = (error_abs_macro / masa_exp_macro) * 100.0 if masa_exp_macro > 0 else 0.0
        escalas_str = np.array2string(factores_escala_wimv, precision=3, separator=', ', suppress_small=True)
        print(f"    [Macro-Ciclo {macro+1:02d}/{macro_iteraciones}] RP Error: {rp_error:4.1f}% | Escalas NNLS: {escalas_str}")

    print("--- Reconstrucción exitosa ---")
    
    # ====================================================================
    # REPORTE TABULAR DE EVOLUCIÓN DE ESCALAS
    # ====================================================================
    print("\n=========================================================")
    print("📈 EVOLUCIÓN DE LOS PARÁMETROS DE NORMALIZACIÓN (WIMV)")
    print("=========================================================")
    nombres_hkl = [f"{{{ ''.join([str(int(x)) for x in pf.hkl.hkil.flatten()]) }}}" for pf in lista_pfs_exp]
    encabezado = "Macro | " + " | ".join([f"{nombre:>8}" for nombre in nombres_hkl])
    print(encabezado)
    print("-" * len(encabezado))
    
    for iteracion, escalas in enumerate(historial_escalas):
        fila = f"  {iteracion+1:02d}  | " + " | ".join([f"{val:8.4f}" for val in escalas])
        print(fila)
    print("=========================================================\n")
    
    print(f" -> Sincronizando escalas MUD experimentales vs teóricas...")
    for i, pf in enumerate(lista_pfs_exp):
        factor_total = factores_pre_escala[i] * factores_escala_wimv[i]
        pf.intensidades *= factor_total

    return orientaciones, pesos_odf