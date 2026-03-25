# -*- coding: utf-8 -*-
"""
Motor de Reconstrucción ODF - WIMV por Binning Discreto
Arquitectura: Block Coordinate Descent (WIMV Interno + NNLS Positivo Externo)
Entorno: texturaPy3.10
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from orix.vector import Vector3d

from utils_so3 import SO3Grid

def reconstruir_odf_wimv(lista_pfs_exp, fase_cristal, simetria_muestra=None, resolucion_grados=None, iteraciones=25):
    print(f"\n=========================================================")
    print(f"🧩 MOTOR DE INVERSIÓN WIMV (ARQUITECTURA MACRO/MICRO)")
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
    # 3. PRE-NORMALIZACIÓN EXPERIMENTAL (MÁXIMO = 1.0)
    # ====================================================================
    print(" -> Aplicando Pre-Normalización por MÁXIMO a datos experimentales...")
    exp_vals_por_pf = []
    factores_pre_escala = []
    
    for pf in lista_pfs_exp:
        # Buscamos el valor máximo absoluto de la PF
        intensidad_maxima = np.max(pf.intensidades)
        factor_pre = 1.0 / intensidad_maxima if intensidad_maxima > 0 else 1.0
        
        factores_pre_escala.append(factor_pre)
        
        interp = NearestNDInterpolator(pf.direcciones.data, pf.intensidades * factor_pre)
        exp_vals_por_pf.append(interp)

    # ====================================================================
    # 4. PRE-CÁLCULO: MAPEO DE BINS WIMV
    # ====================================================================
    print(" -> Pre-calculando tabla de proyecciones (Binning)...")
    paso_pf = resolucion_grados 
    p_bins = np.arange(0, 90 + paso_pf, paso_pf)
    a_bins = np.arange(0, 360 + paso_pf, paso_pf)
    n_bins = (len(p_bins) - 1) * (len(a_bins) - 1)

    p_centers = (p_bins[:-1] + p_bins[1:]) / 2
    a_centers = (a_bins[:-1] + a_bins[1:]) / 2
    A_grid, P_grid = np.meshgrid(np.radians(a_centers), np.radians(p_centers))
    v_centers = np.column_stack((np.sin(P_grid.ravel()) * np.cos(A_grid.ravel()), 
                                 np.sin(P_grid.ravel()) * np.sin(A_grid.ravel()), 
                                 np.cos(P_grid.ravel())))

    pesos_bins = np.sin(P_grid.ravel())
    pesos_bins[pesos_bins == 0] = 1e-6

    indices_por_pf = []
    b_targets_raw = []
    
    ops_muestra = simetria_real_usuario if simetria_real_usuario is not None else [None]

    for i, pf in enumerate(lista_pfs_exp):
        interp = exp_vals_por_pf[i]
        val_exp_sym = np.zeros(v_centers.shape[0])
        n_ops = 0
        
        for op in ops_muestra:
            if op is not None:
                v_rot = (~op * Vector3d(v_centers)).data
            else:
                v_rot = v_centers.copy()
            v_rot = np.where(v_rot[:, 2:3] < 0, -v_rot, v_rot) 
            val_exp_sym += np.nan_to_num(interp(v_rot), nan=0.0)
            n_ops += 1
            
        val_exp = val_exp_sym / n_ops
        val_exp = np.maximum(val_exp, 1e-4)
        
        # Normalizamos la grilla binned para asegurar que el máximo sea exactamente 1.0
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
        
        polos_muestra = (~orientaciones)[:, np.newaxis] * vecs_cristal 
        
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

        polar = np.degrees(np.arccos(np.clip(coords[:, 2], -1.0, 1.0)))
        azimut = np.mod(np.degrees(np.arctan2(coords[:, 1], coords[:, 0])), 360)

        p_idx = np.clip(np.digitize(polar, p_bins) - 1, 0, len(p_bins)-2)
        a_idx = np.clip(np.digitize(azimut, a_bins) - 1, 0, len(a_bins)-2)
        
        lin_idx = p_idx * (len(a_bins) - 1) + a_idx
        indices_por_pf.append(lin_idx.reshape(n_ori, n_sym_actual))

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
        # CICLO INTERNO: WIMV Puro (Optimiza forma ODF, Escalas Congeladas)
        # -----------------------------------------------------------------
        for micro in range(micro_iteraciones):
            for i in range(len(lista_pfs_exp)):
                lin_idx_2d = indices_por_pf[i]
                val_exp_target = b_targets_raw[i] * factores_escala_wimv[i]
                
                n_sym_actual = lin_idx_2d.shape[1]
                pesos_exp_expand = np.repeat(pesos_odf, n_sym_actual)
                p_calc_sum = np.bincount(lin_idx_2d.ravel(), weights=pesos_exp_expand, minlength=n_bins)
                p_calc_count = np.bincount(lin_idx_2d.ravel(), minlength=n_bins)

                valid = p_calc_count > 0
                p_calc = np.zeros(n_bins)
                p_calc[valid] = p_calc_sum[valid] / p_calc_count[valid]
                
                ratio_bins = np.ones(n_bins)
                ratio_bins[valid] = np.clip(val_exp_target[valid] / (p_calc[valid] + 1e-6), 0.1, 10.0)
                
                ratios_polos = ratio_bins[lin_idx_2d]
                factor_correccion_odf = np.exp(np.mean(np.log(ratios_polos + 1e-8), axis=1))
                
                pesos_odf *= (1.0 + 0.5 * (factor_correccion_odf - 1.0))
                
            media_odf = np.mean(pesos_odf)
            if media_odf > 0:
                pesos_odf /= media_odf

        # -----------------------------------------------------------------
        # CICLO EXTERNO: NNLS (Optimiza Escalas Globales, ODF Congelada)
        # -----------------------------------------------------------------
        error_abs_macro = 0.0
        masa_exp_macro = 0.0
        
        for i in range(len(lista_pfs_exp)):
            lin_idx_2d = indices_por_pf[i]
            val_exp_base = b_targets_raw[i]
            
            n_sym_actual = lin_idx_2d.shape[1]
            pesos_exp_expand = np.repeat(pesos_odf, n_sym_actual)
            p_calc_sum = np.bincount(lin_idx_2d.ravel(), weights=pesos_exp_expand, minlength=n_bins)
            p_calc_count = np.bincount(lin_idx_2d.ravel(), minlength=n_bins)
            
            valid = p_calc_count > 0
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
    
    # ====================================================================
    # 6. ACTUALIZACIÓN DE LOS DATOS EXPERIMENTALES
    # ====================================================================
    print(f" -> Sincronizando escalas MUD experimentales vs teóricas...")
    for i, pf in enumerate(lista_pfs_exp):
        factor_total = factores_pre_escala[i] * factores_escala_wimv[i]
        pf.intensidades *= factor_total
        print(f"    * PF {pf.hkl} corregida por factor NNLS: {factor_total:.3e}")

    return orientaciones, pesos_odf