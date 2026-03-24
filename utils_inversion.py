# -*- coding: utf-8 -*-
"""
Motor de Inversión de Textura Continuo (Estilo MTEX)
Algoritmo: M-ART (Richardson-Lucy) con BLAS Multi-hilo
Modelo: Phon Estricto extraído del piso MUD global de las PFs
Calibración: Isotropía MUD Estricta con Topología SO(3) Equiespaciada
Estabilidad: Pre-Normalización MUD + Escalador Híbrido + Filtro Dinámico
Entorno: texturaPy3.10
"""
import numpy as np
import time
from orix.quaternion import Orientation

from utils_odf import ODFComponent, ODFIsotropic
from utils_kernels import OrientationKernel
from utils_so3 import SO3Grid

def reconstruir_odf_nnls(lista_pfs_exp, fase_cristal, simetria_muestra=None, tipo_kernel='gaussian', resolucion_grados=None):
    print(f"\n=========================================================")
    print(f"🧠 MOTOR DE INVERSIÓN (M-ART + SO3 + FAST-MATRIX)")
    print(f"=========================================================")
    t_total_inicio = time.time()
    
    simetria_real_usuario = simetria_muestra
    
    if resolucion_grados is None:
        pts_promedio = np.mean([pf.direcciones.size for pf in lista_pfs_exp])
        espaciado_rad = np.sqrt(2 * np.pi / pts_promedio)
        espaciado_deg = np.degrees(espaciado_rad)
        if espaciado_deg <= 3.5: resolucion_grados = 5.0
        elif espaciado_deg <= 7.0: resolucion_grados = 7.5
        else: resolucion_grados = 10.0
        print(f" -> Auto-Resolución detectada : {resolucion_grados:.1f}°")
    
    fwhm = resolucion_grados * 1.5 
    print(f" -> FWHM del kernel dinámico  : {fwhm:.1f}°")
    
    kernel = OrientationKernel(tipo=tipo_kernel, fwhm_grados=fwhm)
    
    # ====================================================================
    # 1. GENERAR GRILLA TOPOLÓGICA SO(3)
    # ====================================================================
    print(" -> Generando grilla topológica SO(3) en la Zona Fundamental...")
    grilla_so3 = SO3Grid(
        resolucion_grados=resolucion_grados, 
        simetria_cristal=fase_cristal.point_group, 
        simetria_muestra=simetria_real_usuario
    )
    
    oris_base = grilla_so3.orientaciones
    N = grilla_so3.N
    print(f" -> Espacio SO(3) generado : {N} orientaciones base")
    
    # ====================================================================
    # 2. PREPARAR VECTOR EXPERIMENTAL (b) Y PRE-NORMALIZACIÓN MUD
    # ====================================================================
    print(" -> Aplicando Pre-Normalización MUD a datos de entrada...")
    b_original_list = []
    pf_limites = []
    idx_actual = 0
    factores_pre_escala = []

    for pf in lista_pfs_exp:
        z_coords = pf.direcciones.data[:, 2]
        angulos_polares = np.arccos(np.clip(np.abs(z_coords), 0.0, 1.0))
        pesos_area = np.sin(angulos_polares)
        pesos_area[pesos_area == 0] = 1e-6
        
        intensidad_media = np.sum(pf.intensidades * pesos_area) / np.sum(pesos_area)
        factor_pre = 1.0 / intensidad_media if intensidad_media > 0 else 1.0
        
        b_norm = pf.intensidades * factor_pre
        b_original_list.append(b_norm)
        factores_pre_escala.append(factor_pre)
        
        size = pf.direcciones.size
        pf_limites.append((idx_actual, idx_actual + size))
        idx_actual += size

    b_original = np.concatenate(b_original_list)
    M_total = b_original.size
    print(f" -> Píxeles experimentales    : {M_total} (Vector b)")
    
    # ====================================================================
    # 3. PRE-NORMALIZACIÓN ESTRICTA DE VECTORES
    # ====================================================================
    pg = fase_cristal.point_group
    pfs_crystal_poles = []
    pfs_direcciones = []
    
    for pf in lista_pfs_exp:
        p_c = (pg * pf.hkl).data.astype(np.float32)
        n_c = np.linalg.norm(p_c, axis=1, keepdims=True)
        n_c[n_c == 0] = 1.0
        pfs_crystal_poles.append(p_c / n_c)
        
        v = pf.direcciones.data.astype(np.float32)
        n_v = np.linalg.norm(v, axis=1, keepdims=True)
        n_v[n_v == 0] = 1.0
        pfs_direcciones.append(v / n_v)

    # ====================================================================
    # 4. CONSTRUIR MATRIZ DE PROYECCIÓN (A) OPTIMIZADA (FAST-MATRIX)
    # ====================================================================
    print(f" -> Construyendo Matriz A     : [{M_total} x {N}] (Con Fast-Mask)...")
    t_matriz_inicio = time.time()
    
    A = np.zeros((M_total, N), dtype=np.float32)
    
    batch_size = 500
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_oris = oris_base[start:end]
        
        es_hexagonal = '6' in getattr(pg, 'name', str(pg)) or pg.size in [12, 24]
        batch_eval = batch_oris
        if es_hexagonal:
            euler_batch = batch_eval.to_euler()
            euler_batch[:, 2] += np.radians(30)
            batch_eval = type(batch_eval).from_euler(euler_batch, symmetry=pg)
            
        R_c2s = (~batch_eval).to_matrix().astype(np.float32)
        
        fila_inicio = 0
        for idx in range(len(lista_pfs_exp)):
            M_pf = pfs_direcciones[idx].shape[0]
            v_g = pfs_direcciones[idx]
            p_c = pfs_crystal_poles[idx]
            
            polos_data_base = np.einsum('bij,pj->bpi', R_c2s, p_c)
            polos_flat = polos_data_base.reshape(-1, 3) 
            
            # Cálculo rápido:
            dot_prod_flat = np.abs(np.dot(polos_flat, v_g.T)) 
            vals_flat = np.zeros_like(dot_prod_flat, dtype=np.float32)
            
            # TRUCO DE ACELERACIÓN: Solo evalúa kernel si el ángulo es < 45°
            mascara = dot_prod_flat > 0.707
            if np.any(mascara):
                om_mask = np.arccos(np.clip(dot_prod_flat[mascara], 0.0, 1.0))
                vals_flat[mascara] = kernel.evaluate(om_mask, modo='pf')
                
            vals = vals_flat.reshape(batch_oris.size, p_c.shape[0], M_pf)
            suma_polos_pf = np.sum(vals, axis=1) 
            
            A[fila_inicio : fila_inicio + M_pf, start:end] = suma_polos_pf.T
            fila_inicio += M_pf
            
    # --- CALIBRACIÓN MUD ESTRICTA ---
    print(" -> Calibrando Matriz A a espacio MUD estricto (Topología SO3)...")
    w_metric = np.ones(N) / N
    
    b_iso_dummy = A @ w_metric
    b_iso_dummy[b_iso_dummy == 0] = 1e-8
    
    A /= b_iso_dummy[:, np.newaxis]
    t_matriz_fin = time.time()

    # ====================================================================
    # 5. SOLUCIONADOR ITERATIVO M-ART
    # ====================================================================
    t_nnls_inicio = time.time()
    
    tol_error = 1e-3         
    max_iter_escala = 10    
    max_iter_mart = 10       
    alpha = 1.0             
    
    print(f" -> Optimizando sistema       : M-ART con Extracción MUD de Phon...")
    
    b_ajustado = np.maximum(b_original.copy(), 1e-4)
    factores_escala = np.ones(len(lista_pfs_exp))
    w = np.copy(w_metric)
    A_sum = np.sum(A, axis=0) + 1e-8 
    
    residuo_global_anterior = np.inf
    w_iso = 0.0
    
    for iteracion in range(max_iter_escala):
        fondos_pf = []
        for inicio, fin in pf_limites:
            fondo_local = np.percentile(b_ajustado[inicio:fin], 5)
            fondos_pf.append(fondo_local)
            
        w_iso = np.clip(np.median(fondos_pf), 0.0, 0.999)
        masa_discreta = 1.0 - w_iso
        
        b_target = np.zeros_like(b_ajustado)
        for i, (inicio, fin) in enumerate(pf_limites):
            b_target[inicio:fin] = np.maximum(b_ajustado[inicio:fin] - w_iso, 1e-5)
        
        residuo_interno_ant = np.inf
        for j in range(max_iter_mart): 
            b_calc = A @ w + 1e-8
            ratio = b_target / b_calc
            w *= (A.T @ ratio) / A_sum
            
            if j % 5 == 0:
                residuo_actual = np.linalg.norm((A @ w) - b_target)
                if residuo_interno_ant != np.inf:
                    mejora = (residuo_interno_ant - residuo_actual) / residuo_interno_ant
                    if mejora < tol_error and mejora >= 0:
                        break
                residuo_interno_ant = residuo_actual
                
        suma_w = np.sum(w)
        if suma_w > 0:
            w = (w / suma_w) * masa_discreta
            
        b_teorico = A @ w + w_iso
        
        for i, (inicio, fin) in enumerate(pf_limites):
            pf_exp = b_ajustado[inicio:fin]
            pf_teo = b_teorico[inicio:fin]
            
            contraste = np.max(pf_exp) / (np.min(pf_exp) + 1e-5)
            if contraste < 1.3:
                factor_bruto = np.sum(pf_teo) / np.sum(pf_exp)
            else:
                umbral_ruido = np.max(pf_exp) * 0.15 
                mask = pf_exp > umbral_ruido
                if np.any(mask): factor_bruto = np.median(pf_teo[mask] / pf_exp[mask])
                else: factor_bruto = np.sum(pf_teo) / np.sum(pf_exp)
                
            factor_suave = 1.0 + alpha * (factor_bruto - 1.0)
            b_ajustado[inicio:fin] *= factor_suave
            factores_escala[i] *= factor_suave
                
        residuo_global_actual = np.linalg.norm(b_teorico - b_ajustado)
        print(f"    [Iteración {iteracion + 1:2d}/{max_iter_escala}] Residuo: {residuo_global_actual:.4f} | Phon: {w_iso*100:.1f}%")
        
        if residuo_global_anterior != np.inf:
            mejora_global = (residuo_global_anterior - residuo_global_actual) / residuo_global_anterior
            if iteracion > 5 and mejora_global < tol_error and mejora_global >= 0:
                print(f"    * Convergencia suave alcanzada en ciclo externo {iteracion + 1}.")
                break
                
        residuo_global_anterior = residuo_global_actual

    pesos_optimos = w
    peso_isotropico_final = w_iso
    residuo = np.linalg.norm((A @ pesos_optimos + peso_isotropico_final) - b_ajustado)
    t_nnls_fin = time.time()
    
    # ====================================================================
    # 6. FILTRO DE POLVO DINÁMICO (Acelera el graficado manteniendo textura)
    # ====================================================================
    masa_discreta_original = np.sum(pesos_optimos)
    
    umbral_polvo = np.max(pesos_optimos) * 0.005
    idx_activos = pesos_optimos > umbral_polvo
    
    # Reasignamos el peso del polvo borrado al fondo isotrópico para no perder masa
    masa_polvo_borrado = np.sum(pesos_optimos[~idx_activos])
    peso_isotropico_final += masa_polvo_borrado
    
    pesos_limpios = np.zeros_like(pesos_optimos)
    pesos_limpios[idx_activos] = pesos_optimos[idx_activos]
    pesos_optimos = pesos_limpios
    
    # Renormalización final estricta
    masa_discreta_nueva = np.sum(pesos_optimos)
    suma_total_final = masa_discreta_nueva + peso_isotropico_final
    if suma_total_final > 0:
        pesos_optimos /= suma_total_final
        peso_isotropico_final /= suma_total_final
        
    idx_finales = pesos_optimos > 0
    oris_activas = oris_base[idx_finales]
    pesos_activos = pesos_optimos[idx_finales]
    
    activos = oris_activas.size
    t_total_fin = time.time()
    
    t_matriz = t_matriz_fin - t_matriz_inicio
    t_nnls = t_nnls_fin - t_nnls_inicio
    t_total = t_total_fin - t_total_inicio
    
    # ====================================================================
    # REPORTE EN PANTALLA
    # ====================================================================
    print(f"\n=========================================================")
    print(f"📊 BALANCE DE MASA FINAL DE LA TEXTURA")
    print(f"=========================================================")
    print(f" -> Tiempo Matriz A           : {t_matriz:.2f} segundos")
    print(f" -> Tiempo Inversión M-ART    : {t_nnls:.2f} segundos")
    print(f" -> Tiempo Total Inversión    : {t_total:.2f} segundos")
    print(f" -> Residuo de ajuste (Error) : {residuo:.4f}")
    print(f"---------------------------------------------------------")
    print(f" 🟣 COMPONENTES CONTINUAS (Nube M-ART)")
    print(f"    - Nodos SO(3) activos     : {activos} de {N}")
    print(f"    - Peso total (Fracción)   : {np.sum(pesos_activos) * 100:.2f} %")
    print(f"")
    print(f" 🌫️ COMPONENTE ISOTRÓPICA (Phon)")
    print(f"    - Nivel de fondo global   : {peso_isotropico_final:.4f} MUD")
    print(f"    - Peso total (Fracción)   : {peso_isotropico_final * 100:.2f} %")
    print(f"=========================================================\n")
    
    # ====================================================================
    # 7. ACTUALIZACIÓN DE LOS DATOS EXPERIMENTALES Y ENSAMBLE
    # ====================================================================
    for i, pf in enumerate(lista_pfs_exp):
        factor_total = factores_pre_escala[i] * factores_escala[i]
        pf.intensidades *= factor_total
        
    odf_discreta = ODFComponent(
        orientaciones=oris_activas, 
        pesos=pesos_activos, 
        kernels=kernel, 
        crystal_sym=fase_cristal.point_group, 
        sample_sym=simetria_real_usuario
    )
    
    if peso_isotropico_final > 1e-4:
        odf_iso = ODFIsotropic(
            crystal_sym=fase_cristal.point_group, 
            peso=peso_isotropico_final, 
            sample_sym=simetria_real_usuario
        )
        return odf_discreta + odf_iso
    else:
        return odf_discreta