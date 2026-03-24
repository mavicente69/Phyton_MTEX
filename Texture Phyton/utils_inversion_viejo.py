# -*- coding: utf-8 -*-
"""
Motor de Inversión de Textura Continuo (Estilo MTEX - NNLS)
Soporte Multi-Material (HCP, BCC, FCC) y Simetría de Muestra
Optimizado con Álgebra Tensorial, Relajación Triclínica y Auto-Escalado Iterativo
Entorno: texturaPy3.10
"""
import numpy as np
import time
from scipy.optimize import nnls
from orix.quaternion import Orientation

from utils_odf import ODFComponent
from utils_kernels import OrientationKernel

def reconstruir_odf_nnls(lista_pfs_exp, fase_cristal, simetria_muestra=None, tipo_kernel='gaussian', resolucion_grados=None, iteraciones_escala=3):
    """
    Reconstruye una ODF implementando relajación triclínica y 
    auto-escalado iterativo para ajustar las intensidades MUD experimentales.
    """
    print(f"\n=========================================================")
    print(f"🧠 MOTOR DE INVERSIÓN CONTINUO (Mínimos Cuadrados NNLS)")
    print(f"=========================================================")
    t_total_inicio = time.time()
    
    # 🚀 RELAJACIÓN TRICLÍNICA AUTOMÁTICA
    simetria_real_usuario = simetria_muestra
    simetria_calculo = None 
    
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
    # 1. GENERAR GRILLA DE EULER UNIFORME
    # ====================================================================
    print(" -> Generando grilla de Euler uniforme...")
    sim_name = getattr(fase_cristal.point_group, 'name', str(fase_cristal.point_group))
    
    if simetria_calculo is not None and getattr(simetria_calculo, 'name', '') in ['222', 'mmm', 'D2h', 'm-m-m']:
        lim_phi1 = 90
    else: lim_phi1 = 360
        
    p1 = np.arange(0, lim_phi1 + resolucion_grados, resolucion_grados)
    P  = np.arange(0, 90 + resolucion_grados, resolucion_grados)
    
    if '6' in sim_name or fase_cristal.point_group.size in [12, 24]: lim_phi2 = 60
    elif 'm-3m' in sim_name or fase_cristal.point_group.size in [24, 48]: lim_phi2 = 90
    else: lim_phi2 = 360 
        
    p2 = np.arange(0, lim_phi2 + resolucion_grados, resolucion_grados)
    
    P1, PP, P2 = np.meshgrid(p1, P, p2, indexing='ij')
    euler_grid = np.column_stack((P1.ravel(), PP.ravel(), P2.ravel()))
    
    oris_crudas = Orientation.from_euler(np.radians(euler_grid), symmetry=fase_cristal.point_group)
    oris_base = oris_crudas.unique()
    N = oris_base.size
    print(f" -> Espacio de Euler generado : {N} orientaciones base")
    
    # ====================================================================
    # 2. PREPARAR VECTOR EXPERIMENTAL (b) E ÍNDICES PARA ESCALADO
    # ====================================================================
    b_original = np.concatenate([pf.intensidades for pf in lista_pfs_exp])
    M_total = b_original.size
    print(f" -> Píxeles experimentales    : {M_total} (Vector b)")
    
    # Guardamos dónde empieza y termina cada PF en el vector global
    pf_limites = []
    idx_actual = 0
    for pf in lista_pfs_exp:
        size = pf.direcciones.size
        pf_limites.append((idx_actual, idx_actual + size))
        idx_actual += size
    
    # ====================================================================
    # 3. CONSTRUIR MATRIZ DE PROYECCIÓN (A)
    # ====================================================================
    print(f" -> Construyendo Matriz A     : [{M_total} x {N}] ...")
    t_matriz_inicio = time.time()
    
    A = np.zeros((M_total, N), dtype=np.float32)
    pg = fase_cristal.point_group
    pfs_crystal_poles = [(pg * pf.hkl).data.astype(np.float32) for pf in lista_pfs_exp]
    
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
        for idx, pf in enumerate(lista_pfs_exp):
            M_pf = pf.direcciones.size
            v_g = pf.direcciones.data 
            
            p_c = pfs_crystal_poles[idx]
            polos_data_base = np.einsum('bij,pj->bpi', R_c2s, p_c)
            
            suma_polos_pf = np.zeros((batch_oris.size, M_pf), dtype=np.float32)
            
            normas = np.linalg.norm(polos_data_base, axis=2, keepdims=True)
            normas[normas == 0] = 1.0
            polos_data_base /= normas
            
            dot_prod = np.abs(np.einsum('bpc,mc->bpm', polos_data_base, v_g))
            np.clip(dot_prod, 0.0, 1.0, out=dot_prod)
            om = np.arccos(dot_prod)
            
            vals = kernel.evaluate(om, modo='pf')
            suma_polos_pf += np.sum(vals, axis=1)
                
            A[fila_inicio : fila_inicio + M_pf, start:end] = suma_polos_pf.T
            fila_inicio += M_pf
            
    t_matriz_fin = time.time()

    # ====================================================================
    # 4. RESOLUCIÓN NNLS CON AUTO-ESCALADO ITERATIVO
    # ====================================================================
    print(f" -> Optimizando sistema NNLS  : Auto-escalando intensidades MUD ({iteraciones_escala} iteraciones)...")
    t_nnls_inicio = time.time()
    
    b_ajustado = b_original.copy()
    factores_escala = np.ones(len(lista_pfs_exp))
    
    for iteracion in range(iteraciones_escala):
        # 1. Resolvemos el sistema
        pesos_optimos, residuo = nnls(A, b_ajustado)
        
        # 2. Forzamos volumen 1.0 en la ODF para tener un MUD teórico estricto
        suma_w = np.sum(pesos_optimos)
        if suma_w > 0:
            pesos_optimos /= suma_w
            
        # 3. Calculamos cómo se ven las PFs teóricas perfectas
        b_teorico = A @ pesos_optimos
        
        # 4. Comparamos y ajustamos cada PF experimental individualmente
        if iteracion < iteraciones_escala - 1: # No necesitamos re-escalar en la última vuelta
            for i, (inicio, fin) in enumerate(pf_limites):
                pf_exp = b_ajustado[inicio:fin]
                pf_teo = b_teorico[inicio:fin]
                
                suma_exp = np.sum(pf_exp)
                if suma_exp > 0:
                    factor = np.sum(pf_teo) / suma_exp
                    b_ajustado[inicio:fin] *= factor
                    factores_escala[i] *= factor

    t_nnls_fin = time.time()
    
    # ====================================================================
    # 5. POST-PROCESAMIENTO Y EMPAQUETADO EN ODFComponent
    # ====================================================================
    umbral = np.max(pesos_optimos) * 0.01
    pesos_optimos[pesos_optimos < umbral] = 0.0
    
    suma_pesos = np.sum(pesos_optimos)
    if suma_pesos > 0:
        pesos_optimos /= suma_pesos
        
    idx_activos = pesos_optimos > 0
    oris_activas = oris_base[idx_activos]
    pesos_activos = pesos_optimos[idx_activos]
    
    activos = oris_activas.size
    t_total_fin = time.time()
    
    t_matriz = t_matriz_fin - t_matriz_inicio
    t_nnls = t_nnls_fin - t_nnls_inicio
    t_total = t_total_fin - t_total_inicio
    
    print(f"\n=========================================================")
    print(f"⏱️  REPORTE DE RENDIMIENTO Y ESCALADO")
    print(f"---------------------------------------------------------")
    print(f" -> Tiempo Matriz A           : {t_matriz:.2f} segundos")
    print(f" -> Tiempo NNLS Iterativo     : {t_nnls:.2f} segundos")
    print(f" -> Tiempo Total Inversión    : {t_total:.2f} segundos")
    print(f"---------------------------------------------------------")
    print(f" -> Residuo de ajuste (Error) : {residuo:.4f}")
    print(f" -> Factores de Escala PFs    : {np.round(factores_escala, 3)}")
    print(f" -> Componentes activas       : {activos} de {N}")
    print(f"=========================================================\n")
    
    odf_resultante = ODFComponent(
        orientaciones=oris_activas, 
        pesos=pesos_activos, 
        kernels=kernel, 
        crystal_sym=fase_cristal.point_group, 
        sample_sym=simetria_real_usuario
    )
    
    return odf_resultante