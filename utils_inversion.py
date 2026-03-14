# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:12:36 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Motor de Inversión de Textura Continuo (Estilo MTEX - NNLS)
Entorno: texturaPy3.10
"""
import numpy as np
import time
from scipy.optimize import nnls
from orix.quaternion import Orientation

# Importamos tu clase nativa de componentes y tu kernel
from utils_odf import ODFComponent
from utils_kernels import OrientationKernel

def reconstruir_odf_nnls(lista_pfs_exp, fase_cristal, simetria_muestra=None, resolucion_grados=10):
    print(f"\n=========================================================")
    print(f"🧠 MOTOR DE INVERSIÓN CONTINUO (Mínimos Cuadrados NNLS)")
    print(f"=========================================================")
    t0 = time.time()
    
    # ====================================================================
    # 1. GENERAR ESPACIO BASE DE LA ODF (Región Fundamental)
    # ====================================================================
    # Límite de phi1 dinámico según el orden de la simetría de la muestra
    lim_phi1 = 360
    if simetria_muestra is not None:
        num_ops = simetria_muestra.size  # <-- CORREGIDO: Usamos .size para objetos de orix
        if num_ops >= 4:      # Ortorrómbica (ej. D2h, 222)
            lim_phi1 = 90
        elif num_ops == 2:    # Monoclínica (ej. C2, 2/m)
            lim_phi1 = 180
        else:                 # Triclínica (ej. C1) o sin simetría
            lim_phi1 = 360
            
    lim_PHI = 90
    
    # Límites de phi2 según la simetría del cristal
    if '6/mmm' in fase_cristal.point_group.name or '622' in fase_cristal.point_group.name:
        lim_phi2 = 60
    elif 'm-3m' in fase_cristal.point_group.name or '432' in fase_cristal.point_group.name:
        lim_phi2 = 90
    else:
        lim_phi2 = 360
        
    p1 = np.arange(0, lim_phi1 + resolucion_grados, resolucion_grados)
    P  = np.arange(0, lim_PHI + resolucion_grados, resolucion_grados)
    p2 = np.arange(0, lim_phi2 + resolucion_grados, resolucion_grados)
    
    P1, PP, P2 = np.meshgrid(p1, P, p2, indexing='ij')
    euler_grid = np.column_stack((P1.ravel(), PP.ravel(), P2.ravel()))
    
    oris_crudas = Orientation.from_euler(np.radians(euler_grid), symmetry=fase_cristal.point_group)
    oris_base = oris_crudas.unique()
    N = oris_base.size
    print(f" -> Espacio de Euler generado : {N} orientaciones base (phi1 max: {lim_phi1}°)")
    
    # ====================================================================
    # 2. PREPARAR VECTOR EXPERIMENTAL (b)
    # ====================================================================
    b = np.concatenate([pf.intensidades for pf in lista_pfs_exp])
    M_total = b.size
    print(f" -> Píxeles experimentales    : {M_total} (Vector b)")
    
    # ====================================================================
    # 3. CONSTRUIR MATRIZ DE PROYECCIÓN (A)
    # ====================================================================
    print(f" -> Construyendo Matriz A     : [{M_total} x {N}] ...")
    A = np.zeros((M_total, N), dtype=np.float32)
    
    # Relación FWHM a Sigma para la gaussiana (FWHM = 2.3548 * sigma)
    fwhm_usado = 15.0
    sigma_rad = np.radians(fwhm_usado / 2.3548)
    pg = fase_cristal.point_group
    
    batch_size = 500
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_oris = oris_base[start:end]
        
        fila_inicio = 0
        for pf in lista_pfs_exp:
            M_pf = pf.direcciones.size
            v_g = pf.direcciones.data 
            
            polos_c = (~batch_oris)[:, np.newaxis] * (pg * pf.hkl)
            
            if simetria_muestra is not None:
                polos_lista = [(op * polos_c).data for op in simetria_muestra]
                polos_data = np.concatenate(polos_lista, axis=1).astype(np.float32)
            else:
                polos_data = polos_c.data.astype(np.float32)
                
            normas = np.linalg.norm(polos_data, axis=2, keepdims=True)
            normas[normas == 0] = 1.0
            polos_data /= normas
            
            dot_prod = np.abs(np.einsum('bpc,mc->bpm', polos_data, v_g))
            np.clip(dot_prod, 0.0, 1.0, out=dot_prod)
            
            om = np.arccos(dot_prod)
            om /= sigma_rad
            np.square(om, out=om)
            om *= -0.5
            
            vals = np.exp(om)
            suma_polos = np.sum(vals, axis=1) 
            
            A[fila_inicio : fila_inicio + M_pf, start:end] = suma_polos.T
            fila_inicio += M_pf

    # ====================================================================
    # 4. RESOLUCIÓN DEL SISTEMA MÍNIMOS CUADRADOS NO NEGATIVOS (NNLS)
    # ====================================================================
    print(" -> Optimizando sistema NNLS  : Calculando pesos exactos...")
    pesos_optimos, residuo = nnls(A, b)
    
    # ====================================================================
    # 5. POST-PROCESAMIENTO Y EMPAQUETADO EN ODFComponent
    # ====================================================================
    # Filtramos el "ruido" matemático para no saturar ODFComponent con cientos de 
    # componentes numéricamente irrelevantes. Cortamos lo menor al 1% del pico.
    umbral = np.max(pesos_optimos) * 0.01
    pesos_optimos[pesos_optimos < umbral] = 0.0
    
    # NORMALIZACIÓN CORRECTA: La suma de las fracciones de volumen debe dar 1.0
    suma_pesos = np.sum(pesos_optimos)
    if suma_pesos > 0:
        pesos_optimos /= suma_pesos
        
    # Extraemos únicamente las orientaciones que sobrevivieron al filtro
    idx_activos = pesos_optimos > 0
    oris_activas = oris_base[idx_activos]
    pesos_activos = pesos_optimos[idx_activos]
    
    activos = oris_activas.size
    t1 = time.time()
    
    print(f" -> Inversión completada en   : {t1 - t0:.2f} segundos")
    print(f" -> Residuo de ajuste         : {residuo:.4f}")
    print(f" -> Componentes activas       : {activos} de {N}")
    print(f"=========================================================\n")
    
    # Instanciamos el kernel que va a acompañar a todas estas orientaciones
    kernel_salida = OrientationKernel(fwhm_grados=fwhm_usado, tipo='gaussian')
    
    # Retornamos tu clase pura y rigurosa (pasando el point_group!)
    odf_resultante = ODFComponent(
        orientaciones=oris_activas, 
        pesos=pesos_activos, 
        kernels=kernel_salida, 
        crystal_sym=fase_cristal.point_group, 
        sample_sym=simetria_muestra
    )
    
    return odf_resultante