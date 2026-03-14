# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:05:13 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:05:13 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Motor de Reconstrucción ODF - WIMV por Binning Discreto (Secuencial M-ART)
Entorno: texturaPy3.10
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from orix.quaternion import Orientation
from orix.vector import Vector3d

def reconstruir_odf_wimv(lista_pfs_exp, fase_cristal, simetria_muestra=None, resolucion_grados=None, iteraciones=15):
    print(f"\n--- Ejecutando WIMV Discreto: Región Fundamental ({fase_cristal.name}) ---")
    
    # 1. CÁLCULO DINÁMICO DE RESOLUCIÓN
    if resolucion_grados is None:
        puntos_promedio = np.mean([pf.direcciones.size for pf in lista_pfs_exp])
        res_rad = np.sqrt(2 * np.pi / puntos_promedio)
        res_ideal_deg = np.degrees(res_rad)
        resoluciones_estandar = np.array([2.5, 3.0, 5.0, 7.5, 10.0, 15.0])
        idx_cercano = np.argmin(np.abs(resoluciones_estandar - res_ideal_deg))
        resolucion_grados = resoluciones_estandar[idx_cercano]
        print(f" -> Resolución dinámica auto-ajustada: {resolucion_grados}°")
    else:
        print(f" -> Resolución fijada por usuario: {resolucion_grados}°")

    # =========================================================
    # 2. RECORTAMOS EL ESPACIO DE EULER SEGÚN LA SIMETRÍA
    # =========================================================
    n_samp = simetria_muestra.size if hasattr(simetria_muestra, 'size') else 1
    phi1_max = 90 if n_samp > 2 else 360  
    phi2_max = 60 # Límite para Hexagonal
    
    if n_samp > 2:
        print(f" -> Simetría de muestra detectada. Recortando phi1 a {phi1_max}° (Cálculo optimizado)")

    p1 = np.arange(0, phi1_max + resolucion_grados, resolucion_grados)
    P  = np.arange(0, 90 + resolucion_grados, resolucion_grados)
    p2 = np.arange(0, phi2_max + resolucion_grados, resolucion_grados)
    
    p1, P, p2 = p1[p1 <= phi1_max], P[P <= 90], p2[p2 <= phi2_max]
    
    P1, PP, P2 = np.meshgrid(p1, P, p2, indexing='ij')
    euler_grid = np.column_stack((P1.ravel(), PP.ravel(), P2.ravel()))
    
    orientaciones = Orientation.from_euler(
        np.radians(euler_grid), 
        symmetry=fase_cristal.point_group
    ).unique()
    
    n_ori = orientaciones.size
    pesos = np.ones(n_ori)

    # 3. Pre-cálculo: Mapeo de Orientaciones a la Grilla de las PF
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

    indices_por_pf = []
    exp_vals_por_pf = []

    for pf in lista_pfs_exp:
        interp = NearestNDInterpolator(pf.direcciones.data, pf.intensidades)
        
        # ======================================================================
        # CORRECCIÓN VITAL: Simetrizar los datos experimentales (KOWARI rotado)
        # ======================================================================
        if simetria_muestra is not None:
            val_exp_sym = np.zeros(v_centers.shape[0])
            for op in simetria_muestra:
                # Rotamos la grilla por la simetría y forzamos al hemisferio superior
                v_rot = (~op * Vector3d(v_centers)).data
                v_rot = np.where(v_rot[:, 2:3] < 0, -v_rot, v_rot) 
                val_exp_sym += np.nan_to_num(interp(v_rot), nan=0.0)
            val_exp = val_exp_sym / simetria_muestra.size
        else:
            val_exp = np.nan_to_num(interp(v_centers), nan=0.0)
            
        # Normalizamos a media 1
        if np.mean(val_exp) > 0: val_exp /= np.mean(val_exp)
        val_exp[val_exp < 0.01] = 0.01
        exp_vals_por_pf.append(val_exp)

        sym_hkl = fase_cristal.point_group * pf.hkl
        normales_cristal = sym_hkl.data.astype(float)
        normas = np.linalg.norm(normales_cristal, axis=1, keepdims=True)
        normales_cristal = np.divide(normales_cristal, normas, out=np.zeros_like(normales_cristal), where=normas!=0)
        
        normales_upper = np.where(normales_cristal[:, 2:3] < 0, -normales_cristal, normales_cristal)
        normales_unicas = np.unique(normales_upper.round(4), axis=0)
        vecs_cristal = Vector3d(normales_unicas)
        
        polos_muestra = (~orientaciones)[:, np.newaxis] * vecs_cristal 
        
        if simetria_muestra is not None:
            polos_lista = [(op * polos_muestra).data for op in simetria_muestra]
            coords = np.concatenate(polos_lista, axis=1).reshape(-1, 3).astype(float)
            n_sym_actual = vecs_cristal.size * simetria_muestra.size
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
    # 4. BUCLE WIMV REAL (Multiplicative ART Secuencial)
    # =========================================================
    for it in range(iteraciones):
        cambios_totales = []

        # En vez de promediar todo, actualizamos PF por PF para romper estancamientos
        for i in range(len(lista_pfs_exp)):
            lin_idx_2d = indices_por_pf[i]
            val_exp = exp_vals_por_pf[i]
            n_sym_actual = lin_idx_2d.shape[1]

            pesos_exp = np.repeat(pesos, n_sym_actual)
            p_calc_sum = np.bincount(lin_idx_2d.ravel(), weights=pesos_exp, minlength=n_bins)
            p_calc_count = np.bincount(lin_idx_2d.ravel(), minlength=n_bins)

            valid = p_calc_count > 0
            p_calc = np.zeros(n_bins)
            p_calc[valid] = p_calc_sum[valid] / p_calc_count[valid]
            
            # NORMALIZACIÓN CORRECTA: Igualamos escalas solo en los bines válidos
            if np.sum(p_calc[valid]) > 0:
                factor_escala = np.sum(val_exp[valid]) / np.sum(p_calc[valid])
                p_calc[valid] *= factor_escala

            ratio_bins = np.ones(n_bins)
            # Clip: Limitamos el multiplicador para no matar orientaciones prematuramente
            ratio_bins[valid] = np.clip(val_exp[valid] / (p_calc[valid] + 1e-4), 0.1, 10.0)
            
            ratios_polos = ratio_bins[lin_idx_2d]
            
            # Media geométrica de corrección para los polos de ESTA figura de polos
            factor_pf = np.exp(np.mean(np.log(ratios_polos + 1e-8), axis=1))
            
            # Relajación para suavizar el salto y asegurar convergencia global
            relajacion = 0.5
            factor_suavizado = 1.0 + relajacion * (factor_pf - 1.0)
            
            pesos *= factor_suavizado
            pesos /= np.mean(pesos)
            
            cambios_totales.append(np.mean(np.abs(1.0 - factor_suavizado)))

        mejora = np.mean(cambios_totales)
        print(f" Iteración {it+1:02d} | Cambio medio de pesos: {mejora:.4f}")

    print("--- Reconstrucción exitosa ---")
    return orientaciones, pesos