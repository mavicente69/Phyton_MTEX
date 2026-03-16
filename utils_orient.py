# -*- coding: utf-8 -*-
"""
Módulo de cálculo de Figuras de Polos (Grilla Hexagonal + LUT + KD-Tree Multihilo)
Entorno: texturaPy3.10
"""
import numpy as np
import concurrent.futures
from scipy.spatial import cKDTree
from orix.vector import Vector3d
from utils_pf import PoleFigure

def calc_pole_figures(odf_obj, lista_hkl, resolution=30):
    # ====================================================================
    # 1. GRILLA TIPO PANEL DE ABEJAS (Hexagonal)
    # ====================================================================
    paso = 1.0 / resolution  
    puntos_2d = []
    n_max = int(np.ceil(2.0 / paso))
    
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            x = i * paso + j * (paso / 2.0)
            y = j * (paso * np.sqrt(3) / 2.0)
            if x**2 + y**2 <= 1.0 + 1e-6:
                puntos_2d.append([x, y])
                
    puntos_2d = np.array(puntos_2d)
    
    R = np.clip(np.linalg.norm(puntos_2d, axis=1), 0.0, 1.0)
    theta = 2.0 * np.arcsin(R / np.sqrt(2.0))
    phi = np.arctan2(puntos_2d[:, 1], puntos_2d[:, 0])
    
    v_g_data = np.vstack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ]).T
    
    v_g = Vector3d(v_g_data)
    
    print(f"\n=========================================================")
    print(f"⚙️ MOTOR DE RENDERIZADO DE FIGURAS DE POLOS")
    print(f"=========================================================")
    print(f" -> Nodos en la grilla 2D (Panel de Abejas): {v_g.size}")
    print(f"---------------------------------------------------------\n")
    
    lista_pfs = []

    for i, hkl in enumerate(lista_hkl):
        dens = np.zeros(v_g.size)
        
        # Corrección: Compatibilidad con ODFMixed (español e inglés)
        if hasattr(odf_obj, 'componentes'):
            comps = odf_obj.componentes
        elif hasattr(odf_obj, 'components'):
            comps = odf_obj.components
        else:
            comps = [odf_obj]

        for c in comps:
            pg = c.crystal_sym.point_group if hasattr(c.crystal_sym, 'point_group') else c.crystal_sym
            
            # ====================================================================
            # 1. COMPONENTES IDEALES
            # ====================================================================
            if hasattr(c, 'orientaciones') and hasattr(c, 'kernels'): 
                for idx in range(c.orientaciones.size):
                    polos = (~c.orientaciones[idx:idx+1]) * (pg * hkl)
                    v_p = np.vstack([(op*polos).data for op in c.sample_sym]) if getattr(c, 'sample_sym', None) else polos.data.reshape(-1, 3)
                    v_p = v_p.astype(float)
                    v_p /= np.linalg.norm(v_p, axis=1, keepdims=True)
                    v_p = np.where(v_p[:, 2:3] < 0, -v_p, v_p)
                    
                    om = np.arccos(np.clip(np.abs(np.dot(v_g_data, v_p.T)), 0.0, 1.0))
                    vals = c.kernels[idx].evaluate(om, modo='pf')
                    if vals.ndim > 1: vals = np.sum(vals, axis=1)
                    dens += (c.pesos[idx] / v_p.shape[0]) * vals.flatten()
            
            # ====================================================================
            # 2. TEXTURAS DE FIBRA
            # ====================================================================
            elif hasattr(c, 'hkl') and hasattr(c, 'uvw'): 
                p_fam_data = (pg * hkl).data.astype(float)
                p_fam_u = p_fam_data / np.linalg.norm(p_fam_data, axis=1, keepdims=True)
                
                # Leemos los nombres correctos: c.hkl y c.uvw
                c_dir_data = (pg * c.hkl).data.astype(float) 
                c_dir_u = c_dir_data / np.linalg.norm(c_dir_data, axis=1, keepdims=True)
                
                m_dir_u = c.uvw.data.reshape(-1, 3).astype(float)
                m_dir_u /= np.linalg.norm(m_dir_u, axis=1, keepdims=True)
                
                d_lat = np.arccos(np.clip(np.abs(np.dot(p_fam_u, c_dir_u.T)), 0.0, 1.0)).flatten()
                d_grid = np.arccos(np.clip(np.abs(np.dot(v_g_data, m_dir_u.T)), 0.0, 1.0)).flatten()
                
                for dl in d_lat:
                    vals = c.kernel.evaluate(np.abs(d_grid - dl), modo='pf').flatten()
                    perimetro_fibra = 2 * np.pi * np.sin(dl)
                    factor_forma = 1.0 / perimetro_fibra if perimetro_fibra > 0.1 else 1.0
                    dens += (c.peso / len(d_lat)) * factor_forma * vals

            # ====================================================================
            # 3. TEXTURA ISOTRÓPICA (AZAR)
            # ====================================================================
            elif hasattr(c, 'peso') and not hasattr(c, 'hkl') and not hasattr(c, 'orientaciones'): 
                dens += c.peso

            # ====================================================================
            # 4. ODF DISCRETA MASIVA (Núcleo Triclínico + Multihilo + LUT)
            # ====================================================================
            elif hasattr(c, 'orientaciones') and not hasattr(c, 'kernels'):
                sigma_rad = np.radians(7.5) 
                
                print(f"--- Calculando Figura de Polos: {hkl.hkl} ---")
                print(f" -> Orientaciones ODF entrantes   : {c.orientaciones.size}")
                
                normales_cristal = (pg * hkl).data.astype(float)
                normales_upper = np.where(normales_cristal[:, 2:3] < 0, -normales_cristal, normales_cristal)
                vecs_cristal = Vector3d(np.unique(normales_upper.round(5), axis=0))
                
                # Proyección Base Triclínica
                polos_base = (~c.orientaciones)[:, np.newaxis] * vecs_cristal
                coords = polos_base.data.reshape(-1, 3).astype(float)
                
                normas_c = np.linalg.norm(coords, axis=1, keepdims=True)
                normas_c[normas_c == 0] = 1.0
                coords /= normas_c
                
                n_sym_total = coords.shape[0] // c.orientaciones.size
                pesos_expandidos = np.repeat(c.pesos, n_sym_total)
                
                # Consolidación On-The-Fly
                coords_upper = np.where(coords[:, 2:3] < 0, -coords, coords)
                coords_unicos, indices_inversos = np.unique(coords_upper.round(4), axis=0, return_inverse=True)
                pesos_unicos = np.bincount(indices_inversos, weights=pesos_expandidos)
                
                print(f" -> Polos TRICLÍNICOS colapsados  : {coords_unicos.shape[0]} (Evaluados en el KD-Tree)")
                
                coords_full = np.vstack([coords_unicos, -coords_unicos])
                pesos_full = np.concatenate([pesos_unicos, pesos_unicos])
                
                tree = cKDTree(coords_full)
                radio_corte = np.sqrt(2.0 - 2.0 * np.cos(3.5 * sigma_rad))
                dens_triclinica = np.zeros(v_g.size)
                
                # --- PRE-CÁLCULO LUT (Look-Up Table) ---
                LUT_SIZE = 100000
                lut_x = np.linspace(0.0, 1.0, LUT_SIZE)
                lut_om = np.arccos(lut_x)
                lut_vals = np.exp(-0.5 * (lut_om / sigma_rad)**2)
                
                def procesar_chunk(start, end):
                    v_chunk = v_g_data[start:end]
                    vecinos_list = tree.query_ball_point(v_chunk, r=radio_corte)
                    lens = np.array([len(v) for v in vecinos_list])
                    
                    if lens.sum() == 0:
                        return start, end, np.zeros(end - start)
                        
                    idx_grids_local = np.repeat(np.arange(start, end), lens).astype(int)
                    idx_polos = np.concatenate(vecinos_list).astype(int)
                    
                    p_valid = coords_full[idx_polos]
                    v_valid = v_g_data[idx_grids_local]
                    w_valid = pesos_full[idx_polos]
                    
                    # Producto punto exacto
                    dot_prod = np.abs(np.sum(p_valid * v_valid, axis=1))
                    np.clip(dot_prod, 0.0, 1.0, out=dot_prod)
                    
                    # Lectura ultrarrápida de la LUT precalculada
                    idx_lut = (dot_prod * (LUT_SIZE - 1)).astype(int)
                    vals = lut_vals[idx_lut] * w_valid
                    
                    dens_local = np.bincount(idx_grids_local - start, weights=vals, minlength=end - start)
                    return start, end, dens_local

                # --- PROCESAMIENTO MULTIHILO ---
                chunk_size = 500 
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futuros = []
                    for start in range(0, v_g.size, chunk_size):
                        end = min(start + chunk_size, v_g.size)
                        futuros.append(executor.submit(procesar_chunk, start, end))
                        
                    for f in concurrent.futures.as_completed(futuros):
                        s, e, res_local = f.result()
                        dens_triclinica[s:e] += res_local

                # === POST-SIMETRIZACIÓN DE MUESTRA ===
                if getattr(c, 'sample_sym', None) is not None:
                    print(f" -> Post-Simetrización 2D         : Aplicando {c.sample_sym.size} operadores de muestra\n")
                    grid_tree = cKDTree(v_g_data) 
                    dens_discreta = np.zeros(v_g.size)
                    
                    for op in c.sample_sym:
                        v_lookup = (~op * v_g).data
                        v_lookup = np.where(v_lookup[:, 2:3] < 0, -v_lookup, v_lookup)
                        
                        _, idx_map = grid_tree.query(v_lookup)
                        dens_discreta += dens_triclinica[idx_map.astype(int)]
                else:
                    print(f" -> Post-Simetrización 2D         : No aplica (Muestra Triclínica)\n")
                    dens_discreta = dens_triclinica

                # Normalización
                if np.mean(dens_discreta) > 0:
                    dens += dens_discreta / np.mean(dens_discreta)

        dens = np.nan_to_num(dens, nan=0.0)
        
        pf_obj = PoleFigure(direcciones=v_g, intensidades=dens, hkl=hkl)
        if not hasattr(pf_obj, 'puntos_2d'):
            pf_obj.puntos_2d = puntos_2d 
            
        lista_pfs.append(pf_obj)

    return lista_pfs