# -*- coding: utf-8 -*-
"""
Módulo de Funciones de Distribución de Orientaciones (ODF).
Soporta componentes, texturas de fibra, isotrópicas, discretas y texturas mixtas.
Entorno: texturaPy3.10
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Rotation, Orientation
from orix.vector import Vector3d
from scipy.spatial import cKDTree

# =========================================================================================
# CLASE BASE
# =========================================================================================
class ODF:
    def __init__(self, crystal_sym, sample_sym=None, lims=None):
        self.crystal_sym = crystal_sym
        self.sample_sym = sample_sym
        
        if lims is not None:
            self.lims = lims
        else:
            if sample_sym is None:
                n_samp = 1
            elif hasattr(sample_sym, 'size'):
                n_samp = sample_sym.size
            else:
                try: n_samp = len(sample_sym)
                except TypeError: n_samp = 1
            phi1_max = 90 if n_samp > 2 else 360
            self.lims = {'phi1': phi1_max, 'Phi': 90, 'phi2': 60}

    def __add__(self, other):
        if not isinstance(other, ODF):
            raise TypeError(f"No se puede sumar ODF con {type(other)}")
        return ODFMixed([self, other])

    def evaluate(self, target_orientations):
        raise NotImplementedError("Debe implementarse en la subclase.")

    def get_density(self, orientaciones, degrees=True):
        if isinstance(orientaciones, (list, np.ndarray)):
            arr = np.array(orientaciones)
            if arr.ndim == 1 and len(arr) == 3: arr = arr.reshape(1, 3)
            if degrees: arr = np.radians(arr)
            pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
            rot = Orientation.from_euler(arr, symmetry=pg)
        elif isinstance(orientaciones, Orientation):
            rot = orientaciones
        else:
            raise TypeError("Formato no soportado. Usá un array de Euler (N,3) o un objeto de orix.")
        return self.evaluate(rot)

    def get_value_at(self, phi1, Phi, phi2, degrees=True):
        return self.get_density([[phi1, Phi, phi2]], degrees=degrees)[0]

    def calc_texture_index(self, res_grados=5.0):
        print(f" -> Calculando J-Index (Resolución: {res_grados}°)...")
        phi1_max, Phi_max, phi2_max = self.lims['phi1'], self.lims['Phi'], self.lims['phi2']
        x = np.arange(0, phi1_max + res_grados, res_grados)
        y = np.arange(0, Phi_max + res_grados, res_grados)
        z = np.arange(0, phi2_max + res_grados, res_grados)
        PHI1, PHI, PHI2 = np.meshgrid(x, y, z, indexing='ij')
        eulers = np.stack([PHI1.ravel(), PHI.ravel(), PHI2.ravel()], axis=-1)
        f_g = self.get_density(eulers, degrees=True)
        sin_Phi = np.sin(np.radians(PHI.ravel()))
        vol_total = np.sum(sin_Phi)
        if vol_total == 0: return 1.0 
        return np.sum((f_g**2) * sin_Phi) / vol_total

    def calc_component_volume(self, center_euler, radius_degrees=15.0, res_grados=5.0):
        print(f" -> Calculando volumen de componente cerca de {center_euler} (Radio: {radius_degrees}°)...")
        phi1_max, Phi_max, phi2_max = self.lims['phi1'], self.lims['Phi'], self.lims['phi2']
        x = np.arange(0, phi1_max + res_grados, res_grados)
        y = np.arange(0, Phi_max + res_grados, res_grados)
        z = np.arange(0, phi2_max + res_grados, res_grados)
        PHI1, PHI, PHI2 = np.meshgrid(x, y, z, indexing='ij')
        eulers = np.stack([PHI1.ravel(), PHI.ravel(), PHI2.ravel()], axis=-1)
        
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        grid_oris = Orientation.from_euler(np.radians(eulers), symmetry=pg)
        center_ori = Orientation.from_euler(np.radians([center_euler]), symmetry=pg)
        
        distancias = grid_oris.angle_with(center_ori)
        mask = distancias <= np.radians(radius_degrees)
        if not np.any(mask): return 0.0
            
        eulers_dentro = eulers[mask]
        f_g = self.get_density(eulers_dentro, degrees=True)
        sin_Phi_dentro = np.sin(np.radians(eulers_dentro[:, 1])) 
        vol_dentro = np.sum(f_g * sin_Phi_dentro)
        vol_total_zona = np.sum(np.sin(np.radians(PHI.ravel())))
        return vol_dentro / vol_total_zona
    
    def calc_pole_figures(self, lista_hkl, res_grados=5.0):
        import utils_orient
        puntos_radiales = max(1, int(np.round(90.0 / res_grados)))
        return utils_orient.calc_pole_figures(self, lista_hkl, resolution=puntos_radiales)

    def calc_bunge_coeffs(self, L_max):
        from utils_fourier import calc_symmetry_projectors, calc_symmetry_coefficients, get_bunge_coefs
        print(f" -> Extrayendo Base Irreducible de Bunge (Mu, Nu) para L_max={L_max}...")
        coefs_sym_mn = self.calc_fourier_coeffs(L_max, return_triclinic=False)
        proj_C, proj_S = calc_symmetry_projectors(L_max, self.crystal_sym, self.sample_sym)
        A_cryst, B_samp = calc_symmetry_coefficients(L_max, proj_C, proj_S)
        return get_bunge_coefs(coefs_sym_mn, L_max, A_cryst, B_samp)

    def plot_sections(self, sections=None, axis='phi2', res_grados=2.5, cmap='jet'):
        if sections is None: sections = [0, 15, 30, 45, 60]
        n_secs = len(sections)
        fig, axes = plt.subplots(1, n_secs, figsize=(4 * n_secs, 4))
        if n_secs == 1: axes = [axes]
        phi1_max, Phi_max, phi2_max = self.lims['phi1'], self.lims['Phi'], self.lims['phi2']
        all_vals, titulos, grids = [], [], []

        for i, val in enumerate(sections):
            if axis == 'phi2':
                if val > phi2_max: continue
                x, y = np.arange(0, phi1_max + res_grados, res_grados), np.arange(0, Phi_max + res_grados, res_grados)
                X, Y = np.meshgrid(x, y)
                eulers = np.stack([X.ravel(), Y.ravel(), np.full(X.size, val)], axis=-1)
                titulos.append(rf"$\varphi_2 = {val}^\circ$"); xlabel, ylabel = rf"$\varphi_1$", rf"$\Phi$"
            elif axis == 'phi1':
                if val > phi1_max: continue
                x, y = np.arange(0, phi2_max + res_grados, res_grados), np.arange(0, Phi_max + res_grados, res_grados)
                X, Y = np.meshgrid(x, y)
                eulers = np.stack([np.full(X.size, val), Y.ravel(), X.ravel()], axis=-1)
                titulos.append(rf"$\varphi_1 = {val}^\circ$"); xlabel, ylabel = rf"$\varphi_2$", rf"$\Phi$"
            elif axis == 'Phi':
                if val > Phi_max: continue
                x, y = np.arange(0, phi1_max + res_grados, res_grados), np.arange(0, phi2_max + res_grados, res_grados)
                X, Y = np.meshgrid(x, y)
                eulers = np.stack([X.ravel(), np.full(X.size, val), Y.ravel()], axis=-1)
                titulos.append(rf"$\Phi = {val}^\circ$"); xlabel, ylabel = rf"$\varphi_1$", rf"$\varphi_2$"
            grids.append((X, Y))
            all_vals.append(self.get_density(eulers, degrees=True))

        global_max = max([np.max(v) for v in all_vals]) if all_vals else 1.0

        for i in range(len(all_vals)):
            ax = axes[i]
            X, Y = grids[i]
            Z = all_vals[i].reshape(X.shape)
            cp = ax.contourf(X, Y, Z, levels=np.linspace(0, global_max, 15), cmap=cmap)
            ax.set_title(titulos[i]); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(0, X.max()+1, 30)); ax.set_yticks(np.arange(0, Y.max()+1, 30))
            ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box')

        fig.colorbar(cp, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='MUD')
        plt.show(block=False)

# =========================================================================================
# CLASE ODF COMPONENTE (Maneja Kernels y Fourier Optimizado)
# =========================================================================================
class ODFComponent(ODF):
    def __init__(self, orientaciones, pesos, kernels, crystal_sym, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.orientaciones = orientaciones
        self.pesos = np.array(pesos)
        self.kernels = kernels if isinstance(kernels, list) else [kernels] * len(pesos)

    def evaluate(self, target_orientations):
        rot_targets = Rotation(target_orientations.data)
        total_density = np.zeros(rot_targets.size)
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        n_sym_total = pg.size * (self.sample_sym.size if self.sample_sym is not None else 1)
        
        for i in range(self.orientaciones.size):
            equiv_cryst = pg * self.orientaciones[i]
            if self.sample_sym: equiv_rot = Rotation(np.vstack([(op * equiv_cryst).data for op in self.sample_sym]))
            else: equiv_rot = Rotation(equiv_cryst.data)
                
            dist_matrix = equiv_rot.angle_with_outer(rot_targets)
            kernel_vals = self.kernels[i].evaluate(dist_matrix, modo='odf')
            total_density += self.pesos[i] * (np.sum(kernel_vals, axis=0) / n_sym_total)
        return total_density

    def calc_fourier_coeffs(self, L_max, return_triclinic=False):
        from utils_fourier import calc_component_fourier, calc_symmetry_projectors, symmetrize_coefs
        C_tri_dict = {}
        # ✅ NO MÁS EXPANSIÓN A PEDAL: Evaluamos solo el centro puro.
        for i in range(self.orientaciones.size):
            coefs = calc_component_fourier(self.kernels[i], self.orientaciones[i], L_max)
            for row in coefs:
                l, m, n = int(row[0]), int(row[1]), int(row[2])
                C_tri_dict[(l, m, n)] = C_tri_dict.get((l, m, n), 0.0j) + self.pesos[i] * (row[3] + 1j * row[4])
                
        data = [[l, m, n, v.real, v.imag, abs(v)] for (l, m, n), v in C_tri_dict.items() if abs(v) > 1e-10]
        C_tri_array = np.array(data) if data else np.zeros((0, 6))
        
        proj_C, proj_S = calc_symmetry_projectors(L_max, self.crystal_sym, self.sample_sym)
        C_sym_array = symmetrize_coefs(C_tri_array, L_max, proj_C, proj_S)
        
        if return_triclinic: return C_sym_array, C_tri_array
        return C_sym_array

# =========================================================================================
# CLASES DE TEXTURAS IDEALES Y MIXTAS
# =========================================================================================
class ODFIsotropic(ODF):
    def __init__(self, crystal_sym, peso=1.0, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.peso = peso
        
    def evaluate(self, target_orientations):
        return np.ones(target_orientations.size) * self.peso

    def calc_fourier_coeffs(self, L_max, return_triclinic=False):
        arr = np.array([[0, 0, 0, self.peso, 0.0, self.peso]])
        if return_triclinic: return arr, arr.copy()
        return arr

class ODFFiber(ODF):
    def __init__(self, hkl, uvw, kernel, crystal_sym, peso=1.0, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.hkl = hkl
        self.uvw = uvw
        self.kernel = kernel
        self.peso = peso
        
    def evaluate(self, target_orientations):
        rot_targets = Rotation(target_orientations.data)
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        hkl_sym = pg * self.hkl
        hkl_data = hkl_sym.data.astype(float)
        hkl_data /= np.linalg.norm(hkl_data, axis=1, keepdims=True)
        uvw_data = self.uvw.data.reshape(3).astype(float)
        uvw_data /= np.linalg.norm(uvw_data)

        polos_muestra = (~rot_targets)[:, np.newaxis] * hkl_sym
        coords = polos_muestra.data.astype(float)
        normas = np.linalg.norm(coords, axis=-1, keepdims=True); normas[normas==0] = 1.0; coords /= normas
        dot_prods = np.abs(np.sum(coords * uvw_data, axis=-1))
        min_angulos = np.min(np.arccos(np.clip(dot_prods, 0.0, 1.0)), axis=1)
        return self.kernel.evaluate(min_angulos, modo='odf') * self.peso

    def calc_fourier_coeffs(self, L_max, return_triclinic=False):
        try:
            from scipy.special import sph_harm_y
            def sph_harm(m, l, azim, polar): return sph_harm_y(l, m, polar, azim)
        except ImportError:
            from scipy.special import sph_harm

        from utils_fourier import calc_component_fourier
        from orix.vector import Vector3d
        coefs_origen = calc_component_fourier(self.kernel, Rotation.identity(), L_max)
        
        q_l = {l: 0.0 for l in range(L_max + 1)}
        for fila in coefs_origen:
            if int(fila[1]) == 0 and int(fila[2]) == 0: q_l[int(fila[0])] = fila[3]
        
        C_total = { (l, m, n): 0.0 + 0.0j for l in range(L_max + 1) for m in range(-l, l + 1) for n in range(-l, l + 1) }
        
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        hkl_sym = pg * self.hkl; hkl_data = hkl_sym.data.astype(float); hkl_data /= np.linalg.norm(hkl_data, axis=1, keepdims=True)
        uvw_data = self.uvw.data.reshape(3).astype(float); uvw_data /= np.linalg.norm(uvw_data)
        
        if self.sample_sym: uvw_variantes = np.vstack([(op * Vector3d(uvw_data)).data for op in self.sample_sym])
        else: uvw_variantes = uvw_data.reshape(-1, 3)
            
        uvw_variantes /= np.linalg.norm(uvw_variantes, axis=1, keepdims=True)
        peso_variante = self.peso / (hkl_data.shape[0] * uvw_variantes.shape[0])
        
        for h_vec in hkl_data:
            polar_h, azim_h = np.arccos(np.clip(h_vec[2], -1.0, 1.0)), np.arctan2(h_vec[1], h_vec[0])
            for y_vec in uvw_variantes:
                polar_y, azim_y = np.arccos(np.clip(y_vec[2], -1.0, 1.0)), np.arctan2(y_vec[1], y_vec[0])
                for l in range(L_max + 1):
                    factor = (4.0 * np.pi) / (2 * l + 1)
                    q = q_l[l]
                    for m in range(-l, l + 1):
                        Y_h = sph_harm(m, l, azim_h, polar_h)
                        for n in range(-l, l + 1):
                            Y_y = sph_harm(n, l, azim_y, polar_y)
                            val = peso_variante * factor * q * Y_h * np.conj(Y_y) * (2 * l + 1)
                            C_total[(l, m, n)] += val
                            
        data = [[l, m, n, v.real, v.imag, abs(v)] for (l, m, n), v in C_total.items() if abs(v) > 1e-10]
        C_sym = np.array(data) if data else np.zeros((0, 6))
        if len(C_sym) > 0: C_sym = C_sym[np.lexsort((C_sym[:, 2], C_sym[:, 1], C_sym[:, 0]))]
            
        if return_triclinic: return C_sym, C_sym.copy()
        return C_sym

class ODFMixed(ODF):
    def __init__(self, componentes_odf):
        super().__init__(componentes_odf[0].crystal_sym, componentes_odf[0].sample_sym, lims=componentes_odf[0].lims)
        self.componentes = []
        for comp in componentes_odf:
            if isinstance(comp, ODFMixed): self.componentes.extend(comp.componentes)
            else: self.componentes.append(comp)

    def evaluate(self, target_orientations):
        total_density = np.zeros(target_orientations.size)
        for comp in self.componentes: total_density += comp.evaluate(target_orientations)
        return total_density

    def __add__(self, other):
        if isinstance(other, ODFMixed): return ODFMixed(self.componentes + other.componentes)
        elif isinstance(other, ODF): return ODFMixed(self.componentes + [other])
        else: raise TypeError(f"No se puede sumar ODFMixed con {type(other)}")

    def calc_fourier_coeffs(self, L_max, return_triclinic=False):
        from utils_fourier import sum_fourier_arrays
        if return_triclinic:
            pares = [c.calc_fourier_coeffs(L_max, return_triclinic=True) for c in self.componentes]
            return sum_fourier_arrays([p[0] for p in pares]), sum_fourier_arrays([p[1] for p in pares])
        else:
            return sum_fourier_arrays([c.calc_fourier_coeffs(L_max) for c in self.componentes])

# =========================================================================================
# CLASE ODF DISCRETA (KD-Tree Rápido con Interpolación Suave)
# =========================================================================================
class ODFDiscreta(ODF):
    def __init__(self, pesos, crystal_sym, euler_grid=None, orientaciones=None, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        
        if orientaciones is not None: self.orientaciones = orientaciones
        elif euler_grid is not None: self.orientaciones = Orientation.from_euler(euler_grid, symmetry=pg)
        else: raise ValueError("Debes proveer 'euler_grid' o 'orientaciones'.")
            
        self.pesos = np.array(pesos)
        oris_sym = pg[:, np.newaxis] * self.orientaciones
        
        if self.sample_sym: q_data_matrix = np.vstack([ (op * oris_sym).data for op in self.sample_sym ])
        else: q_data_matrix = oris_sym.data
            
        q_data = q_data_matrix.reshape(-1, 4); q_data_full = np.vstack([q_data, -q_data])
        multiplicador = pg.size * (self.sample_sym.size if self.sample_sym is not None else 1)
        pesos_exp = np.tile(self.pesos, multiplicador)
        self.pesos_full = np.concatenate([pesos_exp, pesos_exp])
        self.tree = cKDTree(q_data_full)
        
    def evaluate(self, target_orientations):
        dist, idx = self.tree.query(target_orientations.data, k=5)
        pesos_vecinos = np.exp(-0.5 * (np.maximum(dist, 1e-6) / 0.10)**2)
        pesos_vecinos /= np.sum(pesos_vecinos, axis=1, keepdims=True)
        return np.sum(self.pesos_full[idx] * pesos_vecinos, axis=1)
    
    def calc_fourier_coeffs(self, L_max, return_triclinic=False):
        from utils_fourier import calc_discrete_fourier, calc_symmetry_projectors, symmetrize_coefs
        import time
        
        # ✅ NO MÁS EXPANSIÓN A PEDAL: Evaluamos directo sobre los nodos nativos (ej. 13.000)
        print(f"\n -> Integrando Fourier (L_max={L_max}) sobre {self.orientaciones.size} nodos nativos WIMV (Turbo Numba)...")
        t0 = time.time()
        C_tri_array = calc_discrete_fourier(self.orientaciones, self.pesos, L_max)
        print(f" -> Integración completada en {time.time() - t0:.3f} segundos.")
        
        # La multiplicación por los proyectores resuelve TODO el caleidoscopio simétrico analíticamente.
        proj_C, proj_S = calc_symmetry_projectors(L_max, self.crystal_sym, self.sample_sym)
        C_sym_array = symmetrize_coefs(C_tri_array, L_max, proj_C, proj_S)
        
        if return_triclinic:
            return C_sym_array, C_tri_array
        return C_sym_array

# =========================================================================================
# CLASE ODF FOURIER Y ODF BUNGE (Síntesis Analítica)
# =========================================================================================
class ODFFourier(ODF):
    def __init__(self, coefs_array, crystal_sym, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.coefs_array = coefs_array

    def evaluate(self, target_orientations):
        from utils_fourier import eval_odf_from_fourier
        return np.maximum(eval_odf_from_fourier(self.coefs_array, target_orientations), 0.0)

    def calc_fourier_coeffs(self, L_max=None, return_triclinic=False):
        if return_triclinic: return self.coefs_array, self.coefs_array
        return self.coefs_array
        
    def calc_pole_figures(self, lista_hkl, res_grados=5.0):
        from utils_pf import PoleFigure
        return [PoleFigure.from_fourier(self.coefs_array, hkl, self.crystal_sym, res_grados=res_grados) for hkl in lista_hkl]
    
class ODFBunge(ODF):
    def __init__(self, C_bunge, crystal_sym, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.C_bunge = C_bunge
        self.L_max = max([k[0] for k in C_bunge.keys()]) if C_bunge else 0
        from utils_fourier import calc_symmetry_projectors, calc_symmetry_coefficients
        proj_C, proj_S = calc_symmetry_projectors(self.L_max, self.crystal_sym, self.sample_sym)
        self.A_cryst, self.B_samp = calc_symmetry_coefficients(self.L_max, proj_C, proj_S)

    def evaluate(self, target_orientations):
        from utils_fourier import eval_odf_from_bunge
        return eval_odf_from_bunge(self.C_bunge, target_orientations, self.A_cryst, self.B_samp)

    def calc_fourier_coeffs(self, L_max=None, return_triclinic=False):
        import numpy as np
        C_sym_list = []
        for l in range(self.L_max + 1):
            if l not in self.A_cryst or l not in self.B_samp: continue
            M_l, N_l = self.A_cryst[l].shape[1], self.B_samp[l].shape[1]
            if M_l == 0 or N_l == 0: continue
            
            C_l_red = np.zeros((M_l, N_l), dtype=complex)
            for mu in range(M_l):
                for nu in range(N_l): C_l_red[mu, nu] = self.C_bunge.get((l, mu, nu), 0.0j)
                    
            C_l_full = self.A_cryst[l] @ C_l_red @ np.conj(self.B_samp[l]).T
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    val = C_l_full[m + l, n + l]
                    if abs(val) > 1e-10: C_sym_list.append([l, m, n, val.real, val.imag, abs(val)])
                        
        C_sym = np.array(C_sym_list) if C_sym_list else np.zeros((0, 6))
        if len(C_sym) > 0: C_sym = C_sym[np.lexsort((C_sym[:, 2], C_sym[:, 1], C_sym[:, 0]))]
        if return_triclinic: return C_sym, C_sym
        return C_sym

    def calc_pole_figures(self, lista_hkl, res_grados=5.0):
        from utils_pf import PoleFigure
        return [PoleFigure.from_bunge(self.C_bunge, self.A_cryst, self.B_samp, hkl, res_grados=res_grados) for hkl in lista_hkl]