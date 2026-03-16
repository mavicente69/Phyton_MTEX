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
        
        # LÍMITES DINÁMICOS:
        if lims is not None:
            self.lims = lims
        else:
            if sample_sym is None:
                n_samp = 1
            elif hasattr(sample_sym, 'size'):
                n_samp = sample_sym.size
            else:
                try:
                    n_samp = len(sample_sym)
                except TypeError:
                    n_samp = 1
                    
            phi1_max = 90 if n_samp > 2 else 360
            self.lims = {'phi1': phi1_max, 'Phi': 90, 'phi2': 60}

    def __add__(self, other):
        """ Sobrecarga del operador '+' para sumar texturas """
        if not isinstance(other, ODF):
            raise TypeError(f"No se puede sumar ODF con {type(other)}")
        return ODFMixed([self, other])

    def evaluate(self, target_orientations):
        """Método abstracto para evaluar la densidad en un array de cuaterniones."""
        raise NotImplementedError("Debe implementarse en la subclase.")

    def get_density(self, orientaciones, degrees=True):
        """
        Calcula la intensidad de la ODF en puntos específicos.
        """
        if isinstance(orientaciones, (list, np.ndarray)):
            arr = np.array(orientaciones)
            if arr.ndim == 1 and len(arr) == 3:
                arr = arr.reshape(1, 3)
            
            if degrees:
                arr = np.radians(arr)
                
            pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
            rot = Orientation.from_euler(arr, symmetry=pg)
            
        elif isinstance(orientaciones, Orientation):
            rot = orientaciones
        else:
            raise TypeError("Formato no soportado. Usá un array de Euler (N,3) o un objeto de orix.")
            
        return self.evaluate(rot)

    def get_value_at(self, phi1, Phi, phi2, degrees=True):
        """
        Devuelve el valor de intensidad (MUD) en una orientación específica.
        """
        return self.get_density([[phi1, Phi, phi2]], degrees=degrees)[0]

    def calc_texture_index(self, res_grados=5.0):
        """
        Calcula el Índice de Textura (J-Index) integrando numéricamente 
        la ODF al cuadrado sobre todo el espacio de Euler válido.
        """
        print(f" -> Calculando J-Index (Resolución: {res_grados}°)...")
        phi1_max = self.lims['phi1']
        Phi_max = self.lims['Phi']
        phi2_max = self.lims['phi2']
        
        x = np.arange(0, phi1_max + res_grados, res_grados)
        y = np.arange(0, Phi_max + res_grados, res_grados)
        z = np.arange(0, phi2_max + res_grados, res_grados)
        
        PHI1, PHI, PHI2 = np.meshgrid(x, y, z, indexing='ij')
        eulers = np.stack([PHI1.ravel(), PHI.ravel(), PHI2.ravel()], axis=-1)
        
        f_g = self.get_density(eulers, degrees=True)
        sin_Phi = np.sin(np.radians(PHI.ravel()))
        
        vol_total = np.sum(sin_Phi)
        if vol_total == 0:
            return 1.0 
            
        j_index = np.sum((f_g**2) * sin_Phi) / vol_total
        return j_index

    def calc_component_volume(self, center_euler, radius_degrees=15.0, res_grados=5.0):
        """
        Calcula la fracción de volumen de una componente (integral de la ODF) 
        dentro de un radio de desorientación dado respecto a un centro.
        """
        print(f" -> Calculando volumen de componente cerca de {center_euler} (Radio: {radius_degrees}°)...")
        phi1_max = self.lims['phi1']
        Phi_max = self.lims['Phi']
        phi2_max = self.lims['phi2']
        
        # 1. Armamos la grilla en el espacio de Euler
        x = np.arange(0, phi1_max + res_grados, res_grados)
        y = np.arange(0, Phi_max + res_grados, res_grados)
        z = np.arange(0, phi2_max + res_grados, res_grados)
        
        PHI1, PHI, PHI2 = np.meshgrid(x, y, z, indexing='ij')
        eulers = np.stack([PHI1.ravel(), PHI.ravel(), PHI2.ravel()], axis=-1)
        
        # 2. Convertimos a objetos de orix para medir distancias con simetría
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        grid_oris = Orientation.from_euler(np.radians(eulers), symmetry=pg)
        
        # La orientación central que queremos evaluar
        center_ori = Orientation.from_euler(np.radians([center_euler]), symmetry=pg)
        
        # 3. Calculamos la desorientación y filtramos los que caen dentro del radio
        # angle_with() nos da el ángulo mínimo considerando la simetría del cristal
        distancias = grid_oris.angle_with(center_ori)
        mask = distancias <= np.radians(radius_degrees)
        
        if not np.any(mask):
            return 0.0
            
        # 4. Evaluamos la ODF solo en los puntos que cayeron dentro de la burbuja
        eulers_dentro = eulers[mask]
        f_g = self.get_density(eulers_dentro, degrees=True)
        
        # 5. Integración numérica (Invariante de volumen sin(Phi))
        sin_Phi_dentro = np.sin(np.radians(eulers_dentro[:, 1])) # Columna 1 es Phi
        vol_dentro = np.sum(f_g * sin_Phi_dentro)
        
        # Normalizamos dividiendo por el volumen total de la caja fundamental
        vol_total_zona = np.sum(np.sin(np.radians(PHI.ravel())))
        
        return vol_dentro / vol_total_zona
    
    def calc_pole_figures(self, lista_hkl, resolution=10):
        """
        Calcula y devuelve una lista de objetos PoleFigure.
        """
        import utils_orient
        return utils_orient.calc_pole_figures(self, lista_hkl, resolution=resolution)

    def plot_sections(self, sections=None, axis='phi2', res_grados=2.5, cmap='jet'):
        """
        Grafica secciones 2D del espacio de Euler.
        """
        if sections is None:
            sections = [0, 15, 30, 45, 60]

        n_secs = len(sections)
        fig, axes = plt.subplots(1, n_secs, figsize=(4 * n_secs, 4))
        if n_secs == 1: axes = [axes]
        
        phi1_max = self.lims['phi1']
        Phi_max = self.lims['Phi']
        phi2_max = self.lims['phi2']

        all_vals = []
        titulos = []
        grids = []

        for i, val in enumerate(sections):
            if axis == 'phi2':
                if val > phi2_max: continue
                x = np.arange(0, phi1_max + res_grados, res_grados)
                y = np.arange(0, Phi_max + res_grados, res_grados)
                X, Y = np.meshgrid(x, y)
                eulers = np.stack([X.ravel(), Y.ravel(), np.full(X.size, val)], axis=-1)
                titulos.append(rf"$\varphi_2 = {val}^\circ$")
                xlabel, ylabel = rf"$\varphi_1$", rf"$\Phi$"
            elif axis == 'phi1':
                if val > phi1_max: continue
                x = np.arange(0, phi2_max + res_grados, res_grados)
                y = np.arange(0, Phi_max + res_grados, res_grados)
                X, Y = np.meshgrid(x, y)
                eulers = np.stack([np.full(X.size, val), Y.ravel(), X.ravel()], axis=-1)
                titulos.append(rf"$\varphi_1 = {val}^\circ$")
                xlabel, ylabel = rf"$\varphi_2$", rf"$\Phi$"
            elif axis == 'Phi':
                if val > Phi_max: continue
                x = np.arange(0, phi1_max + res_grados, res_grados)
                y = np.arange(0, phi2_max + res_grados, res_grados)
                X, Y = np.meshgrid(x, y)
                eulers = np.stack([X.ravel(), np.full(X.size, val), Y.ravel()], axis=-1)
                titulos.append(rf"$\Phi = {val}^\circ$")
                xlabel, ylabel = rf"$\varphi_1$", rf"$\varphi_2$"
                
            grids.append((X, Y))
            all_vals.append(self.get_density(eulers, degrees=True))

        global_max = max([np.max(v) for v in all_vals]) if all_vals else 1.0

        for i in range(len(all_vals)):
            ax = axes[i]
            X, Y = grids[i]
            Z = all_vals[i].reshape(X.shape)
            
            cp = ax.contourf(X, Y, Z, levels=np.linspace(0, global_max, 15), cmap=cmap)
            ax.set_title(titulos[i])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            ax.set_xticks(np.arange(0, X.max()+1, 30))
            ax.set_yticks(np.arange(0, Y.max()+1, 30))
            ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')

        fig.colorbar(cp, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        plt.show()

# =========================================================================================
# CLASE ODF COMPONENTE (Maneja Kernels y Fourier)
# =========================================================================================
class ODFComponent(ODF):
    def __init__(self, orientaciones, pesos, kernels, crystal_sym, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.orientaciones = orientaciones
        self.pesos = np.array(pesos)
        
        # Flexibilidad: si le pasás un solo kernel, lo multiplica para todas las componentes
        if isinstance(kernels, list):
            self.kernels = kernels
        else:
            self.kernels = [kernels] * len(pesos)

    def evaluate(self, target_orientations):
        rot_targets = Rotation(target_orientations.data)
        total_density = np.zeros(rot_targets.size)
        
        for i in range(self.orientaciones.size):
            equiv_cryst = self.crystal_sym * self.orientaciones[i]
            
            if self.sample_sym:
                variantes = [ (op * equiv_cryst).data for op in self.sample_sym ]
                equiv_rot = Rotation(np.vstack(variantes))
            else: 
                equiv_rot = Rotation(equiv_cryst.data)
                
            dist_matrix = equiv_rot.angle_with_outer(rot_targets)
            total_density += self.pesos[i] * self.kernels[i].evaluate(np.min(dist_matrix, axis=0), modo='odf')
            
        return total_density

    def calc_fourier_coeffs(self, L_max):
        """
        Calcula los coeficientes de Fourier (Triclínico-Triclínico) por fuerza bruta,
        expandiendo simetrías para evitar desbalances en orientaciones arbitrarias.
        """
        from utils_fourier import calc_component_fourier
        from orix.quaternion import Rotation
        
        C_total = {}
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    C_total[(l, m, n)] = 0.0 + 0.0j
                    
        total_componentes = self.orientaciones.size
        print(f" -> Calculando Fourier (L_max={L_max}) con expansión de simetría (Brute Force)...")
        
        for i in range(total_componentes):
            ori = self.orientaciones[i]
            peso = self.pesos[i]
            kernel_comp = self.kernels[i]
            
            equiv_cryst = self.crystal_sym * ori
            if self.sample_sym:
                variantes = [ (op * equiv_cryst).data for op in self.sample_sym ]
                variantes_rot = Rotation(np.vstack(variantes))
            else: 
                variantes_rot = Rotation(equiv_cryst.data)
                
            peso_variante = peso / variantes_rot.size
            
            for j in range(variantes_rot.size):
                var_ori = variantes_rot[j]
                coefs_comp = calc_component_fourier(kernel_comp, var_ori, L_max)
                
                # coefs_comp es una lista de tuplas (l, m, n, val)
                for l, m, n, c_val in coefs_comp:
                    C_total[(l, m, n)] += peso_variante * c_val
                
        return C_total

# =========================================================================================
# CLASES DE TEXTURAS IDEALES Y MIXTAS
# =========================================================================================
class ODFIsotropic(ODF):
    def __init__(self, crystal_sym, peso=1.0, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.peso = peso
        
    def evaluate(self, target_orientations):
        rot_targets = Rotation(target_orientations.data)
        return np.ones(rot_targets.size) * self.peso

    def calc_fourier_coeffs(self, L_max):
        """ Fourier para textura Isotrópica: Solo existe L=0. """
        C_total = {}
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    # El armónico l=0, m=0, n=0 lleva todo el peso
                    if l == 0 and m == 0 and n == 0:
                        C_total[(l, m, n)] = self.peso + 0.0j
                    else:
                        C_total[(l, m, n)] = 0.0 + 0.0j
        return C_total


class ODFFiber(ODF):
    def __init__(self, hkl, uvw, kernel, crystal_sym, peso=1.0, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        self.hkl = hkl
        self.uvw = uvw
        self.kernel = kernel
        self.peso = peso
        
    def evaluate(self, target_orientations):
        """ Evalúa la densidad 3D de la fibra en todo el espacio de Euler. """
        rot_targets = Rotation(target_orientations.data)
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym

        # 1. Polos hkl simétricos en el marco del cristal
        hkl_sym = pg * self.hkl
        hkl_data = hkl_sym.data.astype(float)
        hkl_data /= np.linalg.norm(hkl_data, axis=1, keepdims=True)

        # 2. Dirección de la fibra en la muestra
        uvw_data = self.uvw.data.reshape(3).astype(float)
        uvw_data /= np.linalg.norm(uvw_data)

        # 3. Rotamos los polos desde el cristal a la muestra 
        # (orix: ~rot rota vector cristal -> vector muestra)
        polos_muestra = (~rot_targets)[:, np.newaxis] * hkl_sym
        coords = polos_muestra.data.astype(float)

        normas = np.linalg.norm(coords, axis=-1, keepdims=True)
        normas[normas==0] = 1.0
        coords /= normas

        # 4. Buscamos el ángulo mínimo respecto al eje de la fibra
        dot_prods = np.abs(np.sum(coords * uvw_data, axis=-1))
        angulos = np.arccos(np.clip(dot_prods, 0.0, 1.0))
        min_angulos = np.min(angulos, axis=1)

        # 5. Evaluamos el kernel en modo 'odf'
        return self.kernel.evaluate(min_angulos, modo='odf') * self.peso

    def calc_fourier_coeffs(self, L_max):
        """
        Calcula los coeficientes de Fourier triclínicos para una Fibra
        utilizando la solución analítica con Armónicos Esféricos.
        """
        # --- PARCHE DE COMPATIBILIDAD PARA SCIPY (VERSIÓN NUEVA VS CLÁSICA) ---
        try:
            from scipy.special import sph_harm_y
            def sph_harm(m, l, azim, polar):
                # En las nuevas versiones de SciPy (>1.15.0), se llama sph_harm_y 
                # y cambia el orden de los argumentos a (l, m, polar, azimutal)
                return sph_harm_y(l, m, polar, azim)
        except ImportError:
            from scipy.special import sph_harm
            # En la versión clásica la firma es (m, l, azimutal, polar)
        # ----------------------------------------------------------------------

        from utils_fourier import calc_component_fourier
        from orix.quaternion import Rotation
        from orix.vector import Vector3d
        
        # 1. Obtenemos los q_l (coeficientes 1D del kernel) con el Truco de Identidad
        coefs_origen = calc_component_fourier(self.kernel, Rotation.identity(), L_max)
        
        # --- CORRECCIÓN: Leer la lista de tuplas (l, m, n, val) ---
        q_l = {l: 0.0 for l in range(L_max + 1)}
        for l, m, n, val in coefs_origen:
            if m == 0 and n == 0:
                q_l[l] = val.real
        # ----------------------------------------------------------
        
        C_total = { (l, m, n): 0.0 + 0.0j for l in range(L_max + 1) 
                                          for m in range(-l, l + 1) 
                                          for n in range(-l, l + 1) }
                    
        print(f" -> Calculando Fourier (L_max={L_max}) para Fibra (Solución Analítica)...")
        
        # 2. Expandimos las variantes de simetría (Muestra y Cristal)
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        hkl_sym = pg * self.hkl
        hkl_data = hkl_sym.data.astype(float)
        hkl_data /= np.linalg.norm(hkl_data, axis=1, keepdims=True)
        
        uvw_data = self.uvw.data.reshape(3).astype(float)
        uvw_data /= np.linalg.norm(uvw_data)
        
        if self.sample_sym:
            uvw_variantes = np.vstack([(op * Vector3d(uvw_data)).data for op in self.sample_sym])
        else:
            uvw_variantes = uvw_data.reshape(-1, 3)
            
        uvw_variantes /= np.linalg.norm(uvw_variantes, axis=1, keepdims=True)
        peso_variante = self.peso / (hkl_data.shape[0] * uvw_variantes.shape[0])
        
        # 3. Integración Analítica
        for h_vec in hkl_data:
            # Ángulos esféricos para h (Cristal): polar (0 a pi), azimutal (-pi a pi)
            polar_h = np.arccos(np.clip(h_vec[2], -1.0, 1.0))
            azim_h = np.arctan2(h_vec[1], h_vec[0])
            
            for y_vec in uvw_variantes:
                # Ángulos esféricos para y (Muestra)
                polar_y = np.arccos(np.clip(y_vec[2], -1.0, 1.0))
                azim_y = np.arctan2(y_vec[1], y_vec[0])
                
                for l in range(L_max + 1):
                    factor = (4.0 * np.pi) / (2 * l + 1)
                    q = q_l[l]
                    
                    for m in range(-l, l + 1):
                        Y_h = sph_harm(m, l, azim_h, polar_h)
                        for n in range(-l, l + 1):
                            Y_y = sph_harm(n, l, azim_y, polar_y)
                            
                            # Ecuación de convolución de la fibra
                            val = peso_variante * factor * q * Y_h * np.conj(Y_y)
                            C_total[(l, m, n)] += val
                            
        return C_total


class ODFMixed(ODF):
    def __init__(self, componentes_odf):
        super().__init__(componentes_odf[0].crystal_sym, componentes_odf[0].sample_sym, lims=componentes_odf[0].lims)
        self.componentes = []
        for comp in componentes_odf:
            if isinstance(comp, ODFMixed):
                self.componentes.extend(comp.componentes)
            else:
                self.componentes.append(comp)

    def evaluate(self, target_orientations):
        rot_targets = Rotation(target_orientations.data)
        total_density = np.zeros(rot_targets.size)
        for comp in self.componentes:
            total_density += comp.evaluate(target_orientations)
        return total_density

    def __add__(self, other):
        if isinstance(other, ODFMixed):
            return ODFMixed(self.componentes + other.componentes)
        elif isinstance(other, ODF):
            return ODFMixed(self.componentes + [other])
        else:
            raise TypeError(f"No se puede sumar ODFMixed con {type(other)}")

    def calc_fourier_coeffs(self, L_max):
        """
        Calcula los coeficientes triclínicos de la textura completa
        sumando linealmente las contribuciones de todas sus partes.
        """
        # Inicializamos en 0
        C_sum = { (l, m, n): 0.0 + 0.0j for l in range(L_max + 1) 
                                        for m in range(-l, l + 1) 
                                        for n in range(-l, l + 1) }
        
        print("\n=== INICIANDO EXPANSIÓN DE FOURIER GLOBAL ===")
        for comp in self.componentes:
            # Polimorfismo: Cada clase (Component, Fiber, Isotropic) sabe cómo calcularse
            c_parcial = comp.calc_fourier_coeffs(L_max)
            
            # Sumamos al acumulador global
            for key in C_sum.keys():
                C_sum[key] += c_parcial[key]
                
        print("=== EXPANSIÓN COMPLETADA ===\n")
        return C_sum

# =========================================================================================
# CLASE ODF DISCRETA (KD-Tree Rápido)
# =========================================================================================
class ODFDiscreta(ODF):
    def __init__(self, euler_grid, pesos, crystal_sym, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        self.orientaciones = Orientation.from_euler(euler_grid, symmetry=pg)
        pesos_base = np.array(pesos)
        
        # 1. Simetría de Cristal
        oris_sym = pg * self.orientaciones
        
        # 2. Simetría de Muestra
        if self.sample_sym:
            variantes = [ (op * oris_sym).data for op in self.sample_sym ]
            q_data_matrix = np.vstack(variantes)
        else:
            q_data_matrix = oris_sym.data
            
        q_data = q_data_matrix.reshape(-1, 4)
        
        # 3. Simetría de Friedel en el espacio 4D (q y -q)
        q_data_full = np.vstack([q_data, -q_data])
        
        # 4. Expansión de pesos para mapear la grilla extendida
        multiplicador = pg.size * (self.sample_sym.size if self.sample_sym is not None else 1)
        pesos_exp = np.repeat(pesos_base, multiplicador)
        self.pesos_full = np.concatenate([pesos_exp, pesos_exp])
        
        # 5. Árbol de búsqueda espacial
        self.tree = cKDTree(q_data_full)
        
    def evaluate(self, target_orientations):
        """ Evalúa la densidad ODF buscando el cuaternión discreto más cercano. """
        q_targets = target_orientations.data
        dist, idx = self.tree.query(q_targets)
        return self.pesos_full[idx]