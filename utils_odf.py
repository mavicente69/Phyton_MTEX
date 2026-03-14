# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:07:43 2026

@author: mavic
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Rotation, Orientation
from orix.vector import Vector3d
from scipy.spatial import cKDTree

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
                
            rot = Orientation.from_euler(arr, symmetry=self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym)
            
        elif isinstance(orientaciones, Orientation):
            rot = orientaciones
        else:
            raise TypeError("Formato no soportado. Usá un array de Euler (N,3) o un objeto de orix.")
            
        return self.evaluate(rot)

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
            
            # Formato de ejes para que parezca MTEX
            ax.set_xticks(np.arange(0, X.max()+1, 30))
            ax.set_yticks(np.arange(0, Y.max()+1, 30))
            ax.invert_yaxis()
            ax.set_aspect('equal', adjustable='box')

        fig.colorbar(cp, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        plt.show()

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
        Calcula los coeficientes de Fourier (Triclínico-Triclínico) de la ODF completa.
        Aplica explícitamente las simetrías de cristal y muestra para garantizar 
        una representación verdaderamente simetrizada, a prueba de componentes 
        definidas fuera de la Zona Fundamental.
        """
        from utils_fourier import calc_component_fourier
        from orix.quaternion import Rotation
        
        # Inicializamos el diccionario de coeficientes globales en cero
        C_total = {}
        for l in range(L_max + 1):
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    C_total[(l, m, n)] = 0.0 + 0.0j
                    
        total_componentes = self.orientaciones.size
        print(f" -> Calculando Fourier (L_max={L_max}) con expansión de simetría (Brute Force)...")
        
        # Bucle principal: Barremos sobre todas las orientaciones base
        for i in range(total_componentes):
            ori = self.orientaciones[i]
            peso = self.pesos[i]
            kernel_comp = self.kernels[i]
            
            # 1. Generar todas las variantes simétricas (Cristal + Muestra)
            # Esto blinda el cálculo contra componentes fuera de la Zona Fundamental
            equiv_cryst = self.crystal_sym * ori
            
            if self.sample_sym:
                variantes = [ (op * equiv_cryst).data for op in self.sample_sym ]
                variantes_rot = Rotation(np.vstack(variantes))
            else: 
                variantes_rot = Rotation(equiv_cryst.data)
                
            # 2. Conservación de masa: dividimos el peso original entre las N variantes
            peso_variante = peso / variantes_rot.size
            
            # 3. Barremos sobre las variantes para rotar el kernel a cada posición
            for j in range(variantes_rot.size):
                var_ori = variantes_rot[j]
                
                # Motor triclínico para esta variante específica
                coefs_comp = calc_component_fourier(kernel_comp, var_ori, L_max)
                
                # Sumamos al acumulador global
                for l, m, n, c_val in coefs_comp:
                    C_total[(l, m, n)] += peso_variante * c_val
                
        return C_total

# =========================================================================================
# (Acá abajo sigue tu clase ODFDiscreta tal cual la tenías)
# =========================================================================================

class ODFDiscreta(ODF):
    def __init__(self, euler_grid, pesos, crystal_sym, sample_sym=None, lims=None):
        super().__init__(crystal_sym, sample_sym, lims=lims)
        
        # Asumiendo que euler_grid son ángulos en radianes (N,3)
        self.orientaciones = Orientation.from_euler(euler_grid, symmetry=self.crystal_sym)
        self.pesos = np.array(pesos)
        
        # Manejo de la simetría y KD-Tree (Tal como lo tenías implementado)
        pg = self.crystal_sym.point_group if hasattr(self.crystal_sym, 'point_group') else self.crystal_sym
        
        # ... (Tu implementación de Friedel y KD-Tree para ODFDiscreta sigue acá adentro)
        # Para no sobreescribir tu lógica de KDTree, dejá tu clase ODFDiscreta intacta acá abajo.