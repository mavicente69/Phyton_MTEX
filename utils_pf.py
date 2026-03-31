# -*- coding: utf-8 -*-
"""
Módulo de Figuras de Polos (PF).
Define la clase PoleFigure como un objeto independiente que encapsula 
la grilla estereográfica, las intensidades y los métodos de graficado.
Soporta construcción analítica directa desde coeficientes de Fourier.
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.vector import Vector3d
from orix.quaternion import Rotation, Orientation
from utils_vector3d import proyectar_igual_area

# --- COMPATIBILIDAD SCIPY ---
try:
    from scipy.special import sph_harm_y
    def sph_harm(m, l, azim, polar):
        # SciPy nuevo invirtió el orden de los argumentos
        return sph_harm_y(l, m, polar, azim)
except ImportError:
    from scipy.special import sph_harm
# ----------------------------

class PoleFigure:
    """
    Clase para almacenar, rotar y visualizar Figuras de Polos (PFs).
    """
    def __init__(self, direcciones, intensidades, hkl):
        """
        direcciones: Objeto Vector3d de orix.
        intensidades: Array 1D o 2D con los valores de densidad (MUD o conteos).
        hkl: Objeto Miller de orix que define el plano cristalográfico.
        """
        if not isinstance(direcciones, Vector3d):
            raise TypeError("Error: 'direcciones' debe ser un objeto Vector3d de la librería orix.")
            
        self.direcciones = Vector3d(direcciones.data.reshape(-1, 3))
        self.intensidades = np.asarray(intensidades).flatten()
        self.hkl = hkl
        
        if self.direcciones.size != self.intensidades.size:
            raise ValueError(f"Desajuste de tamaño: {self.direcciones.size} direcciones vs {self.intensidades.size} intensidades.")

    @classmethod
    def from_fourier(cls, coefs, hkl_miller, crystal_sym, resolution=2.5):
        """
        Genera una Figura de Polos evaluando analíticamente los coeficientes 
        de Fourier (C_lmn) sobre una grilla esférica (Hemisferio Norte).
        Aplica la Ley de Friedel (solo armónicos pares).
        """
        import time
        t0 = time.time()
        
        # Obtenemos la etiqueta visual del plano (ej: "0001" o "10-10")
        hkl_vals = hkl_miller.hkil.flatten() if hasattr(hkl_miller, 'hkil') else hkl_miller.hkl.flatten()
        label_plano = ''.join([str(int(x)) for x in hkl_vals])
        print(f" -> Construyendo Figura de Polos {{{label_plano}}} desde Fourier...")
        
        # 1. Crear Grilla Esférica de la Muestra (Solo Hemisferio Norte, Z >= 0)
        azi = np.radians(np.arange(0, 360, resolution))
        pol = np.radians(np.arange(0, 90 + resolution, resolution))
        AZI, POL = np.meshgrid(azi, pol)
        
        X_sph = np.sin(POL) * np.cos(AZI)
        Y_sph = np.sin(POL) * np.sin(AZI)
        Z_sph = np.cos(POL)
        
        polar_y = POL.ravel()
        azim_y = AZI.ravel()
        
        # Empaquetamos las direcciones de la grilla en un Vector3d de orix
        direcciones_muestra = Vector3d(np.column_stack((X_sph.ravel(), Y_sph.ravel(), Z_sph.ravel())))

        # 2. Obtener polos cristalográficos simetrizados en coordenadas cartesianas (Vector3d)
        pg = crystal_sym.point_group if hasattr(crystal_sym, 'point_group') else crystal_sym
        hkl_simetricos = pg * hkl_miller
        
        # Normalizamos los vectores de los planos hkl a longitud 1
        coords_hkl = hkl_simetricos.data.astype(float)
        normas = np.linalg.norm(coords_hkl, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        polos_simetricos = coords_hkl / normas
        num_h = len(polos_simetricos)
        
        # 3. Sumatoria de Fourier (Ley de Friedel: L pares)
        intensities = np.zeros(len(polar_y), dtype=complex)
        L_max = max([key[0] for key in coefs.keys()])
        
        for h_vec in polos_simetricos:
            hx, hy, hz = h_vec
            polar_h = np.arccos(np.clip(hz, -1.0, 1.0))
            azim_h = np.arctan2(hy, hx)

            for l in range(0, L_max + 1, 2):  # Solo pares
                factor = (4.0 * np.pi) / (2 * l + 1)
                for m in range(-l, l + 1):
                    for n in range(-l, l + 1):
                        C_lmn = coefs.get((l, m, n), 0.0j)
                        if abs(C_lmn) > 1e-8:
                            # sph_harm mapeado correctamente gracias al bloque try/except
                            Y_h = sph_harm(m, l, azim_h, polar_h)
                            Y_y = sph_harm(n, l, azim_y, polar_y)
                            
                            intensities += C_lmn * factor * np.conj(Y_h) * Y_y / num_h

        # 4. Limpiamos ruido numérico (Gibbs) garantizando intensidad positiva
        intensidades_finales = np.maximum(np.real(intensities), 0.0)
        
        print(f"    [Completado en {time.time() - t0:.2f} s - Máximo MUD: {np.max(intensidades_finales):.2f}]")

        # 5. Instanciamos y retornamos el objeto PoleFigure
        return cls(direcciones=direcciones_muestra, 
                   intensidades=intensidades_finales, 
                   hkl=hkl_miller)


    def rotate(self, rotacion):
        """
        Rota la Figura de Polos aplicando una rotación espacial a sus direcciones.
        Retorna un nuevo objeto PoleFigure con la rotación aplicada (inmutable).
        """
        if isinstance(rotacion, (list, tuple, np.ndarray)):
            angulos_rad = np.radians(rotacion)
            rot_obj = Rotation.from_euler(angulos_rad)
        elif isinstance(rotacion, (Rotation, Orientation)):
            rot_obj = rotacion
        else:
            raise TypeError("La rotación debe ser un objeto Orientation, Rotation o una lista de Euler [phi1, Phi, phi2].")
            
        direcciones_rotadas = rot_obj * self.direcciones
        
        coords = direcciones_rotadas.data
        coords_upper = np.where(coords[:, 2:3] < 0, -coords, coords)
        nuevas_direcciones = Vector3d(coords_upper)
        
        return PoleFigure(direcciones=nuevas_direcciones, 
                          intensidades=self.intensidades, 
                          hkl=self.hkl)

    def plot(self, ax=None, cmap='jet', niveles=25, max_val=None, direccion_x='horizontal'):
        """
        Grafica la Figura de Polos usando proyección de igual área (tricontourf).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            creo_figura = True
        else:
            creo_figura = False

        xp, yp = proyectar_igual_area(self.direcciones.data, np.ones(self.direcciones.size))
        
        if direccion_x == 'vertical':
            xp_rot = -yp
            yp_rot = xp
            xp, yp = xp_rot, yp_rot

        intensidades_limpias = np.nan_to_num(self.intensidades, nan=0.0)
        
        if max_val is None:
            max_val = np.max(intensidades_limpias)
        if max_val <= 0.01:
            max_val = 1.0 
            
        niveles_contour = np.linspace(0, max_val, niveles)

        cf = ax.tricontourf(xp, yp, intensidades_limpias, levels=niveles_contour, cmap=cmap)
        
        circ = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(circ), np.sin(circ), 'k', lw=1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        offset = 1.15
        if direccion_x == 'horizontal':
            ax.text(offset, 0.0, 'X', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(0.0, offset, 'Y', ha='center', va='center', fontsize=12, fontweight='bold')
        elif direccion_x == 'vertical':
            ax.text(0.0, offset, 'X', ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(-offset, 0.0, 'Y', ha='center', va='center', fontsize=12, fontweight='bold')
        
        hkl_vals = self.hkl.hkil.flatten() if hasattr(self.hkl, 'hkil') else self.hkl.hkl.flatten()
        label_plano = ''.join([str(int(x)) for x in hkl_vals])
        ax.set_title(f"Polo {{{label_plano}}}", fontsize=14, pad=20)
        
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label('Intensidad (MUD)')
        
        if creo_figura:
            plt.tight_layout()
            return fig, ax
        return cf


# ====================================================================
# FUNCIONES GLOBALES DE VISUALIZACIÓN MULTIPLE
# ====================================================================

def plot_pfs(lista_pfs, titulos=None, cmap='jet', max_val_global=False, direccion_x='horizontal'):
    n = len(lista_pfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
        
    max_val = None
    if max_val_global:
        max_val = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in lista_pfs])

    for i, ax in enumerate(axes):
        pf = lista_pfs[i]
        pf.plot(ax=ax, cmap=cmap, max_val=max_val, direccion_x=direccion_x)
        if titulos is not None:
            ax.set_title(titulos[i], fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


def plot_pf_comparison(pfs_in, pfs_out, titulos=None, suptitle="Comparativa de Figuras de Polos", cmap='jet', unificar_escala='hkl', direccion_x='horizontal'):
    n = len(pfs_in)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    fig.suptitle(suptitle, fontsize=16)
    if n == 1: axes = axes.reshape(2, 1)

    max_in_global = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in pfs_in])
    max_out_global = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in pfs_out])
    max_total = max(max_in_global, max_out_global)

    for i in range(n):
        pf_in = pfs_in[i]
        pf_out = pfs_out[i]
        
        max_in_local = np.max(np.nan_to_num(pf_in.intensidades, nan=0.0))
        max_out_local = np.max(np.nan_to_num(pf_out.intensidades, nan=0.0))
        max_par = max(max_in_local, max_out_local)
        
        if unificar_escala == 'global': max_in, max_out = max_total, max_total
        elif unificar_escala == 'hkl': max_in, max_out = max_par, max_par
        elif unificar_escala == 'fila': max_in, max_out = max_in_global, max_out_global
        else: max_in, max_out = None, None
        
        pf_in.plot(ax=axes[0, i], cmap=cmap, max_val=max_in, direccion_x=direccion_x)
        titulo_base = axes[0, i].get_title()
        pf_out.plot(ax=axes[1, i], cmap=cmap, max_val=max_out, direccion_x=direccion_x)
        
        if titulos is not None: titulo_base = titulos[i]
            
        axes[0, i].set_title(f"Entrada - {titulo_base}", fontsize=14, pad=20)
        axes[1, i].set_title(f"Salida - {titulo_base}", fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()