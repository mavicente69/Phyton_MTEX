# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:17:01 2026

@author: mavic
"""

import numpy as np
import matplotlib.pyplot as plt
from orix.vector import Vector3d
from orix.quaternion import Rotation, Orientation
from utils_vector3d import proyectar_igual_area

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

    def rotate(self, rotacion):
        """
        Rota la Figura de Polos aplicando una rotación espacial a sus direcciones.
        Retorna un nuevo objeto PoleFigure con la rotación aplicada (inmutable).
        
        rotacion: Puede ser un objeto Orientation/Rotation de orix o una lista de Euler [phi1, Phi, phi2].
        """
        # 1. Procesamos la entrada (ahora acepta Orientation nativo)
        if isinstance(rotacion, (list, tuple, np.ndarray)):
            angulos_rad = np.radians(rotacion)
            rot_obj = Rotation.from_euler(angulos_rad)
        elif isinstance(rotacion, (Rotation, Orientation)):
            rot_obj = rotacion
        else:
            raise TypeError("La rotación debe ser un objeto Orientation, Rotation o una lista de Euler [phi1, Phi, phi2].")
            
        # 2. Aplicamos la rotación matemática a los vectores
        direcciones_rotadas = rot_obj * self.direcciones
        
        # 3. Simetría antipodal: Si un polo cae al hemisferio sur (Z < 0), lo proyectamos al norte
        coords = direcciones_rotadas.data
        coords_upper = np.where(coords[:, 2:3] < 0, -coords, coords)
        nuevas_direcciones = Vector3d(coords_upper)
        
        # 4. Devolvemos una NUEVA Figura de Polos
        return PoleFigure(direcciones=nuevas_direcciones, 
                          intensidades=self.intensidades, 
                          hkl=self.hkl)

    def plot(self, ax=None, cmap='jet', niveles=25, max_val=None, direccion_x='horizontal'):
        """
        Grafica la Figura de Polos usando proyección de igual área.
        direccion_x: 'horizontal' (X apunta al Este, Y al Norte) o 
                     'vertical' (X apunta al Norte, Y al Oeste).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            creo_figura = True
        else:
            creo_figura = False

        xp, yp = proyectar_igual_area(self.direcciones.data, np.ones(self.direcciones.size))
        
        # Aplicamos la rotación geométrica para el ploteo si el usuario pide X vertical
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
        
        # Etiquetado automático de los Ejes de la Muestra
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
    """
    Grafica una lista de Figuras de Polos en una sola fila.
    Aprovecha el método interno .plot() de cada PoleFigure.
    """
    n = len(lista_pfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    
    if n == 1:
        axes = [axes]
        
    # Calcular máximo global si se solicita para unificar la barra de colores
    max_val = None
    if max_val_global:
        max_val = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in lista_pfs])

    for i, ax in enumerate(axes):
        pf = lista_pfs[i]
        
        # Usamos tu método nativo de ploteo con tricontourf
        pf.plot(ax=ax, cmap=cmap, max_val=max_val, direccion_x=direccion_x)
        
        # Sobrescribimos el título si el usuario proveyó uno personalizado
        if titulos is not None:
            ax.set_title(titulos[i], fontsize=14, pad=20)

    plt.tight_layout()
    plt.show()


def plot_pf_comparison(pfs_in, pfs_out, titulos=None, suptitle="Comparativa de Figuras de Polos", cmap='jet', unificar_escala='hkl', direccion_x='horizontal'):
    """
    Grafica dos listas de Figuras de Polos en 2 filas (Entrada vs Salida) para evaluar inversiones.
    unificar_escala: 
        - 'global': Misma escala para TODAS las 10 figuras.
        - 'hkl': Cada par Entrada/Salida del mismo plano comparte su propia escala.
        - 'fila': Una escala para toda la Entrada y otra para toda la Salida.
        - 'independiente': Cada figura individual tiene su propia escala.
    """
    n = len(pfs_in)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    fig.suptitle(suptitle, fontsize=16)
    
    if n == 1:
        axes = axes.reshape(2, 1)

    # Calcular máximos globales (por si se usa 'global' o 'fila')
    max_in_global = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in pfs_in])
    max_out_global = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in pfs_out])
    max_total = max(max_in_global, max_out_global)

    for i in range(n):
        pf_in = pfs_in[i]
        pf_out = pfs_out[i]
        
        # Máximos locales del par (mismo hkl)
        max_in_local = np.max(np.nan_to_num(pf_in.intensidades, nan=0.0))
        max_out_local = np.max(np.nan_to_num(pf_out.intensidades, nan=0.0))
        max_par = max(max_in_local, max_out_local)
        
        # Selección de la escala según el modo
        if unificar_escala == 'global':
            max_in, max_out = max_total, max_total
        elif unificar_escala == 'hkl':
            max_in, max_out = max_par, max_par
        elif unificar_escala == 'fila':
            max_in, max_out = max_in_global, max_out_global
        else: # independiente
            max_in, max_out = None, None
        
        # Fila 1: Entrada
        pf_in.plot(ax=axes[0, i], cmap=cmap, max_val=max_in, direccion_x=direccion_x)
        titulo_base = axes[0, i].get_title()
        
        # Fila 2: Salida
        pf_out.plot(ax=axes[1, i], cmap=cmap, max_val=max_out, direccion_x=direccion_x)
        
        # Aplicamos títulos personalizados o usamos los por defecto
        if titulos is not None:
            titulo_base = titulos[i]
            
        axes[0, i].set_title(f"Entrada - {titulo_base}", fontsize=14, pad=20)
        axes[1, i].set_title(f"Salida - {titulo_base}", fontsize=14, pad=20)

    plt.tight_layout()