# -*- coding: utf-8 -*-
"""
Módulo de Figuras de Polos (PF).
Define la clase PoleFigure como un objeto independiente que encapsula 
la grilla estereográfica, las intensidades y los métodos de graficado.
Entorno: texturaPy3.10
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from orix.vector import Vector3d
from orix.quaternion import Rotation, Orientation
from utils_vector3d import proyectar_igual_area

def _es_hexagonal(simetria, hkl):
    """
    Detecta si la fase a proyectar es hexagonal. Busca tanto en el argumento
    directo de simetría como en la metadata profunda del objeto Miller.
    """
    if simetria is not None:
        s = str(getattr(simetria, 'name', str(simetria))).lower()
        if '6' in s or 'hex' in s or 'd6' in s or 'c6' in s: 
            return True
            
    if hasattr(hkl, 'phase') and hkl.phase is not None:
        # 1. Buscamos el nombre del grupo puntual (Ej. "6/mmm")
        pg = getattr(hkl.phase, 'point_group', None)
        if pg is not None:
            s = str(getattr(pg, 'name', str(pg))).lower()
            if '6' in s or 'hex' in s or 'd6' in s or 'c6' in s:
                return True
                
        # 2. Respaldo: Buscamos en el nombre de la fase (por si le pusieron "Titanio Alpha Hexagonal")
        p = str(getattr(hkl.phase, 'name', '')).lower()
        if 'hex' in p: 
            return True
            
    return False

class PoleFigure:
    def __init__(self, direcciones, intensidades, hkl):
        if not isinstance(direcciones, Vector3d):
            raise TypeError("Error: 'direcciones' debe ser un objeto Vector3d de la librería orix.")
        self.direcciones = Vector3d(direcciones.data.reshape(-1, 3))
        self.intensidades = np.asarray(intensidades).flatten()
        self.hkl = hkl
        
    @classmethod
    def from_fourier(cls, coefs_array, hkl_miller, crystal_sym, res_grados=5.0):
        t0 = time.time()
        paso = res_grados / 90.0  
        puntos_2d = []
        n_max = int(np.ceil(2.0 / paso))
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                x, y = i * paso + j * (paso / 2.0), j * (paso * np.sqrt(3) / 2.0)
                if x**2 + y**2 <= 1.0 + 1e-6: puntos_2d.append([x, y])
        puntos_2d = np.array(puntos_2d)
        
        R = np.clip(np.linalg.norm(puntos_2d, axis=1), 0.0, 1.0)
        theta, phi = 2.0 * np.arcsin(R / np.sqrt(2.0)), np.arctan2(puntos_2d[:, 1], puntos_2d[:, 0])
        polar_y, azim_y = theta, np.mod(phi, 2 * np.pi)
        
        # EXTRACCIÓN CON ORIX: Respeta la matriz de orientación del cristal
        v_h = Vector3d(hkl_miller)
        polar_h = v_h.polar.flatten()[0]
        azim_h = v_h.azimuth.flatten()[0]
        
        # PARCHE CONDICIONAL DE BUNGE PARA REDES HEXAGONALES
        if _es_hexagonal(crystal_sym, hkl_miller):
            azim_h += np.pi / 2.0
        
        from utils_fourier import eval_pf_from_wigner
        intensidades = eval_pf_from_wigner(coefs_array, polar_h, azim_h, polar_y, azim_y)

        direcciones_muestra = Vector3d(np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))))
        pf_obj = cls(direcciones=direcciones_muestra, intensidades=intensidades, hkl=hkl_miller)
        pf_obj.puntos_2d = puntos_2d
        return pf_obj

    @classmethod
    def from_bunge(cls, C_bunge, A_cryst, B_samp, hkl_miller, res_grados=5.0):
        t0 = time.time()
        paso = res_grados / 90.0  
        puntos_2d = []
        n_max = int(np.ceil(2.0 / paso))
        for i in range(-n_max, n_max + 1):
            for j in range(-n_max, n_max + 1):
                x, y = i * paso + j * (paso / 2.0), j * (paso * np.sqrt(3) / 2.0)
                if x**2 + y**2 <= 1.0 + 1e-6: puntos_2d.append([x, y])
        puntos_2d = np.array(puntos_2d)
        
        R = np.clip(np.linalg.norm(puntos_2d, axis=1), 0.0, 1.0)
        theta, phi = 2.0 * np.arcsin(R / np.sqrt(2.0)), np.arctan2(puntos_2d[:, 1], puntos_2d[:, 0])
        polar_y, azim_y = theta, np.mod(phi, 2 * np.pi)
        
        # EXTRACCIÓN CON ORIX
        v_h = Vector3d(hkl_miller)
        polar_h = v_h.polar.flatten()[0]
        azim_h = v_h.azimuth.flatten()[0]
        
        # EL PARCHE QUE FALLABA: Ahora _es_hexagonal busca a fondo en hkl_miller
        if _es_hexagonal(None, hkl_miller):
            azim_h += np.pi / 2.0
        
        from utils_fourier import eval_pf_from_bunge
        intensidades = eval_pf_from_bunge(C_bunge, polar_h, azim_h, polar_y, azim_y, A_cryst, B_samp)
        
        direcciones_muestra = Vector3d(np.column_stack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))))
        pf_obj = cls(direcciones=direcciones_muestra, intensidades=intensidades, hkl=hkl_miller)
        pf_obj.puntos_2d = puntos_2d
        return pf_obj

    def rotate(self, rotacion):
        if isinstance(rotacion, (list, tuple, np.ndarray)):
            rot_obj = Rotation.from_euler(np.radians(rotacion))
        elif isinstance(rotacion, (Rotation, Orientation)):
            rot_obj = rotacion
        else:
            raise TypeError("La rotación debe ser Orientation, Rotation o Euler.")
        coords = (rot_obj * self.direcciones).data
        pf_rotada = PoleFigure(direcciones=Vector3d(np.where(coords[:, 2:3] < 0, -coords, coords)), intensidades=self.intensidades, hkl=self.hkl)
        if hasattr(self, 'puntos_2d'): pf_rotada.puntos_2d = self.puntos_2d
        return pf_rotada

    def plot(self, ax=None, cmap='jet', niveles=25, max_val=None, direccion_x='horizontal'):
        if ax is None: fig, ax = plt.subplots(figsize=(6, 6))
        xp, yp = proyectar_igual_area(self.direcciones.data, np.ones(self.direcciones.size))
        if direccion_x == 'vertical': xp, yp = -yp, xp
        intensidades = np.nan_to_num(self.intensidades, nan=0.0)
        if max_val is None: max_val = np.max(intensidades)
        cf = ax.tricontourf(xp, yp, intensidades, levels=np.linspace(0, max_val if max_val > 0.01 else 1.0, niveles), cmap=cmap)
        circ = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(circ), np.sin(circ), 'k', lw=1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        offset = 1.15
        if direccion_x == 'horizontal':
            ax.text(offset, 0, 'X', ha='center', va='center', fontweight='bold')
            ax.text(0, offset, 'Y', ha='center', va='center', fontweight='bold')
        elif direccion_x == 'vertical':
            ax.text(0, offset, 'X', ha='center', va='center', fontweight='bold')
            ax.text(-offset, 0, 'Y', ha='center', va='center', fontweight='bold')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label('Intensidad (MUD)')
        return cf

def plot_pfs(lista_pfs, titulos=None, cmap='jet', max_val_global=False, direccion_x='horizontal'):
    n = len(lista_pfs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    max_val = max([np.max(np.nan_to_num(pf.intensidades, nan=0.0)) for pf in lista_pfs]) if max_val_global else None
    for i, ax in enumerate(axes):
        lista_pfs[i].plot(ax=ax, cmap=cmap, max_val=max_val, direccion_x=direccion_x)
        if titulos is not None: ax.set_title(titulos[i], fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def plot_pf_comparison(pfs_in, pfs_out, titulos=None, suptitle="Comparativa de Figuras de Polos", unificar_escala='hkl', direccion_x='horizontal'):
    n = len(pfs_in)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1: axes = np.array([[axes[0]], [axes[1]]])
        
    for i in range(n):
        max_val = None
        if unificar_escala == 'hkl':
            max_in = np.max(np.nan_to_num(pfs_in[i].intensidades, nan=0.0))
            max_out = np.max(np.nan_to_num(pfs_out[i].intensidades, nan=0.0))
            max_val = max(max_in, max_out)

        pfs_in[i].plot(ax=axes[0, i], cmap='jet', max_val=max_val, direccion_x=direccion_x)
        titulo_in = f"{titulos[i]}\n(Original)" if titulos else "Original"
        axes[0, i].set_title(titulo_in, fontsize=14, pad=15)

        pfs_out[i].plot(ax=axes[1, i], cmap='jet', max_val=max_val, direccion_x=direccion_x)
        titulo_out = f"{titulos[i]}\n(Recalculada)" if titulos else "Recalculada"
        axes[1, i].set_title(titulo_out, fontsize=14, pad=15)

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight='bold', y=1.02)
        
    plt.tight_layout()
    plt.show()