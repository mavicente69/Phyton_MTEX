# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:44:33 2026

@author: mavic
"""

import numpy as np
import matplotlib.pyplot as plt

def proyectar_igual_area(vectores_3d, pesos):
    """
    Proyección de Igual Área (Lambert) - Estándar Cartesiano Puro.
    Centro = Z, Derecha (Este) = Eje X, Superior (Norte) = Eje Y
    """
    # Normalización de seguridad
    normas = np.linalg.norm(vectores_3d, axis=1, keepdims=True)
    normas[normas == 0] = 1 # Evitar división por cero
    v = vectores_3d / normas
    
    x_vec = v[:, 0]
    y_vec = v[:, 1]
    z_vec = v[:, 2]

    # Factor de escala exacto para Igual Área (Lambert)
    factor = np.sqrt(1 / (1 + np.abs(z_vec))) 

    # Mapeo CARTESIANO PURO (X 3D -> X 2D, Y 3D -> Y 2D)
    x_plot = x_vec * factor
    y_plot = y_vec * factor
    
    return x_plot, y_plot

def plot_pole_figure_manual(vectores, pesos=None, titulo="Pole Figure", hkl_label=""):
    """
    Genera una Figura de Polos directa para debugging rápido.
    Convención por defecto: X derecha, Y arriba, Z centro.
    """
    if hasattr(vectores, 'data'):
        data_3d = vectores.data
    else:
        data_3d = vectores

    if pesos is None:
        pesos = np.ones(len(data_3d))

    xp, yp = proyectar_igual_area(data_3d, pesos)

    fig, ax = plt.subplots(figsize=(7, 7))
    
    circle_theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), 'k', linewidth=1.5)

    scatter = ax.scatter(xp, yp, c=pesos, s=pesos*100, 
                         cmap='viridis', edgecolors='k', alpha=0.9, zorder=3)

    ax.text(1.07, 0, 'X', fontsize=14, fontweight='bold', ha='left', va='center')
    ax.text(0, 1.07, 'Y', fontsize=14, fontweight='bold', ha='center', va='bottom')
    ax.text(0.03, 0.03, 'Z', fontsize=14, fontweight='bold')

    ax.set_aspect('equal')
    ax.axis('off')
    
    full_title = f"{titulo} {hkl_label}"
    plt.title(full_title, fontsize=14, pad=20)
    plt.colorbar(scatter, label='Intensity')

    return fig, ax