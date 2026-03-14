# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:13:46 2026

@author: mavic
"""

import numpy as np
import orix
from orix.quaternion import Rotation, Orientation
from orix.vector import Vector3d 
import matplotlib.pyplot as plt

# 1. Importamos la función de visualización desde tus utilidades
# Asegúrate de que el nombre del archivo sea correcto
from utils_vector3d import plot_pole_figure_manual

# 2. Datos de entrada (Orientaciones en grados y pesos)
# Usamos la convención Bunge (ZXZ)
data = np.array([
    [0, 0, 0, 1.0],      # Componente Cubo (Polo 001 al centro)
    [0, 90, 0, 0.8],     # Rotada 90° en Phi (Polo 001 se mueve al borde en Y)
    [0, 60, 0, 1.2],     # Inclinada 45° en Phi
    [30, 15, 20, 0.5]    # Orientación genérica
])

# Conversión a radianes y separación de pesos
eulers = np.deg2rad(data[:, :3])
pesos = data[:, 3]

# 3. Procesamiento de Orientaciones con Orix
# Creamos el objeto de rotación y lo envolvemos en Orientation
rot = Rotation.from_euler(eulers)
ori = Orientation(rot)

# Definimos el polo cristalino que queremos observar
# Para validar la rotación en X (Phi), el eje [0, 0, 1] es el más claro
hkl = Vector3d([0, 0, 1])
polos_rotados = ori * hkl 

# 4. Generación de la Figura de Polos usando la función modular
# Esta función ya gestiona la proyección, los ejes X-Y-Z y la estética
fig, ax = plot_pole_figure_manual(
    polos_rotados, 
    pesos=pesos, 
    titulo="Aluminium Texture Validation", 
    hkl_label=f"Polo {hkl.data[0].astype(int)}"
)

# 5. Mostrar resultado
plt.show()