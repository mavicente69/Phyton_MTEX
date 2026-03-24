# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:21:02 2026

@author: mavic
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 14:21:02 2026

@author: mavic
"""
# -*- coding: utf-8 -*-
"""
Script principal para evaluación de texturas cristalográficas.
Entorno: texturaPy3.10
"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Rotation, Orientation
from orix.crystal_map import Phase
from orix.vector import Miller
from diffpy.structure import Lattice, Structure

# IMPORTACIÓN DE TUS RUTINAS PERSONALIZADAS
import utils_orient
import utils_sym  
from utils_kernels import OrientationKernel # Importamos la clase

# 1. CONFIGURACIÓN DE FASE Y SIMETRÍAS
print("Cargando fase y simetrías...")
samp_sym = utils_sym.obtener_simetria('mmm')   
samp_sym = utils_sym.obtener_simetria('-1')   
celda_hex = Lattice(2.95, 2.95, 4.68, 90, 90, 120)
estructura_hex = Structure(lattice=celda_hex)

fase_hex = Phase(
    name="Titanium",
    space_group=194, 
    structure=estructura_hex
)
cryst_sym = fase_hex.point_group

# 2. DEFINIR COMPONENTES IDEALES, PESOS Y SUS KERNELS
print("Definiendo componentes de textura, pesos y kernels...")

data_eulers = np.deg2rad([
    [0, 20, 0],         # Componente 1 (Basal)
    [90, 90, 0],
    [0,45,0]        # Componente 2 (Prismática / Rotada)
])

rot_comp = Rotation.from_euler(data_eulers)
ori_comp = Orientation(rot_comp) 

# Pesos (fracciones de volumen)
pesos_componentes = [0, 0, 1] 

# Instanciamos un kernel individual para cada componente
# La basal la hacemos bien aguda (Gaussian, 10°), la prismática más difusa (Lorentzian, 25°)
kernel_basal = OrientationKernel(fwhm_grados=10, tipo='gaussian')
kernel_prism = OrientationKernel(fwhm_grados=10, tipo='lorentzian')
kernel_poussin = OrientationKernel(fwhm_grados=10, tipo='poussin')

lista_kernels = [kernel_basal, kernel_prism, kernel_poussin]

# 3. DEFINIR LOS PLANOS DE MILLER
planos = [
    Miller(hkil=[0, 0, 0, 1], phase=fase_hex),   
    Miller(hkil=[1, 0, -1, 0], phase=fase_hex),  
    Miller(hkil=[1, 0, -1, 2], phase=fase_hex)   
]

# 4. EJECUTAR LA VISUALIZACIÓN POR GRILLA ESFÉRICA
print("Calculando densidades analíticas sobre la grilla...")
fig, axes = utils_orient.plot_pole_figure_grid_density(
    ori_comp, 
    planos, 
    pesos=pesos_componentes,
    kernels=lista_kernels, # <--- Pasamos la lista de objetos aquí
    crystal_sym=cryst_sym, 
    sample_sym=samp_sym,
    resolution=150   
)

# 5. AJUSTES FINALES Y MOSTRAR
plt.show()