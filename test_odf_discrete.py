# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:07:21 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Prueba Unitaria: ODF Discreta con Componente Ensanchada (10°)
Entorno: texturaPy3.10
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.vector import Miller
from diffpy.structure import Lattice, Structure
from orix.quaternion import Orientation
from orix.quaternion.symmetry import D2h, C1

from utils_odf import ODFDiscreta

def main():
    print("--- INICIANDO PRUEBA UNITARIA: MANCHA DISCRETA (10°) ---")
    
    # 1. Configurar Zircaloy-4
    lat_zr = Lattice(3.232, 3.232, 5.147, 90, 90, 120)
    fase_zr = Phase(name="Zirconium", space_group=194, structure=Structure(lattice=lat_zr))
    
    # 2. Fabricar la "Grilla en Blanco" (Resolución 5°)
    res = 5
    p1 = np.arange(0, 360 + res, res)
    P  = np.arange(0, 90 + res, res)
    p2 = np.arange(0, 60 + res, res)
    
    P1, PP, P2 = np.meshgrid(p1, P, p2, indexing='ij')
    euler_grid = np.column_stack((P1.ravel(), PP.ravel(), P2.ravel()))
    
    orientaciones = Orientation.from_euler(np.radians(euler_grid), symmetry=fase_zr.point_group).unique()
    
    # Fondo plano casi cero
    pesos = np.ones(orientaciones.size) * 0.01 
    
    # 3. Inyectar la "Burbuja" de 10° alrededor de (0, 30, 0)
    target_rot = Orientation.from_euler(np.radians([[45, 30, 0]]))
    
    # Calculamos la distancia de toda la red a nuestro punto central
    distancias = orientaciones.angle_with(target_rot)
    
    # Filtramos los nodos que están a menos de 10°
    radio_rad = np.radians(10.0)
    mask_burbuja = distancias <= radio_rad
    
    # Les damos peso usando una campana de Gauss suave para mayor realismo
    # El peso máximo será 25 en el centro, y va decayendo hacia los bordes de los 10°
    sigma_burbuja = np.radians(4.0)
    pesos[mask_burbuja] = 25.0 * np.exp(-0.5 * (distancias[mask_burbuja] / sigma_burbuja)**2)
    
    nodos_encendidos = np.sum(mask_burbuja)
    print(f" -> Centro ideal inyectado en: (0, 30, 0)")
    print(f" -> Nodos de la red de 5° encendidos para formar el ancho de 10°: {nodos_encendidos} orientaciones")
    
    # 4. Instanciar la clase (Extrayendo la simetría pura de la fase)
    mi_odf = ODFDiscreta(
    orientaciones=orientaciones, 
    pesos=pesos, 
    crystal_sym=fase_zr.point_group, # <--- ACÁ ESTÁ EL ARREGLO
    sample_sym=C1
    )   
    
    # ==========================================
    # PRUEBA A: CORTES DE LA ODF
    # ==========================================
    print("\nGraficando cortes de la ODF...")
    mi_odf.plot_sections(sections=[0, 30], axis='phi2', res_grados=5)
    
    # ==========================================
    # PRUEBA B: FIGURAS DE POLOS RECALCULADAS
    # ==========================================
    print("\nRecalculando Figuras de Polos...")
    planos = [
        Miller(hkil=[0, 0, 0, 2], phase=fase_zr),
        Miller(hkil=[1, 0, -1, 0], phase=fase_zr)
    ]
    
    pfs_calc = mi_odf.calc_pole_figures(lista_hkl=planos, resolution=100)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, pf in enumerate(pfs_calc):
        hkl_str = ''.join([str(int(x)) for x in planos[i].hkil.flatten()])
        pf.plot(ax=axes[i], cmap='jet', direccion_x='vertical')
        axes[i].set_title(f"PF Recalculada {{{hkl_str}}}")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()