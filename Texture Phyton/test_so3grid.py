# -*- coding: utf-8 -*-
import numpy as np
from orix.quaternion.symmetry import D6h
from orix.sampling import get_sample_fundamental
from orix.quaternion import Orientation

class SO3Grid:
    """
    Generador de grillas uniformes en el espacio de orientaciones SO(3).
    Utiliza muestreo cubocórico dentro de la zona fundamental de la simetría.
    Equivalente directo a equispacedSO3Grid de MTEX.
    """
    def __init__(self, resolucion_grados, simetria_cristal, metodo='cubochoric'):
        self.resolucion = resolucion_grados
        self.simetria = simetria_cristal
        
        print(f"\n[SO3Grid] Generando grilla SO(3) con resolución {resolucion_grados}°...")
        
        # 1. Generamos el muestreo uniforme en la zona fundamental
        # ¡Acá estaba el bug! El parámetro correcto es point_group, no symmetry.
        rotaciones = get_sample_fundamental(
            resolution=resolucion_grados, 
            point_group=simetria_cristal,  # <--- CORRECCIÓN APLICADA
            method=metodo
        )
        
        # 2. Las encapsulamos como objeto Orientation con su simetría
        self.orientaciones = Orientation(rotaciones.data, symmetry=simetria_cristal)
        self.N = self.orientaciones.size
        print(f"[SO3Grid] ¡Listo! {self.N} orientaciones equiespaciadas generadas.")
        
    def to_euler(self, degrees=True):
        """Devuelve los ángulos de Euler de la grilla."""
        return self.orientaciones.to_euler(degrees=degrees)

if __name__ == "__main__":
    # ==============================================================
    # ENSAYO: Grilla SO(3) vs Grilla de Euler Tradicional
    # ==============================================================
    # Usamos la clase D6h (que equivale a 6/mmm en Orix)
    fase_hex = D6h
    resolucion = 5.0  # Grados
    
    # 1. Generamos la nueva grilla topológica SO(3)
    mi_grilla = SO3Grid(resolucion_grados=resolucion, simetria_cristal=fase_hex)
    
    # 2. Generamos la vieja grilla de Euler (como estaba en utils_inversion)
    p1 = np.arange(0, 360 + resolucion, resolucion)
    P  = np.arange(0, 90 + resolucion, resolucion)
    p2 = np.arange(0, 60 + resolucion, resolucion)
    
    P1, PP, P2 = np.meshgrid(p1, P, p2, indexing='ij')
    euler_viejo = np.column_stack((P1.ravel(), PP.ravel(), P2.ravel()))
    
    # Limpiamos las simetrías redundantes como hacía nuestro código
    oris_viejas = Orientation.from_euler(np.radians(euler_viejo), symmetry=fase_hex).unique()
    
    # 3. Comparamos resultados
    print("\n=========================================================")
    print(f"📊 COMPARATIVA DE GRILLAS (Resolución {resolucion}°)")
    print("=========================================================")
    print(f" -> Puntos Grilla Euler (Vieja)   : {oris_viejas.size}")
    print(f" -> Puntos Grilla SO(3) (Nueva)   : {mi_grilla.N}")
    
    ahorro = 100 * (1 - mi_grilla.N / oris_viejas.size)
    print(f"---------------------------------------------------------")
    print(f"🌟 Reducción del tamaño de la matriz : {ahorro:.1f} %")
    print(f"=========================================================\n")