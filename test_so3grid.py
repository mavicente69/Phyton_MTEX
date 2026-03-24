# -*- coding: utf-8 -*-
import numpy as np
from orix.quaternion.symmetry import D6h, Oh  # Importamos Hexagonal (D6h) y Cúbico (Oh)
from orix.sampling import get_sample_fundamental
from orix.quaternion import Orientation

class SO3Grid:
    """
    Generador de grillas uniformes en el espacio de orientaciones SO(3).
    Soporta la reducción del espacio mediante la intersección de la 
    Zona Fundamental del Cristal y de la Muestra.
    """
    def __init__(self, resolucion_grados, simetria_cristal, simetria_muestra=None, metodo='cubochoric'):
        self.resolucion = resolucion_grados
        self.simetria_cristal = simetria_cristal
        self.simetria_muestra = simetria_muestra
        
        # 1. Muestreo de la FZ del cristal (Orix nativo)
        rotaciones = get_sample_fundamental(
            resolution=resolucion_grados, 
            point_group=simetria_cristal,
            method=metodo
        )
        oris_cristal = Orientation(rotaciones.data, symmetry=simetria_cristal)
        
        # 2. Recorte por la FZ de la muestra (Intersección G_cristal x G_muestra)
        self.limite_phi1 = 360.0
        nom_muestra = getattr(simetria_muestra, 'name', str(simetria_muestra)).lower() if simetria_muestra else 'none'

        if nom_muestra in ['222', 'mmm', 'd2h', 'm-m-m', 'orthorhombic']:
            self.limite_phi1 = 90.0
        elif nom_muestra in ['2/m', '2', 'm', 'monoclinic']:
            self.limite_phi1 = 180.0
            
        # Filtramos la grilla si la muestra tiene simetría
        if self.limite_phi1 < 360.0:
            angulos_euler = oris_cristal.to_euler(degrees=True)
            # Damos 1e-3 de tolerancia por errores de redondeo de punto flotante
            mascara_adentro_fz = angulos_euler[:, 0] <= (self.limite_phi1 + 1e-3)
            self.orientaciones = oris_cristal[mascara_adentro_fz]
        else:
            self.orientaciones = oris_cristal

        self.N = self.orientaciones.size
        
    @property
    def FZ(self):
        """Devuelve un resumen de los límites geométricos de la Zona Fundamental."""
        angulos = self.to_euler(degrees=True)
        return {
            'Simetria_Cristal': getattr(self.simetria_cristal, 'name', str(self.simetria_cristal)),
            'Simetria_Muestra': str(self.simetria_muestra),
            'Limites_Euler (Max)': {
                'phi1': self.limite_phi1,
                'Phi': round(np.max(angulos[:, 1]), 1) if self.N > 0 else 0,
                'phi2': round(np.max(angulos[:, 2]), 1) if self.N > 0 else 0
            },
            'Puntos_Totales': self.N
        }

    def to_euler(self, degrees=True):
        return self.orientaciones.to_euler(degrees=degrees)

if __name__ == "__main__":
    resolucion = 5.0  # Grados
    
    print(f"=========================================================")
    print(f"🔬 ENSAYO DE ZONAS FUNDAMENTALES SO(3) - RES: {resolucion}°")
    print(f"=========================================================")

    # --- CASO 1: Hexagonal sin simetría de muestra ---
    print("\n[CASO 1] Hexagonal (Zircaloy) sin simetría de muestra")
    grilla_hex_triclinic = SO3Grid(resolucion, D6h, simetria_muestra=None)
    fz_1 = grilla_hex_triclinic.FZ
    print(f" -> Cristal: {fz_1['Simetria_Cristal']} | Muestra: {fz_1['Simetria_Muestra']}")
    print(f" -> Rango de Euler Max: phi1={fz_1['Limites_Euler (Max)']['phi1']}° | Phi={fz_1['Limites_Euler (Max)']['Phi']}° | phi2={fz_1['Limites_Euler (Max)']['phi2']}°")
    print(f" -> Puntos (Tamaño Matriz): {fz_1['Puntos_Totales']}")

    # --- CASO 2: Hexagonal en chapa laminada (Ortorrómbica) ---
    print("\n[CASO 2] Hexagonal (Zircaloy) en chapa laminada (mmm)")
    grilla_hex_mmm = SO3Grid(resolucion, D6h, simetria_muestra='mmm')
    fz_2 = grilla_hex_mmm.FZ
    print(f" -> Cristal: {fz_2['Simetria_Cristal']} | Muestra: {fz_2['Simetria_Muestra']}")
    print(f" -> Rango de Euler Max: phi1={fz_2['Limites_Euler (Max)']['phi1']}° | Phi={fz_2['Limites_Euler (Max)']['Phi']}° | phi2={fz_2['Limites_Euler (Max)']['phi2']}°")
    print(f" -> Puntos (Tamaño Matriz): {fz_2['Puntos_Totales']}")

    # --- CASO 3: Cúbico en chapa laminada (Ortorrómbica) ---
    print("\n[CASO 3] Cúbico (Acero/Cobre) en chapa laminada (mmm)")
    grilla_cub_mmm = SO3Grid(resolucion, Oh, simetria_muestra='mmm')
    fz_3 = grilla_cub_mmm.FZ
    print(f" -> Cristal: {fz_3['Simetria_Cristal']} | Muestra: {fz_3['Simetria_Muestra']}")
    print(f" -> Rango de Euler Max: phi1={fz_3['Limites_Euler (Max)']['phi1']}° | Phi={fz_3['Limites_Euler (Max)']['Phi']}° | phi2={fz_3['Limites_Euler (Max)']['phi2']}°")
    print(f" -> Puntos (Tamaño Matriz): {fz_3['Puntos_Totales']}")
    print(f"\n=========================================================\n")