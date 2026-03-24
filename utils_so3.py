# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 13:08:17 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Módulo para la generación de grillas de orientaciones en SO(3).
Topología avanzada equiespaciada para evitar el amontonamiento polar de Euler.
Entorno: texturaPy3.10
"""
import numpy as np
from orix.sampling import get_sample_fundamental
from orix.quaternion import Orientation

class SO3Grid:
    """
    Generador de grillas uniformes en el espacio de orientaciones SO(3).
    Soporta la reducción del espacio mediante la intersección de la 
    Zona Fundamental del Cristal y de la Muestra.
    """
    
    def __init__(self, resolucion_grados, simetria_cristal, simetria_muestra=None, metodo='cubochoric'):
        """
        Inicializa la grilla SO(3) en la verdadera Zona Fundamental (FZ).
        """
        self.resolucion = resolucion_grados
        self.simetria_cristal = simetria_cristal
        self.simetria_muestra = simetria_muestra
        
        print(f"\n[SO3Grid] Generando grilla SO(3) con resolución {self.resolucion}°...")
        
        # 1. Muestreo de la FZ del cristal (Orix nativo)
        rotaciones = get_sample_fundamental(
            resolution=self.resolucion, 
            point_group=self.simetria_cristal,
            method=metodo
        )
        oris_cristal = Orientation(rotaciones.data, symmetry=self.simetria_cristal)
        
        # 2. Recorte por la FZ de la muestra (Intersección G_cristal x G_muestra)
        self.limite_phi1 = 360.0
        if self.simetria_muestra is not None:
            # Detectamos si es ortorrómbica (chapa laminada) o monoclínica
            nom_muestra = getattr(self.simetria_muestra, 'name', str(self.simetria_muestra))
            if nom_muestra in ['222', 'mmm', 'D2h', 'm-m-m', 'orthorhombic']:
                self.limite_phi1 = 90.0
            elif nom_muestra in ['2/m', '2', 'm', 'monoclinic']:
                self.limite_phi1 = 180.0
                
            # Filtramos la grilla (le damos una pequeñísima tolerancia para no perder bordes por redondeo)
            angulos_euler = oris_cristal.to_euler(degrees=True)
            mascara_adentro_fz = angulos_euler[:, 0] <= (self.limite_phi1 + 1e-3)
            self.orientaciones = oris_cristal[mascara_adentro_fz]
            
            print(f"[SO3Grid] Simetría de muestra detectada ({nom_muestra}).")
            print(f"[SO3Grid] Zona Fundamental reducida: phi1 restringido a [0°, {self.limite_phi1}°].")
        else:
            self.orientaciones = oris_cristal
            print(f"[SO3Grid] Sin simetría de muestra. FZ completa en phi1 [0°, 360°].")

        self.N = self.orientaciones.size
        print(f"[SO3Grid] ¡Listo! {self.N} orientaciones conforman la FZ de trabajo.")
        
    @property
    def FZ(self):
        """
        Permite extraer la información geométrica de la Zona Fundamental activa.
        Devuelve un diccionario con los límites en grados.
        """
        return {
            'simetria_cristal': getattr(self.simetria_cristal, 'name', str(self.simetria_cristal)),
            'simetria_muestra': getattr(self.simetria_muestra, 'name', str(self.simetria_muestra)),
            'limites_euler': {
                'phi1_max': self.limite_phi1,
                # Phi y phi2 máximos dependen de la simetría cristalina
                'Phi_max': np.degrees(np.max(self.to_euler(degrees=False)[:, 1])),
                'phi2_max': np.degrees(np.max(self.to_euler(degrees=False)[:, 2]))
            },
            'puntos_totales': self.N
        }

    def to_euler(self, degrees=True):
        """Devuelve la matriz Nx3 con los ángulos de Euler de la grilla."""
        return self.orientaciones.to_euler(degrees=degrees)

    # --- Métodos Mágicos ---
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.orientaciones[idx]