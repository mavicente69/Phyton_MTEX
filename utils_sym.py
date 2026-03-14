# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:47:43 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Módulo auxiliar para manejo y traducción de simetrías cristalográficas.
"""

# Importamos directamente los grupos de Schoenflies desde orix
from orix.quaternion.symmetry import Oh, O, D6h, D4h, D3d, D2h, C2h, Ci, C1

def obtener_simetria(nombre_grupo):
    """
    Traduce la notación estándar (Hermann-Mauguin) al objeto de simetría de orix.
    
    Parámetros:
    - nombre_grupo (str): Nombre del grupo puntual o de Laue (ej. 'm-3m', 'mmm').
    
    Retorna:
    - Objeto orix.quaternion.symmetry correspondiente.
    """
    
    # Diccionario de traducción: Hermann-Mauguin -> Schoenflies (orix)
    diccionario_simetria = {
        # Cúbico
        'm-3m': Oh,    # Cúbica de mayor simetría (ej. fcc, bcc)
        '432': O,      # Cúbica propia
        
        # Hexagonal
        '6/mmm': D6h,  # Hexagonal (ej. Titanio, Magnesio)
        
        # Tetragonal
        '4/mmm': D4h,  # Tetragonal (ej. Martensita)
        
        # Trigonal / Romboédrico
        '-3m': D3d,    
        
        # Ortorrómbico
        'mmm': D2h,    # Ortorrómbica (típica para simetría de muestra laminada)
        
        # Monoclínico
        '2/m': C2h,    
        
        # Triclínico
        '-1': Ci,      # Triclínico con centro de inversión
        '1': C1        # Triclínico sin elementos de simetría
    }
    
    # Limpiamos el string por si hay espacios extra y lo pasamos a minúsculas
    nombre_limpio = nombre_grupo.strip().lower()
    
    if nombre_limpio in diccionario_simetria:
        return diccionario_simetria[nombre_limpio]
    else:
        claves_validas = ", ".join(diccionario_simetria.keys())
        raise ValueError(f"Error: Grupo '{nombre_grupo}' no reconocido en el traductor.\nOpciones válidas: {claves_validas}")