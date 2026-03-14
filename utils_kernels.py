# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:53:06 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Módulo de Kernels optimizado.
Normalización exacta a unidades MUD (Multiples of Uniform Distribution).
Basado en estándares de MTEX.
"""
import numpy as np
from scipy.special import gammaln  # <-- NUEVO: Necesario para los factoriales de Poussin

class OrientationKernel:
    def __init__(self, fwhm_grados=15.0, tipo='gaussian'):
        self.fwhm_rad = np.radians(fwhm_grados)
        self.tipo = tipo.lower()
        
        # 1. GAUSSIANO
        if self.tipo == 'gaussian':
            self.sigma = self.fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            # MUD Max = 4pi * (1 / 2pi*sigma^2) = 2/sigma^2
            self.factor_pf = 2.0 / (self.sigma**2)
            # MUD Max en ODF (Espacio SO3)
            self.factor_odf = (2.0 * np.sqrt(2.0 * np.pi)) / (self.sigma**3)
            
        # 2. DE LA VALLÉE POUSSIN (Exacto como MTEX)
        elif self.tipo == 'poussin':
            # k exacto tal que cos(FWHM/4)^(2k) = 0.5
            half_angle = self.fwhm_rad / 4.0
            self.kappa = np.log(0.5) / (2.0 * np.log(np.cos(half_angle)))
            # MUD Max para Poussin en PF es exactamente k + 1
            self.factor_pf = self.kappa + 1.0
            # MUD Max para Poussin en ODF (Aprox. consistente con Gaussian)
            sigma_equiv = self.fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            self.factor_odf = (2.0 * np.sqrt(2.0 * np.pi)) / (sigma_equiv**3)

        # 3. LORENTZIAN (Cauchy)
        elif self.tipo == 'lorentzian':
            # gamma ajustado para que el FWHM de la Lorentziana^2 sea el pedido
            self.gamma = self.fwhm_rad / (2.0 * np.sqrt(np.sqrt(2.0) - 1.0))
            # La Lorentziana tiene colas pesadas, su pico es naturalmente más bajo (~218 para 10°)
            self.factor_pf = 4.0 / (self.gamma**2)
            self.factor_odf = 8.0 / (self.gamma**3)
            
        else:
            raise ValueError(f"Kernel '{tipo}' no soportado.")

    def evaluate(self, omega_rad, modo='pf'):
        """
        Evalúa el kernel en unidades MUD.
        modo='pf': Para Figuras de Polos (Esfera).
        modo='odf': Para Espacio de Euler (3D).
        """
        w = np.abs(omega_rad)
        
        if self.tipo == 'gaussian':
            exponent = np.clip(-(w**2) / (2.0 * (self.sigma**2)), -500, 0)
            val = np.exp(exponent)
            
        elif self.tipo == 'poussin':
            # Función base de MTEX: cos(w/2)^(2k)
            # Solo válida hasta pi, clip por seguridad numérica
            val = np.cos(np.clip(w/2.0, 0, np.pi/2.0))**(2.0 * self.kappa)
            
        elif self.tipo == 'lorentzian':
            val = 1.0 / (1.0 + (w / self.gamma)**2)**2
            
        factor = self.factor_pf if modo == 'pf' else self.factor_odf
        return val * factor


    def calc_fourier_coeffs(self, L_max, return_full=False):
        """
        Calcula los coeficientes armónicos (Triclínico-Triclínico) del kernel en el origen.
        Retorna un array 2D estructurado con las columnas: [L, m, n, C_lmn].
        Al ser isotrópico y centrado, C_lmn = K_l solo cuando m == n.
        
        Si return_full=True, devuelve la tabla completa (incluyendo los m!=n que valen 0).
        """
        K_l = np.zeros(L_max + 1)
        l_vals = np.arange(L_max + 1)
        
        # 1. Cálculo del decaimiento K_l en Fourier
        if self.tipo == 'gaussian':
            # Decaimiento exponencial del kernel Gaussiano en el espacio de Fourier
            K_l = np.exp(-0.5 * l_vals * (l_vals + 1) * (self.sigma**2))
            
        elif self.tipo == 'poussin':
            # Coeficientes exactos para de la Vallée Poussin en SO(3)
            K_l[0] = 1.0
            for l in range(1, L_max + 1):
                if self.kappa + 1 - l > 0:
                    K_l[l] = np.exp(gammaln(self.kappa + 1) + gammaln(self.kappa + 2) - 
                                    gammaln(self.kappa + 1 - l) - gammaln(self.kappa + 2 + l))
                else:
                    K_l[l] = 0.0
                    
        elif self.tipo == 'lorentzian':
            # La distribución de Cauchy/Lorentziana tiene un decaimiento más suave
            K_l = np.exp(-l_vals * self.gamma)
            
        else:
            raise NotImplementedError(f"Coeficientes de Fourier no implementados para {self.tipo}.")
        
        # Normalizamos para asegurar que la integral de volumen = 1 (K_0 = 1)
        if K_l[0] != 0:
            K_l /= K_l[0]

        # 2. Generación de la tabla [L, m, n, C_lmn]
        coeficientes = []
        for l in range(L_max + 1):
            val_kl = K_l[l]
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    # Solo es no nulo cuando m == n
                    if m == n:
                        coeficientes.append([l, m, n, val_kl])
                    elif return_full:
                        coeficientes.append([l, m, n, 0.0])
                        
        return np.array(coeficientes)