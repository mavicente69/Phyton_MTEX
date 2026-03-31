# -*- coding: utf-8 -*-
"""
Módulo de Kernels optimizado para el espacio curvo SO(3) y esfera S2.
Normalización estricta a unidades físicas MUD usando la Traza de Wigner.
Implementa de la Vallée Poussin de banda limitada.
"""
import numpy as np
from scipy.special import gammaln

class OrientationKernel:
    def __init__(self, fwhm_grados=15.0, tipo='gaussian'):
        self.fwhm_rad = np.radians(fwhm_grados)
        self.tipo = tipo.lower()
        
        # 1. PARÁMETROS DE FORMA DEL KERNEL
        if self.tipo == 'gaussian':
            self.sigma = self.fwhm_rad / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            self.l_max_norm = max(60, int(360.0 / fwhm_grados * 1.5))
            
        elif self.tipo == 'poussin' or self.tipo == 'delavalleepoussin':
            half_angle = self.fwhm_rad / 4.0
            kappa_float = np.log(0.5) / (2.0 * np.log(np.cos(half_angle)))
            # Para banda limitada estricta (estilo MTEX), kappa debe ser entero
            self.kappa = max(1, int(np.round(kappa_float)))
            self.l_max_norm = self.kappa
            
        elif self.tipo == 'lorentzian':
            self.gamma = self.fwhm_rad / (2.0 * np.sqrt(np.sqrt(2.0) - 1.0))
            self.l_max_norm = max(60, int(360.0 / fwhm_grados * 1.5))
            
        else:
            raise ValueError(f"Kernel '{tipo}' no soportado.")

        # 2. NORMALIZACIÓN FÍSICA ESTRICTA (MUD)
        K_l_norm = self._get_Kl_array(self.l_max_norm)
        L_arr = np.arange(self.l_max_norm + 1)
        
        # En la esfera 2D (Figuras de Polos), el pico escala con (2L+1) -> ~91 MUD para 20°
        self.factor_pf = np.sum((2 * L_arr + 1) * K_l_norm)
        
        # En el espacio 3D (ODF), la degeneración exige (2L+1)^2 -> ~1543 MUD para 20°
        self.factor_odf = np.sum(((2 * L_arr + 1)**2) * K_l_norm)

    def evaluate(self, omega_rad, modo='odf'):
        w = np.abs(omega_rad)
        
        if self.tipo == 'gaussian':
            exponent = np.clip(-(w**2) / (2.0 * (self.sigma**2)), -500, 0)
            val = np.exp(exponent)
            
        elif self.tipo == 'poussin' or self.tipo == 'delavalleepoussin':
            val = np.cos(np.clip(w / 2.0, 0, np.pi / 2.0))**(2.0 * self.kappa)
            
        elif self.tipo == 'lorentzian':
            val = 1.0 / (1.0 + (w / self.gamma)**2)**2
            
        factor = self.factor_odf if modo == 'odf' else self.factor_pf
        return val * factor

    def _get_Kl_array(self, L_max):
        K_l = np.zeros(L_max + 1)
        l_vals = np.arange(L_max + 1)
        
        if self.tipo == 'gaussian':
            K_l = np.exp(-0.5 * l_vals * (l_vals + 1) * (self.sigma**2))
            
        elif self.tipo == 'poussin' or self.tipo == 'delavalleepoussin':
            K_l[0] = 1.0
            for l in range(1, L_max + 1):
                if self.kappa + 1 - l > 0:
                    K_l[l] = np.exp(gammaln(self.kappa + 1) + gammaln(self.kappa + 2) - 
                                    gammaln(self.kappa + 1 - l) - gammaln(self.kappa + 2 + l))
                else:
                    K_l[l] = 0.0
                    
        elif self.tipo == 'lorentzian':
            K_l = np.exp(-l_vals * self.gamma)
            
        if K_l[0] != 0:
            K_l /= K_l[0]
            
        return K_l

    def calc_fourier_coeffs(self, L_max, return_full=False):
        K_l = self._get_Kl_array(L_max)
        
        coeficientes = []
        for l in range(L_max + 1):
            val_kl = K_l[l]
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    if m == n:
                        coeficientes.append([l, m, n, val_kl])
                    elif return_full:
                        coeficientes.append([l, m, n, 0.0])
                        
        return np.array(coeficientes)