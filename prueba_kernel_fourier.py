# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:33:51 2026

@author: mavic
"""

# -*- coding: utf-8 -*-
"""
Prueba 1D de Kernels y Transformada de Fourier.
Evalúa el perfil puro de la campana y su reconstrucción armónica.
"""
import numpy as np
import matplotlib.pyplot as plt
from utils_kernels import OrientationKernel

def test_kernel_1d():
    # --- CONFIGURACIÓN DE LA PRUEBA ---
    FWHM = 20.0
    L_MAX = 30
    TIPO_KERNEL = 'gaussian'  # Podés probar con 'delavalleepoussin' si ya lo tenés armado
    
    print("=========================================================")
    print("🔬 TEST 1D DE RECONSTRUCCIÓN DE KERNEL")
    print("=========================================================")
    print(f" -> Tipo de Kernel : {TIPO_KERNEL.capitalize()}")
    print(f" -> FWHM           : {FWHM}°")
    print(f" -> L_max Fourier  : {L_MAX}")

    # 1. Instanciar el Kernel
    kernel = OrientationKernel(tipo=TIPO_KERNEL, fwhm_grados=FWHM)

    # 2. Extraer los coeficientes K_l del kernel
    tabla_kernel = kernel.calc_fourier_coeffs(L_MAX, return_full=False)
    K_l = np.zeros(L_MAX + 1)
    for fila in tabla_kernel:
        l, m, n, val = int(fila[0]), int(fila[1]), int(fila[2]), fila[3]
        if m == n:
            K_l[l] = val.real if isinstance(val, complex) else val

    # 3. Crear el dominio 1D (Ángulo de desorientación omega)
    omegas_deg = np.linspace(0, 180, 500)
    omegas_rad = np.radians(omegas_deg)

    # 4. Evaluar el Kernel EXACTO en el Espacio Real
    K_exacto = kernel.evaluate(omegas_rad, modo='odf')

    # 5. Reconstruir el Kernel desde los coeficientes de FOURIER
    K_fourier = np.zeros_like(omegas_rad)
    for i, w in enumerate(omegas_rad):
        val_fourier = 0.0
        for l in range(L_MAX + 1):
            # Caracter de la representación SO(3) (Traza de la matriz Wigner-D)
            if w < 1e-8:
                chi_l = (2 * l + 1)
            else:
                chi_l = np.sin((l + 0.5) * w) / np.sin(w / 2.0)

            # Sumatoria de Bunge: K_l * (2l+1) * chi_l
            val_fourier += K_l[l] * (2 * l + 1) * chi_l

        K_fourier[i] = val_fourier

    # 6. Resultados Numéricos
    max_exacto = K_exacto[0]
    max_fourier = K_fourier[0]
    error_abs = abs(max_exacto - max_fourier)
    
    print("\n -> VALORES EN EL PICO (omega = 0°):")
    print(f"    * Exacto (Espacio Real)  : {max_exacto:.4f} MUD")
    print(f"    * Fourier (Reconstruido) : {max_fourier:.4f} MUD")
    print(f"    * Diferencia absoluta    : {error_abs:.4f} MUD")
    print("=========================================================\n")

    # 7. Gráfico Comparativo
    plt.figure(figsize=(10, 6))
    plt.plot(omegas_deg, K_exacto, 'b-', linewidth=3, label='Exacto (Espacio Real)')
    plt.plot(omegas_deg, K_fourier, 'r--', linewidth=2, label=f'Fourier (L_max={L_MAX})')

    plt.title(f"Perfil 1D del Kernel ({TIPO_KERNEL.capitalize()}) - FWHM = {FWHM}°")
    plt.xlabel("Ángulo de desorientación $\omega$ (grados)")
    plt.ylabel("Intensidad (MUD)")
    plt.xlim(0, max(FWHM * 2.5, 40))  # Zoom a la campana
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_kernel_1d()