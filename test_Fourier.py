# -*- coding: utf-8 -*-
"""
Prueba de Validación Integral de Fourier contra MTEX.
Genera una ODF, extrae coeficientes, reconstruye en 3D, evalúa el truncamiento 1D
y proyecta analíticamente Figuras de Polos usando armónicos esféricos.
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Orientation, symmetry
from orix.vector import Miller
from orix.crystal_map import Phase  # <-- IMPORTANTE: Necesario para los índices de Miller
from utils_kernels import OrientationKernel
from utils_odf import ODFComponent, ODFFourier
from utils_fourier import eval_odf_from_fourier
from utils_pf import PoleFigure, plot_pfs

def test_fourier_integral():
    print("=========================================================")
    print("🧪 TEST DE VALIDACIÓN INTEGRAL: FOURIER, ODF Y POLE FIGURES")
    print("=========================================================")

    # 1. Definir Simetrías (Hexagonal)
    cs = symmetry.D6 
    ss = None            

    # 2. Definir la Orientación Central
    phi1, Phi, phi2 = 10.0, 30.0, 0.0
    print(f" -> Orientación central : Euler ({phi1}°, {Phi}°, {phi2}°)")
    ori = Orientation.from_euler(np.radians([[phi1, Phi, phi2]]), symmetry=cs)

    # 3. Definir el Kernel
    FWHM = 30.0
    TIPO_KERNEL = 'poussin'  # Kernel de banda limitada ideal para Fourier
    print(f" -> Kernel              : {TIPO_KERNEL.capitalize()} (FWHM = {FWHM}°)")
    kernel = OrientationKernel(tipo=TIPO_KERNEL, fwhm_grados=FWHM)

    # 4. Construir la ODF Sintética Original
    odf_sintetica = ODFComponent(
        orientaciones=ori, pesos=[1.0], kernels=kernel,
        crystal_sym=cs, sample_sym=ss
    )

    # 5. Calcular los Coeficientes de Fourier
    L_MAX = 20  # L_max para la prueba de truncamiento
    print(f"\n -> Calculando expansión de Fourier hasta L_max = {L_MAX}...")
    coefs = odf_sintetica.calc_fourier_coeffs(L_MAX)

    # 6. Imprimir Tabla de Verificación (hasta L=4)
    PRINT_L_MAX = 4
    print(f"\n -> Mostrando coeficientes hasta L = {PRINT_L_MAX}:")
    print("  L |  m |  n |      Real      |    Imaginario  ")
    print("--------------------------------------------------")
    for l in range(0, PRINT_L_MAX + 1, 2):
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                val = coefs.get((l, m, n), 0.0j)
                if abs(val) > 1e-5:
                    print(f" {l:2d} | {m:2d} | {n:2d} | {val.real:12.6f} | {val.imag:12.6f}")
    print("--------------------------------------------------\n")

    # 7. RECONSTRUCCIÓN CONTINUA (Analítica desde Fourier)
    print(" -> Instanciando textura analítica a partir de Fourier...")
    odf_reconstruida = ODFFourier(
        coefs=coefs,
        crystal_sym=cs,
        sample_sym=ss
    )

    # 8. PERFIL 1D EXACTO (Para análisis del error por truncamiento)
    print(" -> Evaluando perfil 1D a lo largo de Phi (0° a 90°)...")
    Phi_grados = np.linspace(0, 90, 300)
    eulers_linea = np.radians(np.column_stack((np.zeros_like(Phi_grados), Phi_grados, np.zeros_like(Phi_grados))))
    oris_linea = Orientation.from_euler(eulers_linea, symmetry=cs)

    perfil_original = odf_sintetica.evaluate(oris_linea)
    perfil_fourier = eval_odf_from_fourier(coefs, oris_linea)

    max_orig = np.max(perfil_original)
    max_four = np.max(perfil_fourier)
    print(f"\n -> Máximo Perfil Original : {max_orig:.2f} MUD")
    print(f" -> Máximo Perfil Fourier  : {max_four:.2f} MUD")
    print(f" -> Diferencia por corte   : {abs(max_orig - max_four):.2f} MUD\n")

    # 9. GRAFICAR ODFs
    print("=========================================================")
    print("📊 COMPARACIÓN GRÁFICA (CERRÁ CADA VENTANA PARA AVANZAR)")
    print("=========================================================")
    
    print(" [1/4] Graficando ODF Sintética Original...")
    odf_sintetica.plot_sections(sections=[0, 15, 30, 45, 60], res_grados=2.5)
    
    print(" [2/4] Graficando ODF Reconstruida por Fourier (Analítica)...")
    odf_reconstruida.plot_sections(sections=[0, 15, 30, 45, 60], res_grados=2.5)
    
    print(" [3/4] Graficando Perfil 1D de Truncamiento...")
    plt.figure(figsize=(10, 6))
    plt.plot(Phi_grados, perfil_original, 'b-', linewidth=3, label=f'Original ({TIPO_KERNEL.capitalize()})')
    plt.plot(Phi_grados, perfil_fourier, 'r--', linewidth=2.5, label=f'Fourier Reconstruido (L_max={L_MAX})')
    
    plt.axvline(0.0, color='gray', linestyle=':', label='Centro de Componente (0°)')
    plt.title(f"Perfil 1D de Textura - Análisis de Truncamiento de Serie (L_max={L_MAX})", fontsize=14)
    plt.xlabel("Ángulo $\Phi$ (grados)", fontsize=12)
    plt.ylabel("Intensidad ODF (MUD)", fontsize=12)
    plt.xlim(0, 60)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 10. FIGURAS DE POLOS ANALÍTICAS
    print("\n=========================================================")
    print("🌍 [4/4] GENERANDO FIGURAS DE POLOS DESDE FOURIER")
    print("=========================================================")
    
    # Configuramos la fase Hexagonal para que la clase Miller pueda construir los vectores
    fase_hex = Phase(name="Hexagonal", space_group=194)
    
    # Inicializamos los planos pasándole los 4 índices y la fase
    basal = Miller(hkil=[0, 0, 0, 1], phase=fase_hex)
    prisma = Miller(hkil=[1, 0, -1, 0], phase=fase_hex)

    pf_basal = PoleFigure.from_fourier(coefs, basal, cs, resolution=2.5)
    pf_prisma = PoleFigure.from_fourier(coefs, prisma, cs, resolution=2.5)

    plot_pfs([pf_basal, pf_prisma], titulos=["PF (0001) - Analítica", "PF (10-10) - Analítica"])
    print("=========================================================\n")
    print("✅ TEST COMPLETADO CON ÉXITO.")

if __name__ == "__main__":
    test_fourier_integral()