# -*- coding: utf-8 -*-
"""
Test de Aislamiento Numérico: Discreta vs Fourier vs Componente Analítica Pura
Configurado para resolución de 2.0° y simetría de muestra C1.
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from orix.vector import Miller
from orix.quaternion import Orientation
from orix.quaternion.symmetry import D2h, C1
from diffpy.structure import Lattice, Structure

from utils_so3 import SO3Grid
from utils_odf import ODFDiscreta, ODFFourier, ODFBunge, ODFComponent, plot_odf_sections_comparison
from utils_pf import plot_pf_comparison
from utils_kernels import OrientationKernel 
from scipy.spatial.transform import Rotation as R_sci

def main():
    print("=========================================================")
    print("🧪 INICIANDO TEST CONCORDANCIA: COMPONENTE VS DISCRETA VS FOURIER")
    print("=========================================================")

    # 1. Configurar red FCC y Simetrías
    lat = Lattice(4.77, 4.77, 4.77, 90, 90, 90)
    fase_fcc = Phase(name="Test_FCC", space_group=225, structure=Structure(lattice=lat))
    simetria_muestra = C1

    # 2. Generar espacio y definir orientación central (Polo Cúbico)
    print(" -> Generando grilla SO(3) y configurando Kernel...")
    grilla = SO3Grid(resolucion_grados=5.0, simetria_cristal=fase_fcc.point_group, simetria_muestra=simetria_muestra)
    orientaciones = grilla.orientaciones
    
    ori_centro = Orientation.from_euler(np.radians([0, 10, 0]), symmetry=fase_fcc.point_group)
    
    # 3. Usamos tu Kernel real
    ancho_grados = 15.0 
    mi_kernel = OrientationKernel(fwhm_grados=ancho_grados, tipo='gaussian') 

    # 4. INSTANCIAR ODF COMPONENTE PRIMERO
    print(" -> Instanciando ODF Componente (Analítica Pura)...")
    odf_comp = ODFComponent(
        orientaciones=ori_centro, 
        pesos=[1.0], 
        kernels=[mi_kernel], 
        crystal_sym=fase_fcc.point_group, 
        sample_sym=simetria_muestra
    )

    # 5. EXTRACCIÓN DE PESOS SIMETRIZADOS
    print(" -> Extrayendo pesos simetrizados para la grilla...")
    pesos_reales_mud = odf_comp.evaluate(orientaciones)

    # 6. INSTANCIAR ODF DISCRETA
    print(" -> Instanciando ODF Discreta (Regresión Lineal Local)...")
    odf_disc = ODFDiscreta(
        pesos=pesos_reales_mud, 
        orientaciones=orientaciones, 
        crystal_sym=fase_fcc.point_group, 
        sample_sym=simetria_muestra
    )

    # 7. Extracción de bases de Fourier
    L_MAX = 30
    
    print(f" -> Extrayendo Base Simetrizada mn (L_max={L_MAX})...")
    coefs_sym = odf_disc.calc_fourier_coeffs(L_MAX)
    
    print(f"\n 🔥 VERIFICACIÓN: El coeficiente C(0,0,0) (Integral) vale: {coefs_sym[0][3]:.4f}\n")
    
    odf_four = ODFFourier(coefs_sym, crystal_sym=fase_fcc.point_group, sample_sym=simetria_muestra)

    print(f" -> Extrayendo Base Irreducible de Bunge mu,nu (L_max={L_MAX})...")
    coefs_bunge = odf_disc.calc_bunge_coeffs(L_MAX)
    odf_bung = ODFBunge(coefs_bunge, crystal_sym=fase_fcc.point_group, sample_sym=simetria_muestra)

    # =========================================================
    # PRUEBA 1: CORTE 1D EXACTO SOBRE EL POLO
    # =========================================================
    print(" -> Evaluando perfil 1D a lo largo de phi2...")
    phi2_rango = np.linspace(0, 90, 200)
    eulers_linea = np.column_stack((np.zeros_like(phi2_rango), np.zeros_like(phi2_rango), phi2_rango))
    
    y_comp = odf_comp.get_density(eulers_linea, degrees=True)
    y_disc = odf_disc.get_density(eulers_linea, degrees=True)
    y_four = odf_four.get_density(eulers_linea, degrees=True)
    y_bung = odf_bung.get_density(eulers_linea, degrees=True)

    plt.figure(figsize=(10, 6))
    
    plt.plot(phi2_rango, y_comp, 'g-', lw=8, alpha=0.3, label='ODF Componente (Verdad Analítica Pura)')
    plt.plot(phi2_rango, y_disc, 'k-', lw=2, label='ODF Discreta (Regresión Lineal Local)')
    
    euler_grados = np.degrees(odf_disc.euler_clean)
    tol = 0.1
    
    mask1 = (np.abs(euler_grados[:, 0] % 360) < tol) & (np.abs(euler_grados[:, 1] % 360) < tol)
    mask1 &= (euler_grados[:, 2] >= -tol) & (euler_grados[:, 2] <= 90 + tol)
    nodos_x1 = euler_grados[mask1, 2]
    nodos_y1 = odf_disc.pesos_clean[mask1]
    
    plt.plot(nodos_x1, nodos_y1, 'ko', markersize=7, zorder=5, label='Nodos Reales de la Grilla')
    plt.plot(phi2_rango, y_four, 'r--', lw=2, label=f'Fourier Simetrizado mn (L={L_MAX})')
    plt.plot(phi2_rango, y_bung, 'b:', lw=3, label=f'Fourier Bunge $\\mu\\nu$ (L={L_MAX})')
    
    plt.title(f"Perfil 1D sobre la Componente Cúbica (Corte en $\\varphi_1=0^\circ, \\Phi=0^\circ$)", fontsize=14, fontweight='bold')
    plt.xlabel("Ángulo $\\varphi_2$ (grados)", fontsize=12)
    plt.ylabel("Intensidad de Orientación (f(g) en MUD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 90)
    plt.tight_layout()
    plt.show(block=False)

    # =========================================================
    # PRUEBA 1.B: CORTE 1D SOBRE LA LÍNEA (0, Phi, 0)
    # =========================================================
    print(" -> Evaluando perfil 1D a lo largo de Phi (0 a 90°)...")
    Phi_rango = np.linspace(0, 90, 200)
    eulers_linea_phi = np.column_stack((np.zeros_like(Phi_rango), Phi_rango, np.zeros_like(Phi_rango)))
    
    y_comp_phi = odf_comp.get_density(eulers_linea_phi, degrees=True)
    y_disc_phi = odf_disc.get_density(eulers_linea_phi, degrees=True)
    y_four_phi = odf_four.get_density(eulers_linea_phi, degrees=True)
    y_bung_phi = odf_bung.get_density(eulers_linea_phi, degrees=True)

    plt.figure(figsize=(10, 6))
    
    plt.plot(Phi_rango, y_comp_phi, 'g-', lw=8, alpha=0.3, label='ODF Componente (Verdad Analítica Pura)')
    plt.plot(Phi_rango, y_disc_phi, 'k-', lw=2, label='ODF Discreta (Regresión Lineal Local)')
    
    mask2 = (np.abs(euler_grados[:, 0] % 360) < tol) & (np.abs(euler_grados[:, 2] % 360) < tol)
    mask2 &= (euler_grados[:, 1] >= -tol) & (euler_grados[:, 1] <= 90 + tol)
    nodos_x2 = euler_grados[mask2, 1]
    nodos_y2 = odf_disc.pesos_clean[mask2]
    
    plt.plot(nodos_x2, nodos_y2, 'ko', markersize=7, zorder=5, label='Nodos Reales de la Grilla')
    plt.plot(Phi_rango, y_four_phi, 'r--', lw=2, label=f'Fourier Simetrizado mn (L={L_MAX})')
    plt.plot(Phi_rango, y_bung_phi, 'b:', lw=3, label=f'Fourier Bunge $\\mu\\nu$ (L={L_MAX})')
    
    plt.title(f"Perfil 1D sobre la línea $\\varphi_1=0^\circ, \\varphi_2=0^\circ$ (Cúbica a Goss)", fontsize=14, fontweight='bold')
    plt.xlabel("Ángulo $\\Phi$ (grados)", fontsize=12)
    plt.ylabel("Intensidad de Orientación (f(g) en MUD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 90)
    plt.tight_layout()
    plt.show(block=False)

    # =========================================================
    # PRUEBA 1.C: CORTE 1D SOBRE LA LÍNEA (90, Phi, 0)
    # =========================================================
    print(" -> Evaluando perfil 1D a lo largo de Phi (0 a 90°) con phi1=90°...")
    Phi_rango_90 = np.linspace(0, 90, 200)
    eulers_linea_phi1_90 = np.column_stack((np.full_like(Phi_rango_90, 90.0), Phi_rango_90, np.zeros_like(Phi_rango_90)))
    
    y_comp_phi1_90 = odf_comp.get_density(eulers_linea_phi1_90, degrees=True)
    y_disc_phi1_90 = odf_disc.get_density(eulers_linea_phi1_90, degrees=True)
    y_four_phi1_90 = odf_four.get_density(eulers_linea_phi1_90, degrees=True)
    y_bung_phi1_90 = odf_bung.get_density(eulers_linea_phi1_90, degrees=True)

    plt.figure(figsize=(10, 6))
    
    plt.plot(Phi_rango_90, y_comp_phi1_90, 'g-', lw=8, alpha=0.3, label='ODF Componente (Verdad Analítica Pura)')
    plt.plot(Phi_rango_90, y_disc_phi1_90, 'k-', lw=2, label='ODF Discreta (Regresión Lineal Local)')
    
    mask3 = (np.abs(euler_grados[:, 0] - 90.0) < tol) & (np.abs(euler_grados[:, 2] % 360) < tol)
    mask3 &= (euler_grados[:, 1] >= -tol) & (euler_grados[:, 1] <= 90 + tol)
    nodos_x3 = euler_grados[mask3, 1]
    nodos_y3 = odf_disc.pesos_clean[mask3]
    
    plt.plot(nodos_x3, nodos_y3, 'ko', markersize=7, zorder=5, label='Nodos Reales de la Grilla')
    plt.plot(Phi_rango_90, y_four_phi1_90, 'r--', lw=2, label=f'Fourier Simetrizado mn (L={L_MAX})')
    plt.plot(Phi_rango_90, y_bung_phi1_90, 'b:', lw=3, label=f'Fourier Bunge $\\mu\\nu$ (L={L_MAX})')
    
    plt.title(f"Perfil 1D sobre la línea $\\varphi_1=90^\circ, \\varphi_2=0^\circ$", fontsize=14, fontweight='bold')
    plt.xlabel("Ángulo $\\Phi$ (grados)", fontsize=12)
    plt.ylabel("Intensidad de Orientación (f(g) en MUD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 90)
    plt.tight_layout()
    plt.show(block=False)

    # =========================================================
    # PRUEBA 2: COMPARATIVA VISUAL DE FIGURAS DE POLOS
    # =========================================================
    print(" -> Proyectando Figuras de Polos...")
    planos = [Miller(hkl=[1, 1, 1], phase=fase_fcc), Miller(hkl=[2, 0, 0], phase=fase_fcc)]
    
    pf_comp = odf_comp.calc_pole_figures(planos, res_grados=2.5)
    pf_disc = odf_disc.calc_pole_figures(planos, res_grados=2.5)
    pf_four = odf_four.calc_pole_figures(planos, res_grados=2.5)

    plot_pf_comparison(pfs_in=pf_comp, pfs_out=pf_disc, titulos=["{111}", "{200}"], suptitle="Componente Pura vs Discreta Local")
    plot_pf_comparison(pfs_in=pf_disc, pfs_out=pf_four, titulos=["{111}", "{200}"], suptitle="Discreta Local vs Fourier Simetrizado")

    # =========================================================
    # PRUEBA 3: COMPARATIVA VISUAL DE SECCIONES ODF
    # =========================================================
    print(" -> Generando Secciones 2D de la ODF...")
    plot_odf_sections_comparison(odf_comp, odf_disc, "Componente", "Discreta", "Secciones: Componente vs Discreta")
    plot_odf_sections_comparison(odf_disc, odf_four, "Discreta", "Fourier", "Secciones: Discreta vs Fourier")

    # =========================================================
    # PRUEBA 4: INTEGRAL DE FIBRAS (RECORRIDO DE 2 POLOS DE LA MISMA PF {111})
    # =========================================================
    print(" -> Evaluando fibras de integración 1D para 2 polos de la PF {111}...")
    
    def generar_fibra_integracion(h_vec, y_vec, num_pts=360):
        """Genera la trayectoria (fibra en Euler) que integra un algoritmo de Figuras de Polos"""
        h_vec = h_vec / np.linalg.norm(h_vec)
        y_vec = y_vec / np.linalg.norm(y_vec)
        z_axis = np.array([0.0, 0.0, 1.0])
        
        def get_R_to_z(v):
            ax = np.cross(z_axis, v)
            if np.linalg.norm(ax) < 1e-6:
                return np.eye(3) if np.dot(z_axis, v) > 0 else np.diag([1, -1, -1])
            ax = ax / np.linalg.norm(ax)
            ang = np.arccos(np.clip(np.dot(z_axis, v), -1.0, 1.0))
            return R_sci.from_rotvec(ax * ang).as_matrix()
            
        R2 = get_R_to_z(h_vec)
        R1 = get_R_to_z(y_vec)
        
        omegas = np.linspace(0, 360, num_pts)
        eulers = []
        for w in omegas:
            Rw = R_sci.from_euler('Z', w, degrees=True).as_matrix()
            g_mat = R1 @ Rw @ R2.T # Transformación Cristal a Muestra
            e = R_sci.from_matrix(g_mat).as_euler('ZXZ', degrees=True)
            eulers.append(e)
        
        eulers = np.array(eulers)
        eulers[:, 0] %= 360; eulers[:, 1] %= 360; eulers[:, 2] %= 360
        return omegas, eulers

    # Tomamos la orientación central y proyectamos dos variantes de la familia {111}
    g0_mat = R_sci.from_euler('ZXZ', [0, 10, 0], degrees=True).as_matrix()
    h_cristal_1 = np.array([1.0, 1.0, 1.0])
    h_cristal_2 = np.array([-1.0, 1.0, 1.0]) # Otra variante del MISMO plano {111}
    
    y_muestra_1 = g0_mat @ (h_cristal_1 / np.linalg.norm(h_cristal_1))
    y_muestra_2 = g0_mat @ (h_cristal_2 / np.linalg.norm(h_cristal_2))
    
    # Generamos las rutas de integración para ambos polos
    omegas_1, fibra_1 = generar_fibra_integracion(h_cristal_1, y_muestra_1)
    omegas_2, fibra_2 = generar_fibra_integracion(h_cristal_2, y_muestra_2)
    
    # Evaluamos ambas ODFs (Analítica y Discreta) en esas dos rutas
    dens_comp_1 = odf_comp.get_density(fibra_1, degrees=True)
    dens_disc_1 = odf_disc.get_density(fibra_1, degrees=True)
    
    dens_comp_2 = odf_comp.get_density(fibra_2, degrees=True)
    dens_disc_2 = odf_disc.get_density(fibra_2, degrees=True)
    
    plt.figure(figsize=(10, 6))
    
    # Componente Analítica (Líneas gruesas)
    # Notarás que el área bajo estas dos curvas es idéntica en la analítica (polos de misma intensidad)
    plt.plot(omegas_1, dens_comp_1, 'g-', lw=8, alpha=0.3, label='Analítica (Polo 1 de {111})')
    plt.plot(omegas_2, dens_comp_2, 'b-', lw=8, alpha=0.3, label='Analítica (Polo 2 de {111})')
    
    # Discreta Local (Líneas finas punteadas)
    # Acá verás cómo la regresión "recorta" de forma distinta a cada polo
    plt.plot(omegas_1, dens_disc_1, 'g--', lw=2, label='Discreta (Polo 1 de {111})')
    plt.plot(omegas_2, dens_disc_2, 'b--', lw=2, label='Discreta (Polo 2 de {111})')
    
    plt.title("Ruta de Integración: ¿Por qué dos polos del plano {111} difieren en intensidad?", fontsize=14, fontweight='bold')
    plt.xlabel("Ángulo de avance $\\omega$ a lo largo de la fibra (grados)", fontsize=12)
    plt.ylabel("Densidad cruzada por la fibra (MUD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.5)
    plt.xlim(0, 360)
    plt.tight_layout()

    print("\n✅ TEST FINALIZADO.")
    plt.show()

if __name__ == "__main__":
    main()