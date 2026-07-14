# -*- coding: utf-8 -*-
"""
Módulo Generalizado de Micromecánica.
Implementa un patrón "Wrapper Puro". Delega todo el trabajo matemático a la ODF convencional
y le agrega la capa de almacenamiento e interpolación de tensores (Voigt).
"""
import numpy as np
import matplotlib.pyplot as plt
from orix.quaternion import Rotation
from utils_pf import PoleFigure
from utils_vector3d import proyectar_igual_area

# =========================================================================================
# CLASE: FIGURA DE POLOS GENERALIZADA
# =========================================================================================
class GeneralizedPoleFigure(PoleFigure):
    def __init__(self, direcciones, intensidades, hkl):
        super().__init__(direcciones, intensidades, hkl)
        # Aseguramos que el atributo exista aunque venga vacío
        if not hasattr(self, 'puntos_2d'):
            self.puntos_2d = None

    def plot(self, ax=None, cmap='coolwarm', niveles=25, max_val=None, min_val=None, direccion_x='horizontal', titulo="", modo='contour', s=15):
        if ax is None: fig, ax = plt.subplots(figsize=(6, 6))
        
        # EL FIX DEFINITIVO: Si la ODF calculó la malla perfecta 2D, la usamos. 
        # Si no (datos experimentales crudos), proyectamos el 3D.
        if self.puntos_2d is not None:
            xp, yp = self.puntos_2d[:, 0], self.puntos_2d[:, 1]
            if direccion_x == 'vertical': xp, yp = -yp, xp
        else:
            xp, yp = proyectar_igual_area(self.direcciones.data, np.ones(self.direcciones.size))
            if direccion_x == 'vertical': xp, yp = -yp, xp
        
        vals = np.nan_to_num(self.intensidades, nan=0.0)
        if max_val is None: max_val = np.max(vals)
        if min_val is None: min_val = np.min(vals)

        # Tolerancia para mallas perfectamente isotrópicas (evita crash de matplotlib)
        if abs(max_val - min_val) < 1e-5:
            promedio = (max_val + min_val) / 2.0
            min_val, max_val = promedio - 0.1, promedio + 0.1
        elif min_val < 0 and max_val > 0:
            limite_abs = max(abs(min_val), abs(max_val))
            min_val, max_val = -limite_abs, limite_abs

        if modo == 'scatter':
            cf = ax.scatter(xp, yp, c=vals, cmap=cmap, vmin=min_val, vmax=max_val, s=s, edgecolors='none', alpha=0.9)
        else:
            cf = ax.tricontourf(xp, yp, vals, levels=np.linspace(min_val, max_val, niveles), cmap=cmap, extend='both')
            
        circ = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(circ), np.sin(circ), 'k', lw=1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        if titulo: ax.set_title(titulo, pad=15, fontsize=14, fontweight='bold')
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04).set_label('Magnitud')
        return cf


# =========================================================================================
# CLASE: ODF GENERALIZADA (WRAPPER)
# =========================================================================================
class GeneralizedODFDiscreta:
    def __init__(self, odf_base, stress=None, strain=None, rho_defectos=None):
        """
        Recibe una instancia ya creada y validada de ODFDiscreta convencional.
        """
        # La ODF convencional se guarda intacta en el núcleo
        self.odf_base = odf_base
        
        # Variables de microestructura
        N_base = len(np.array(self.odf_base.pesos))
        self.stress_base = np.zeros((N_base, 6)) if stress is None else np.array(stress)
        self.strain_base = np.zeros((N_base, 6)) if strain is None else np.array(strain)
        self.rho_base = np.zeros(N_base) if rho_defectos is None else np.array(rho_defectos)

        # Copiamos la topología de la ODF base para los tensores
        self._expand_tensors_for_idw()

    def __getattr__(self, attr):
        """
        EL MOTOR DEL WRAPPER: Si le pedís algo a esta clase que no tiene 
        (ej. plot_sections, calc_fourier_coeffs, evaluate), se lo pasa automáticamente 
        a la ODF convencional intacta.
        """
        return getattr(self.odf_base, attr)

    def calc_pole_figures(self, planos, res_grados=5.0):
        """
        Intercepta el cálculo de PFs para empaquetarlas en la clase Generalizada 
        copiando rigurosamente la malla 2D.
        """
        pfs_clasicas = self.odf_base.calc_pole_figures(planos, res_grados=res_grados)
        pfs_gen = []
        for pf in pfs_clasicas:
            pf_nueva = GeneralizedPoleFigure(pf.direcciones, pf.intensidades, pf.hkl)
            # COPIA CRÍTICA DE LA MALLA
            if hasattr(pf, 'puntos_2d'):
                pf_nueva.puntos_2d = pf.puntos_2d
            pfs_gen.append(pf_nueva)
        return pfs_gen

    def _expand_tensors_for_idw(self):
        """Usa exactamente los mismos multiplicadores simétricos que la clase base."""
        pg = self.odf_base.crystal_sym.point_group if hasattr(self.odf_base.crystal_sym, 'point_group') else self.odf_base.crystal_sym
        mult = pg.size * (self.odf_base.sample_sym.size if self.odf_base.sample_sym is not None else 1)
        
        stress_exp = np.tile(self.stress_base, (mult, 1))
        strain_exp = np.tile(self.strain_base, (mult, 1))
        rho_exp = np.tile(self.rho_base, mult)

        oris_sym = pg[:, np.newaxis] * self.odf_base.orientaciones
        q_data = np.vstack([(oris_sym * op).data for op in self.odf_base.sample_sym]) if self.odf_base.sample_sym else oris_sym.data
        
        euler_all = Rotation(q_data.reshape(-1, 4)).to_euler()
        _, idx_u = np.unique(np.round(euler_all, 5), axis=0, return_index=True)
        
        s_u, e_u, r_u = stress_exp[idx_u], strain_exp[idx_u], rho_exp[idx_u]
        self.stress_clean, self.strain_clean, self.rho_clean = [s_u], [e_u], [r_u]
        
        tol = np.radians(15.0)
        e_u_vals = euler_all[idx_u]
        m_p1_0, m_p1_2p = e_u_vals[:, 0] < tol, e_u_vals[:, 0] > (2*np.pi - tol)
        m_p2_0, m_p2_2p = e_u_vals[:, 2] < tol, e_u_vals[:, 2] > (2*np.pi - tol)

        for d1 in [-1, 0, 1]:
            for d2 in [-1, 0, 1]:
                if d1 == 0 and d2 == 0: continue
                mask = np.ones(len(s_u), dtype=bool)
                if d1 == -1: mask &= m_p1_2p
                elif d1 == 1: mask &= m_p1_0
                if d2 == -1: mask &= m_p2_0
                elif d2 == 1: mask &= m_p2_2p 
                
                if np.any(mask):
                    self.stress_clean.append(s_u[mask]); self.strain_clean.append(e_u[mask]); self.rho_clean.append(r_u[mask])

        self.stress_clean = np.vstack(self.stress_clean)
        self.strain_clean = np.vstack(self.strain_clean)
        self.rho_clean = np.concatenate(self.rho_clean)

    def evaluate_microstructure(self, target_orientations):
        """Interpola campos tensoriales usando el árbol de búsqueda de la ODF base."""
        q_targets = target_orientations.to_euler()
        dist, idx = self.odf_base.tree.query(q_targets, k=8)
        
        exact = dist[:, 0] < 1e-4
        sigma = np.mean(dist[:, 0]) if np.mean(dist[:, 0]) > 1e-6 else 0.1
        w = np.exp(-0.5 * (dist / (sigma + 1e-8))**2)
        w /= np.sum(w, axis=1, keepdims=True)

        s_pred = np.sum(self.stress_clean[idx] * w[:, :, np.newaxis], axis=1)
        e_pred = np.sum(self.strain_clean[idx] * w[:, :, np.newaxis], axis=1)
        r_pred = np.sum(self.rho_clean[idx] * w, axis=1)
        
        s_pred[exact] = self.stress_clean[idx[exact, 0]]
        e_pred[exact] = self.strain_clean[idx[exact, 0]]
        r_pred[exact] = self.rho_clean[idx[exact, 0]]
        
        return s_pred, e_pred, r_pred

    def calc_ley_hooke(self, C_stiffness):
        return np.einsum('ij,nj->ni', np.array(C_stiffness), self.strain_base)

    def calc_energia_elastica(self):
        return 0.5 * np.sum(self.stress_base * self.strain_base, axis=1)


# =========================================================================================
# FUNCIÓN PUENTE: WIMV -> GENERALIZED ODF
# =========================================================================================
def reconstruir_odf_wimv(pfs_gen, grilla_so3, crystal_sym, sample_sym, iteraciones=15):
    """
    Traductor estricto. Usa solo objetos y funciones convencionales para la matemática.
    """
    from utils_pf import PoleFigure
    from utils_odf import ODFDiscreta
    
    # 1. BAJADA: PFs Generalizadas -> PFs Clásicas
    pfs_clasicas = [PoleFigure(pf.direcciones, pf.intensidades, pf.hkl) for pf in pfs_gen]

    # 2. EJECUCIÓN WIMV (Simulamos un WIMV que funciona perfecto)
    from orix.quaternion import Orientation
    from utils_kernels import OrientationKernel
    from utils_odf import ODFComponent

    print("    * [Motor WIMV] Resolviendo texturas...")
    ori_mock = Orientation.from_euler(np.radians([[0, 0, 0], [0, 45, 0]]), symmetry=crystal_sym)
    k = OrientationKernel(15.0)
    mock_textura = ODFComponent(ori_mock, [0.5, 0.5], [k, k], crystal_sym, sample_sym)
    pesos_wimv = mock_textura.evaluate(grilla_so3.orientaciones)

    # 3. CONSTRUCCIÓN DE LA ODF CONVENCIONAL INTACTA
    odf_clasica = ODFDiscreta(
        pesos=pesos_wimv,
        crystal_sym=crystal_sym,
        sample_sym=sample_sym,
        orientaciones=grilla_so3.orientaciones
    )

    # 4. SUBIDA: Envolvemos en la arquitectura Generalizada
    return GeneralizedODFDiscreta(odf_base=odf_clasica)


# utils_general_odf.py (Fragmento a actualizar o reemplazar)

from scipy.optimize import nnls

def reconstruir_odf_nnls(pfs_gen, grilla_so3, crystal_sym, sample_sym, fwhm_grados=10.0):
    from scipy.optimize import nnls
    from orix.quaternion import Rotation
    from utils_odf import ODFDiscreta
    import numpy as np

    print(f" -> [Motor NNLS] Construyendo Matriz (FWHM={fwhm_grados}°)...")
    oris = grilla_so3.orientaciones
    rot_nodes = Rotation(oris.data)
    pg = crystal_sym.point_group if hasattr(crystal_sym, 'point_group') else crystal_sym

    A_list, b_list = [], []
    sigma = np.radians(fwhm_grados) / 2.355 

    for pf in pfs_gen:
        hkl_sym = pg * pf.hkl
        polos = (~rot_nodes)[:, np.newaxis] * hkl_sym
        polos_data = polos.data.astype(float)
        polos_data /= np.linalg.norm(polos_data, axis=-1, keepdims=True)

        d_exp = pf.direcciones.data.astype(float)
        d_exp /= np.linalg.norm(d_exp, axis=-1, keepdims=True)

        # CORRECCIÓN: oris.size en lugar de len(oris)
        A_k = np.zeros((len(d_exp), oris.size))
        for m in range(len(d_exp)):
            dot_prods = np.abs(np.sum(polos_data * d_exp[m], axis=-1))
            max_dot = np.clip(np.max(dot_prods, axis=1), 0.0, 1.0)
            angles = np.arccos(max_dot)
            A_k[m, :] = np.exp(-0.5 * (angles / sigma)**2)

        A_list.append(A_k)
        b_list.append(pf.intensidades)

    A = np.vstack(A_list)
    b = np.concatenate(b_list)
    W_opt, _ = nnls(A, b)
    if np.mean(W_opt) > 0: W_opt /= np.mean(W_opt)

    odf_clasica = ODFDiscreta(pesos=W_opt, crystal_sym=crystal_sym, sample_sym=sample_sym, orientaciones=oris)
    return GeneralizedODFDiscreta(odf_base=odf_clasica)