# -*- coding: utf-8 -*-
"""
Motor de Armónicos Esféricos y Matrices de Wigner (Ultra Optimizado con Numba).
Calcula coeficientes, proyectores de simetría y simula ODFs/PFs (Base Wigner y Base Bunge).
Entorno: texturaPy3.10
"""
import numpy as np
import scipy.special as sp
from numba import njit

# =========================================================================================
# OPTIMIZACIÓN NUMBA: PRE-CÁLCULO DE FACTORIALES
# =========================================================================================
_FACT = np.array([1.0] + [np.prod(np.arange(1.0, i + 1.0)) for i in range(1, 150)], dtype=np.float64)

@njit(cache=True)
def _wigner_small_d_numba(l, m, n, beta_array):
    k_min = max(0, n - m)
    k_max = min(l - m, l + n)
    
    d_val = np.zeros_like(beta_array, dtype=np.float64)
    cos_b = np.cos(beta_array / 2.0)
    sin_b = np.sin(beta_array / 2.0)
    
    prefactor = np.sqrt(_FACT[l + m] * _FACT[l - m] * _FACT[l + n] * _FACT[l - n])
    
    for i in range(len(beta_array)):
        cb = cos_b[i]
        sb = sin_b[i]
        suma = 0.0
        for k in range(k_min, k_max + 1):
            denominador = _FACT[l - m - k] * _FACT[l + n - k] * _FACT[k + m - n] * _FACT[k]
            signo = 1.0 if (k - m + n) % 2 == 0 else -1.0
            pot_cos = 2 * l - m + n - 2 * k
            pot_sin = 2 * k + m - n
            
            termino = signo * (1.0 / denominador) * (cb**pot_cos) * (sb**pot_sin)
            suma += termino
        d_val[i] = prefactor * suma
        
    return d_val

def _wigner_small_d(l, m, n, beta):
    beta = np.asarray(beta, dtype=np.float64)
    return _wigner_small_d_numba(l, m, n, beta)

# =========================================================================================
# OPERADORES CLÁSICOS Y FOURIER TRICLÍNICO (ACELERADOS POR NUMBA)
# =========================================================================================

@njit(cache=True)
def _calc_component_coefs_numba(L_max, K_l, alpha, beta, gamma):
    max_size = 0
    for l in range(L_max + 1): max_size += (2 * l + 1)**2
    out = np.zeros((max_size, 6), dtype=np.float64)
    idx = 0
    beta_arr = np.array([beta], dtype=np.float64)
    
    for l in range(L_max + 1):
        kl = K_l[l]
        if abs(kl) < 1e-12: continue
        prefactor = kl * (2.0 * l + 1.0)
        
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                d_mn = _wigner_small_d_numba(l, m, n, beta_arr)[0]
                fase = m * gamma + n * alpha
                c_real = prefactor * d_mn * np.cos(fase)
                c_imag = prefactor * d_mn * np.sin(fase)
                mod = np.sqrt(c_real**2 + c_imag**2)
                
                if mod > 1e-10:
                    out[idx, 0] = l
                    out[idx, 1] = m
                    out[idx, 2] = n
                    out[idx, 3] = c_real
                    out[idx, 4] = c_imag
                    out[idx, 5] = mod
                    idx += 1
    return out[:idx]

def calc_component_fourier(kernel, orientacion, L_max):
    tabla_kernel = kernel.calc_fourier_coeffs(L_max, return_full=False)
    K_l = np.zeros(L_max + 1)
    for fila in tabla_kernel:
        if int(fila[1]) == int(fila[2]): K_l[int(fila[0])] = fila[3]

    euler_angles = orientacion.to_euler().flatten()
    alpha = euler_angles[0] - (np.pi / 2.0)
    beta  = euler_angles[1]
    gamma = euler_angles[2] + (np.pi / 2.0)
    
    return _calc_component_coefs_numba(L_max, K_l, alpha, beta, gamma)

@njit(cache=True)
def _calc_discrete_coefs_numba(L_max, alpha, beta, gamma, pesos):
    max_size = 0
    for l in range(L_max + 1): max_size += (2 * l + 1)**2
    out = np.zeros((max_size, 6), dtype=np.float64)
    idx = 0
    N = len(beta)
    
    for l in range(L_max + 1):
        prefactor_l = 2.0 * l + 1.0
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                sum_real = 0.0
                sum_imag = 0.0
                d_mn_array = _wigner_small_d_numba(l, m, n, beta)
                
                for i in range(N):
                    fase = m * gamma[i] + n * alpha[i]
                    val = pesos[i] * d_mn_array[i] * prefactor_l
                    sum_real += val * np.cos(fase)
                    sum_imag += val * np.sin(fase)
                    
                mod = np.sqrt(sum_real**2 + sum_imag**2)
                if mod > 1e-10:
                    out[idx, 0] = l
                    out[idx, 1] = m
                    out[idx, 2] = n
                    out[idx, 3] = sum_real
                    out[idx, 4] = sum_imag
                    out[idx, 5] = mod
                    idx += 1
    return out[:idx]

def calc_discrete_fourier(orientaciones, pesos, L_max):
    """Calcula coeficientes triclínicos para una grilla discreta de orientaciones (ej. WIMV)."""
    euler = orientaciones.to_euler().reshape(-1, 3)
    alpha = euler[:, 0] - (np.pi / 2.0)
    beta  = euler[:, 1]
    gamma = euler[:, 2] + (np.pi / 2.0)
    
    pesos_arr = np.asarray(pesos, dtype=np.float64)
    if np.sum(pesos_arr) > 0: pesos_arr /= np.sum(pesos_arr)
        
    # Hemos removido el shift experimental de aquí. La matemática base se mantiene pura.
    return _calc_discrete_coefs_numba(L_max, alpha, beta, gamma, pesos_arr)

@njit(cache=True)
def _build_Tl_matrix_numba(l, alpha, beta, gamma):
    N = len(beta)
    T = np.zeros((N, 2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for m in range(-l, l + 1):
        for n in range(-l, l + 1):
            d_mn = _wigner_small_d_numba(l, m, n, beta)
            for i in range(N):
                fase_ang = -(m * gamma[i] + n * alpha[i])
                T[i, m + l, n + l] = d_mn[i] * np.exp(1j * fase_ang)
    return T

def build_Tl_matrix(l, orientaciones):
    euler = orientaciones.to_euler().reshape(-1, 3)
    alpha = euler[:, 0] - (np.pi / 2.0)
    beta  = euler[:, 1]
    gamma = euler[:, 2] + (np.pi / 2.0)
    return _build_Tl_matrix_numba(l, alpha, beta, gamma)

def calc_symmetry_projectors(L_max, crystal_sym, sample_sym=None):
    pg_c = crystal_sym.point_group if hasattr(crystal_sym, 'point_group') else crystal_sym
    proj_C, proj_S = {}, {}
    for l in range(L_max + 1):
        T_cryst = build_Tl_matrix(l, pg_c)
        proj_C[l] = np.mean(np.conj(T_cryst), axis=0)
        
        if sample_sym is not None:
            T_samp = build_Tl_matrix(l, sample_sym)
            proj_S[l] = np.mean(np.conj(T_samp), axis=0)
        else:
            proj_S[l] = np.eye(2 * l + 1, dtype=complex)
    return proj_C, proj_S

def symmetrize_coefs(coefs_tri_array, L_max, proj_C, proj_S):
    C_sym_list = []
    for l in range(L_max + 1):
        C_matrix = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
        mask = coefs_tri_array[:, 0] == l
        for row in coefs_tri_array[mask]:
            m, n = int(row[1]), int(row[2])
            C_matrix[m + l, n + l] = row[3] + 1j * row[4]
            
        C_matrix_sym = proj_C[l] @ C_matrix @ proj_S[l]
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                val = C_matrix_sym[m + l, n + l]
                if abs(val) > 1e-10:
                    C_sym_list.append([l, m, n, val.real, val.imag, abs(val)])
                    
    if not C_sym_list: return np.zeros((0, 6))
    arr = np.array(C_sym_list)
    return arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))] 

def eval_odf_from_fourier(coefs_array, orientaciones):
    orig_shape = orientaciones.shape
    euler = orientaciones.to_euler().reshape(-1, 3)
    alpha = euler[:, 0] - (np.pi / 2.0)
    beta  = euler[:, 1]
    gamma = euler[:, 2] + (np.pi / 2.0)
    
    densidad = np.zeros(len(beta), dtype=complex)
    for fila in coefs_array:
        l, m, n = int(fila[0]), int(fila[1]), int(fila[2])
        c_val = fila[3] + 1j * fila[4]
        d_mn = _wigner_small_d(l, m, n, beta)
        fase = np.exp(-1j * (m * gamma + n * alpha))
        densidad += c_val * d_mn * fase
        
    return np.maximum(np.real(densidad), 0.0).reshape(orig_shape)

def sum_fourier_arrays(arrays_list):
    C_sum = {}
    for arr in arrays_list:
        for r in arr:
            l, m, n = int(r[0]), int(r[1]), int(r[2])
            C_sum[(l, m, n)] = C_sum.get((l, m, n), 0.0j) + (r[3] + 1j * r[4])
            
    data = [[l, m, n, v.real, v.imag, abs(v)] for (l, m, n), v in C_sum.items() if abs(v) > 1e-10]
    if not data: return np.zeros((0, 6))
    arr = np.array(data)
    return arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]

# =========================================================================================
# BASE IRREDUCIBLE DE BUNGE (Índices Mu y Nu)
# =========================================================================================

def calc_symmetry_coefficients(L_max, proj_C, proj_S):
    A_cryst, B_samp = {}, {}
    for l in range(L_max + 1):
        evals_c, evecs_c = np.linalg.eigh(proj_C[l])
        A_cryst[l] = evecs_c[:, evals_c > 0.99] 
        evals_s, evecs_s = np.linalg.eigh(proj_S[l])
        B_samp[l] = evecs_s[:, evals_s > 0.99] 
    return A_cryst, B_samp

def get_bunge_coefs(coefs_tri_array, L_max, A_cryst, B_samp):
    C_bunge = {}
    for l in range(L_max + 1):
        M_l, N_l = A_cryst[l].shape[1], B_samp[l].shape[1]
        if M_l == 0 or N_l == 0: continue
        C_matrix_tri = np.zeros((2 * l + 1, 2 * l + 1), dtype=complex)
        mask = coefs_tri_array[:, 0] == l
        for row in coefs_tri_array[mask]:
            m, n = int(row[1]), int(row[2])
            C_matrix_tri[m + l, n + l] = row[3] + 1j * row[4]
        C_red = np.conj(A_cryst[l]).T @ C_matrix_tri @ B_samp[l]
        for mu in range(M_l):
            for nu in range(N_l):
                if abs(C_red[mu, nu]) > 1e-10:
                    C_bunge[(l, mu, nu)] = C_red[mu, nu]
    return C_bunge

def eval_odf_from_bunge(C_bunge, orientaciones, A_cryst, B_samp):
    L_max = max([k[0] for k in C_bunge.keys()]) if C_bunge else 0
    orig_shape = orientaciones.shape
    euler = orientaciones.to_euler().reshape(-1, 3)
    alpha = euler[:, 0] - (np.pi / 2.0)
    beta  = euler[:, 1]
    gamma = euler[:, 2] + (np.pi / 2.0)
    N_oris = len(beta)
    densidad = np.zeros(N_oris, dtype=complex)
    
    for l in range(L_max + 1):
        if l not in A_cryst or l not in B_samp: continue
        M_l, N_l = A_cryst[l].shape[1], B_samp[l].shape[1]
        if M_l == 0 or N_l == 0: continue
        C_l_red = np.zeros((M_l, N_l), dtype=complex)
        has_coefs = False
        for mu in range(M_l):
            for nu in range(N_l):
                val = C_bunge.get((l, mu, nu), 0.0j)
                if abs(val) > 1e-10:
                    C_l_red[mu, nu] = val
                    has_coefs = True
        if not has_coefs: continue

        m_activos = np.where(np.sum(np.abs(A_cryst[l]), axis=1) > 1e-8)[0] - l
        n_activos = np.where(np.sum(np.abs(B_samp[l]), axis=1) > 1e-8)[0] - l
        T_tri_esparso = {}
        for m in m_activos:
            for n in n_activos:
                d_mn = _wigner_small_d(l, m, n, beta)
                fase = np.exp(-1j * (m * gamma + n * alpha))
                T_tri_esparso[(m, n)] = d_mn * fase

        for mu in range(M_l):
            for nu in range(N_l):
                c_val = C_l_red[mu, nu]
                if abs(c_val) < 1e-10: continue
                T_sym_munu = np.zeros(N_oris, dtype=complex)
                for m in m_activos:
                    A_val = A_cryst[l][m+l, mu]
                    if abs(A_val) < 1e-10: continue
                    for n in n_activos:
                        B_val = np.conj(B_samp[l][n+l, nu])
                        if abs(B_val) < 1e-10: continue
                        T_sym_munu += A_val * T_tri_esparso[(m, n)] * B_val
                densidad += c_val * T_sym_munu
    return np.maximum(np.real(densidad), 0.0).reshape(orig_shape)

# =========================================================================================
# SÍNTESIS DE FIGURAS DE POLOS: EL CEREBRO DE LA PROYECCIÓN
# =========================================================================================

def eval_sph_harm(m, l, polar, azimuthal):
    try:
        from scipy.special import sph_harm_y
        return sph_harm_y(l, m, polar, azimuthal)
    except ImportError:
        from scipy.special import sph_harm
        return sph_harm(m, l, azimuthal, polar)

def eval_sym_sph_harm(l, polar, azimuthal, proj_matrix, is_sample=False):
    polar, azimuthal = np.atleast_1d(polar), np.atleast_1d(azimuthal)
    Y_raw = np.zeros((2 * l + 1, len(polar)), dtype=complex)
    for m in range(-l, l + 1):
        val = eval_sph_harm(m, l, polar, azimuthal)
        if is_sample:
            val = np.conj(val) * (-1.0)**m
        Y_raw[m + l, :] = val
    return np.conj(proj_matrix).T @ Y_raw

def eval_pf_from_wigner(coefs_array, pole_polar, pole_azim, y_polar, y_azim):
    intensities = np.zeros(len(y_polar), dtype=complex)
    
    Y_y_cache = {}
    for fila in coefs_array:
        l, n = int(fila[0]), int(fila[2])
        if l % 2 == 0 and (l, n) not in Y_y_cache:
            Y_y_cache[(l, n)] = ((-1.0)**n) * np.conj(eval_sph_harm(n, l, y_polar, y_azim))
            
    for fila in coefs_array:
        l, m, n = int(fila[0]), int(fila[1]), int(fila[2])
        if l % 2 != 0: continue 
        
        Y_h = eval_sph_harm(m, l, pole_polar, pole_azim)
        
        c_val = fila[3] + 1j * fila[4]
        intensities += c_val * (4.0 * np.pi / (2 * l + 1)) * np.conj(Y_h) * Y_y_cache[(l, n)]

    return np.maximum(np.real(intensities), 0.0)

def eval_pf_from_bunge(C_bunge, pole_polar, pole_azim, y_polar, y_azim, A_cryst, B_samp):
    L_max = max([k[0] for k in C_bunge.keys()]) if C_bunge else 0
    pf_intensities = np.ones(len(y_polar), dtype=complex) 
    
    for l in range(2, L_max + 1, 2): 
        if l not in A_cryst or l not in B_samp: continue
        M_l, N_l = A_cryst[l].shape[1], B_samp[l].shape[1]
        if M_l == 0 or N_l == 0: continue
            
        C_l = np.zeros((M_l, N_l), dtype=complex)
        for mu in range(M_l):
            for nu in range(N_l): 
                C_l[mu, nu] = C_bunge.get((l, mu, nu), 0.0j)

        Y_c_mu = eval_sym_sph_harm(l, pole_polar, pole_azim, A_cryst[l], is_sample=False)
        Y_s_nu = eval_sym_sph_harm(l, y_polar, y_azim, B_samp[l], is_sample=True)
        
        term_l = (np.conj(Y_c_mu).T @ C_l @ Y_s_nu).flatten()
        pf_intensities += (4.0 * np.pi / (2*l + 1)) * term_l
        
    return np.maximum(np.real(pf_intensities), 0.0)