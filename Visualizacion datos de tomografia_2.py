# -*- coding: utf-8 -*-
"""
Visualizador de Resultados de Tomografía Tensorial de Neutrones (Multi-Material).
Entorno: texturaPy3.10
Estructurado en Bloques para uso interactivo en Spyder (# %%)

CORREGIDO: Se implementó una escala Y global (Global Y-Lim) para evitar ilusiones
ópticas de auto-scaling. Ahora las curvas bajan físicamente en los bordes.
CORREGIDO: Se incorporó la lectura del volumen de celda (v0) desde los archivos lattice 
para que la magnitud relativa entre el Al y el Cu sea estrictamente correcta en cm^-1.
CORREGIDO: Se restauró la Geometría 3D exacta y el menú en bucle interactivo de la V1.
"""
# %% [BLOQUE 1] IMPORTACIONES Y CONFIGURACIÓN
import os
import gc  # Garbage Collector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import traceback
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# Desactivar el modo interactivo global para evitar conflictos con %matplotlib qt
plt.ioff()

# Configuración de rutas
RUTA_BASE = "Resultados_Tomografia"
RUTA_ALMU_DETECTOR = os.path.join(RUTA_BASE, "Proyecciones_Almu_Detector")

# RESOLUCIÓN DE LAMBDA (Para valores por defecto, aunque ahora se lee dinámicamente del disco)
LAMBDAS_DEFAULT = np.linspace(1.0, 6.0, 500)
ANGULOS_DEFAULT = np.linspace(0, 180, 180, endpoint=False)

# Variables globales para almacenar los datos de los materiales (se cargan una sola vez)
MATERIALES_DATA = {}
MATERIALES_DF_B = {}

def detectar_materiales_disponibles():
    if not os.path.exists(RUTA_ALMU_DETECTOR):
        return []
    carpetas = [d for d in os.listdir(RUTA_ALMU_DETECTOR) if os.path.isdir(os.path.join(RUTA_ALMU_DETECTOR, d)) and d.startswith("Material_")]
    return sorted(carpetas)

def detectar_tilteos(ruta_busqueda):
    if not os.path.exists(ruta_busqueda):
        return []
    tilteos = []
    carpetas = [d for d in os.listdir(ruta_busqueda) if os.path.isdir(os.path.join(ruta_busqueda, d)) and d.startswith("Tilt_")]
    for c in carpetas:
        try:
            val = float(c.replace('Tilt_', '').replace('deg', ''))
            tilteos.append(val)
        except ValueError:
            pass
    return sorted(tilteos)

def detectar_coeficientes_disponibles(ruta_material):
    coefs = set()
    for root, _, files in os.walk(ruta_material):
        for f in files:
            if f.startswith("Proyecciones_Detector_Almu_L") and f.endswith(".npy"):
                try:
                    partes = f.replace("Proyecciones_Detector_Almu_L", "").replace(".npy", "").split("_mu")
                    l_val = int(partes[0])
                    mu_val = int(partes[1])
                    coefs.add((l_val, mu_val))
                except:
                    pass
    return sorted(list(coefs), key=lambda x: (x[0], x[1]))

def leer_v0_desde_red(filepath):
    """Extrae el volumen de la celda (v0) desde el archivo de red .txt"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        matrix_rows = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith(('%', '#')) or line.upper().startswith('BACKGROUND') or line[0].isalpha():
                continue
            parts = line.split()
            if len(parts) == 3:
                try:
                    matrix_rows.append([float(p) for p in parts])
                except ValueError:
                    pass
            if len(matrix_rows) == 3:
                break
        if len(matrix_rows) == 3:
            return np.abs(np.linalg.det(np.array(matrix_rows)))
    except:
        pass
    return 1.0  # Fallback

# %% [BLOQUE 2] CARGA DE DATOS FÍSICOS DE LOS MATERIALES
def cargar_datos_materiales():
    global MATERIALES_DF_B
    global MATERIALES_DATA
    
    if MATERIALES_DF_B:
        return
    
    carpetas_materiales = detectar_materiales_disponibles()
    if not carpetas_materiales:
        print("[AVISO] No se encontraron materiales en Proyecciones_Almu_Detector.")
        return
    
    print("Cargando propiedades físicas de los materiales (B_lmu y v0)...")
    
    mapeo_nombres = {
        'Material_1_Aluminio_Nucleo': 'Aluminio_Nucleo',
        'Material_2_Cobre_Coraza': 'Cobre_Coraza'
    }
    
    for carpeta in carpetas_materiales:
        nombre_base = mapeo_nombres.get(carpeta, carpeta.replace('Material_', '').replace('_', ''))
        ruta_csv = None
        
        posibles_rutas = [
            os.path.join('Exp_Data', nombre_base, f'B_lmu_{nombre_base}.csv'),
            os.path.join('Exp_Data', 'Al', f'B_lmu_{nombre_base}.csv'),
            os.path.join('Exp_Data', 'Cu', f'B_lmu_{nombre_base}.csv'),
            os.path.join('Exp_Data', 'Al', 'B_lmu_Aluminio_Nucleo.csv'),
            os.path.join('Exp_Data', 'Cu', 'B_lmu_Cobre_Coraza.csv')
        ]
        
        for ruta in posibles_rutas:
            if os.path.exists(ruta):
                ruta_csv = ruta
                break
                
        # Extraemos el volumen de celda v0 para normalizar la magnitud macroscópica
        v0 = 1.0
        if 'Aluminio' in carpeta or 'Al' in nombre_base:
            ruta_lat = os.path.join('Exp_Data', 'Al', 'Al_lattice.txt')
            if os.path.exists(ruta_lat): v0 = leer_v0_desde_red(ruta_lat)
        elif 'Cobre' in carpeta or 'Cu' in nombre_base:
            ruta_lat = os.path.join('Exp_Data', 'Cu', 'Cu_lattice.txt')
            if os.path.exists(ruta_lat): v0 = leer_v0_desde_red(ruta_lat)
        elif 'Zirconio' in carpeta or 'Zr' in nombre_base:
            ruta_lat = os.path.join('Exp_Data', 'Zr', 'Zr_lattice.txt')
            if os.path.exists(ruta_lat): v0 = leer_v0_desde_red(ruta_lat)
        
        if ruta_csv:
            try:
                df = pd.read_csv(ruta_csv)
                MATERIALES_DF_B[carpeta] = df
                lambdas_df = df['lambda_A'].values
                mu_total = np.zeros(len(lambdas_df))
                for i, row in df.iterrows():
                    mu_total[i] = row.get('B_0_0_Real', 0.0)
                
                MATERIALES_DATA[carpeta] = {'lambdas': lambdas_df, 'mu_total': mu_total, 'v0': v0}
                print(f"  -> Datos cargados para: {carpeta} (v0 = {v0:.2f} A^3)")
            except Exception as e:
                print(f"  -> Error al leer CSV para {carpeta}: {e}")
        else:
            print(f"  -> No se encontró archivo CSV para {carpeta}. Usando genéricos.")
            MATERIALES_DATA[carpeta] = {'lambdas': LAMBDAS_DEFAULT, 'mu_total': np.ones(len(LAMBDAS_DEFAULT)) * 0.1, 'v0': v0}
            MATERIALES_DF_B[carpeta] = pd.DataFrame({'lambda_A': LAMBDAS_DEFAULT, 'B_0_0_Real': np.ones(len(LAMBDAS_DEFAULT)) * 0.1})

# %% [BLOQUE 3] VISUALIZADOR DE PROYECCIONES 4D (ESPECTROS GLOBALES)
def visualizar_espectro_pixel_desde_4d(x, z):
    tilteos_disp = detectar_tilteos(RUTA_BASE)
    if not tilteos_disp:
        print("[ERROR] No se detectaron carpetas de Tilteo en los resultados.")
        return

    data_dict = {}
    for t in tilteos_disp:
        ruta_archivo = os.path.join(RUTA_BASE, f"Tilt_{t}deg", "Proyecciones_4D_Espectro.npy")
        if os.path.exists(ruta_archivo):
            data_dict[t] = np.load(ruta_archivo, mmap_mode='r')
            
    tilteos_disp = list(data_dict.keys())
    if not tilteos_disp:
        print("[ERROR] No se encontraron archivos de Espectro 4D.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.35) 

    t_init = tilteos_disp[0]
    idx_rot_init = 0
    n_lambdas_4d = data_dict[t_init].shape[0]
    lambdas_4d_actuales = np.linspace(1.0, 6.0, n_lambdas_4d)
    
    line, = ax.plot(lambdas_4d_actuales, data_dict[t_init][:, idx_rot_init, z, x], color='darkred', lw=2, marker='o', markersize=3)
    ax.set_xlabel(r"Longitud de Onda $\lambda$ [$\AA$]")
    ax.set_ylabel("Transmisión Total ($I/I_0$)")
    ax.grid(True, linestyle='--', alpha=0.7)

    ax_slider_rot = plt.axes([0.15, 0.15, 0.7, 0.03])
    num_angulos = data_dict[t_init].shape[1]
    slider_rot = Slider(ax_slider_rot, 'Rotación $\\theta$', 0, num_angulos - 1, valinit=idx_rot_init, valstep=1)

    ax_slider_tilt = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider_tilt = Slider(ax_slider_tilt, 'Tilteo $\\alpha$', 0, len(tilteos_disp) - 1, valinit=0, valstep=1)
    
    def update(val):
        idx_rot = int(slider_rot.val)
        idx_tilt = int(slider_tilt.val)
        t_actual = tilteos_disp[idx_tilt]
        angulo_real = ANGULOS_DEFAULT[idx_rot] if idx_rot < len(ANGULOS_DEFAULT) else idx_rot

        slider_tilt.valtext.set_text(f"{t_actual}°")
        slider_rot.valtext.set_text(f"{angulo_real:.1f}°")

        if t_actual in data_dict:
            n_lams = data_dict[t_actual].shape[0]
            lams_arr = np.linspace(1.0, 6.0, n_lams)
            line.set_xdata(lams_arr)
            line.set_ydata(data_dict[t_actual][:, idx_rot, z, x])
            ax.set_title(f"Espectro en [X={x}, Z={z}] | Tilt={t_actual}° | $\\theta \\approx {angulo_real:.1f}°$")
        else:
            line.set_ydata(np.zeros(len(lambdas_4d_actuales)))
            ax.set_title(f"FALTAN DATOS | Tilt={t_actual}°")
            
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    def on_close(event):
        data_dict.clear()  
        gc.collect()       
    fig.canvas.mpl_connect('close_event', on_close)

    slider_tilt.valtext.set_text(f"{tilteos_disp[0]}°")
    slider_rot.valtext.set_text(f"{ANGULOS_DEFAULT[0]:.1f}°")
    update(0) 
    slider_rot.on_changed(update)
    slider_tilt.on_changed(update)
    fig.slider_rot = slider_rot
    fig.slider_tilt = slider_tilt
    plt.show(block=True)

# %% [BLOQUE 4] VISUALIZADOR DE PROYECCIONES DIRECTAS A_lmu (POR MATERIAL)
def visualizar_proyecciones_Almu(carpeta_material):
    ruta_material = os.path.join(RUTA_ALMU_DETECTOR, carpeta_material)
    tilteos_disp = detectar_tilteos(RUTA_BASE)
    coefs_disp = detectar_coeficientes_disponibles(ruta_material)
    
    if not tilteos_disp: return
    if not coefs_disp: return

    coefs_str = [f"L={l}, mu={mu}" for l, mu in coefs_disp]
    paths_globales = {}
    for (l, mu), lbl in zip(coefs_disp, coefs_str):
        paths_globales[lbl] = {}
        for t in tilteos_disp:
            ruta_archivo = os.path.join(ruta_material, f"Tilt_{t}deg", f"Proyecciones_Detector_Almu_L{l}_mu{mu}.npy")
            if os.path.exists(ruta_archivo):
                paths_globales[lbl][t] = ruta_archivo

    lbl_init = coefs_str[0]
    t_init = next((t for t in tilteos_disp if t in paths_globales[lbl_init]), None)
    if t_init is None: return
        
    data_init = np.load(paths_globales[lbl_init][t_init])
    if np.iscomplexobj(data_init): data_init = np.real(data_init)
    
    num_angulos = data_init.shape[0]
    shape_img = data_init[0, :, :].shape

    fig, ax = plt.subplots(figsize=(9, 7))
    plt.subplots_adjust(bottom=0.35) 
    
    im = ax.imshow(data_init[0, :, :], cmap='coolwarm', origin='lower')
    im.set_clim(np.min(data_init), np.max(data_init))
    cbar = plt.colorbar(im, ax=ax, label=f'Amplitud Proyectada')
    
    ax_slider_rot  = plt.axes([0.15, 0.20, 0.7, 0.03])
    slider_rot     = Slider(ax_slider_rot, 'Rotación $\\theta$', 0, num_angulos - 1, valinit=0, valstep=1)

    ax_slider_tilt = plt.axes([0.15, 0.12, 0.7, 0.03])
    slider_tilt    = Slider(ax_slider_tilt, 'Tilteo $\\alpha$', 0, len(tilteos_disp) - 1, valinit=0, valstep=1)

    ax_slider_coef = plt.axes([0.15, 0.04, 0.7, 0.03])
    slider_coef    = Slider(ax_slider_coef, 'Coef. A_lmu', 0, len(coefs_str) - 1, valinit=0, valstep=1)

    estado = {'lbl': lbl_init, 'tilt': t_init, 'data': data_init}

    def update(val):
        idx_rot = int(slider_rot.val)
        idx_tilt = int(slider_tilt.val)
        idx_coef = int(slider_coef.val)
        
        lbl_new = coefs_str[idx_coef]
        t_new = tilteos_disp[idx_tilt]
        angulo_real = ANGULOS_DEFAULT[idx_rot] if idx_rot < len(ANGULOS_DEFAULT) else idx_rot
        
        slider_coef.valtext.set_text(lbl_new)
        slider_tilt.valtext.set_text(f"{t_new}°")
        slider_rot.valtext.set_text(f"{angulo_real:.1f}°")
        
        if lbl_new != estado['lbl'] or t_new != estado['tilt']:
            if t_new in paths_globales[lbl_new]:
                new_data = np.load(paths_globales[lbl_new][t_new])
                if np.iscomplexobj(new_data): new_data = np.real(new_data)
                estado['data'] = new_data
                im.set_clim(np.min(new_data), np.max(new_data))
            else:
                estado['data'] = None
            estado['lbl'] = lbl_new
            estado['tilt'] = t_new

        if estado['data'] is not None:
            im.set_data(estado['data'][idx_rot, :, :])
        else:
            im.set_data(np.zeros(shape_img))
        fig.canvas.draw_idle()

    slider_rot.on_changed(update)
    slider_tilt.on_changed(update)
    slider_coef.on_changed(update)
    
    fig.slider_rot = slider_rot
    fig.slider_tilt = slider_tilt
    fig.slider_coef = slider_coef
    update(0)
    plt.show(block=True)

# %% [BLOQUE 5] VISUALIZADOR DE GEOMETRÍA 3D
def obtener_coordenadas_materiales():
    ruta_volumen = os.path.join(RUTA_BASE, "Volumen_Materiales_IDs.npy")
    if os.path.exists(ruta_volumen):
        volumen_ids = np.load(ruta_volumen)
    else:
        print("\n  -> [AVISO] Generando geometría temporal (Core-Shell) por falta de archivo.")
        N = 50
        volumen_ids = np.zeros((N, N, N), dtype=np.int8)
        cx, cy = (N//2) - 2, N//2
        for z in range(N):
            for y in range(N):
                for x in range(N):
                    r2 = (x - cx)**2 + (y - cy)**2
                    if r2 <= (10)**2: volumen_ids[z,y,x] = 1
                    elif r2 <= (15)**2: volumen_ids[z,y,x] = 2

    nz, ny, nx = volumen_ids.shape
    coords_mat1 = np.where(volumen_ids == 1)
    coords_mat2 = np.where(volumen_ids == 2)
    
    def normalizar(c):
        z, y, x = c
        return (x - nx/2)/(nx/4), (y - ny/2)/(ny/4), (z - nz/2)/(nz/4)

    c1 = normalizar(coords_mat1) if len(coords_mat1[0]) > 0 else None
    c2 = normalizar(coords_mat2) if len(coords_mat2[0]) > 0 else None
    return c1, c2

def visualizar_geometria_3d():
    """
    Visualiza la nube de puntos 3D de forma estricta (no continua).
    Restaurada la geometría exacta de la V1 con Haz y Detector.
    """
    tilteos_disp = detectar_tilteos(RUTA_BASE)
    if not tilteos_disp:
        tilteos_disp = [0.0]
        
    angulos_disp = ANGULOS_DEFAULT

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    c1_base, c2_base = obtener_coordenadas_materiales()
    step1 = max(1, len(c1_base[0]) // 1500) if c1_base else 1
    step2 = max(1, len(c2_base[0]) // 1500) if c2_base else 1

    # --- Sliders Discretos ---
    ax_slider_rot = plt.axes([0.15, 0.15, 0.7, 0.03])
    slider_rot = Slider(ax_slider_rot, 'Rotación $\\omega$', 0, len(angulos_disp) - 1, valinit=0, valstep=1)

    ax_slider_tilt = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider_tilt = Slider(ax_slider_tilt, 'Tilteo $\\alpha$', 0, len(tilteos_disp) - 1, valinit=0, valstep=1)

    def draw_scene(omega_deg, tilt_deg):
        ax.cla() 
        omega_rad = np.radians(omega_deg)
        cos_w, sin_w = np.cos(omega_rad), np.sin(omega_rad)
        
        rad_tilt = np.radians(tilt_deg)
        cos_t, sin_t = np.cos(rad_tilt), np.sin(rad_tilt)

        def rotar_y_plotear(coords, step, color, alpha, label):
            if coords is None: return
            x_b, y_b, z_b = coords
            x, y, z = x_b[::step], y_b[::step], z_b[::step]
            
            # 1. Tilteo físico
            y_t = y * cos_t - z * sin_t
            z_t = y * sin_t + z * cos_t
            x_t = x
            
            # 2. Rotación tomográfica alrededor del Z maestro
            x_f = x_t * cos_w - y_t * sin_w
            y_f = x_t * sin_w + y_t * cos_w
            z_f = z_t
            
            ax.scatter(x_f, y_f, z_f, c=color, alpha=alpha, s=15, label=label, edgecolors='none')

        # Eje de Rotación Absoluto
        ax.plot([0, 0], [0, 0], [-4, 4], color='black', linestyle='-.', lw=2, zorder=10)

        # Muestras
        rotar_y_plotear(c1_base, step1, 'darkorange', 0.8, 'Material 1')
        rotar_y_plotear(c2_base, step2, 'steelblue', 0.3, 'Material 2')
        
        # Haz y Detector
        ax.quiver(0, 6, 0, 0, -4, 0, color='red', lw=4, arrow_length_ratio=0.2)
        xx, zz = np.meshgrid(np.linspace(-3, 3, 5), np.linspace(-3, 3, 5))
        yy = np.full_like(xx, -3)
        ax.plot_surface(xx, yy, zz, color='gold', alpha=0.3, edgecolor='orange')
        
        ax.set_xlim([-4, 4]); ax.set_ylim([-4, 6]); ax.set_zlim([-4, 4])
        ax.set_xlabel('X (Transversal)'); ax.set_ylabel('Y (Haz)'); ax.set_zlabel('Z (Goniometría)')
        ax.set_title(f'Geometría Discreta Real | Tilt = {tilt_deg:.1f}° | $\\omega$ = {omega_deg:.1f}°', fontsize=12, fontweight='bold')
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', lw=2, linestyle='-.', label='Eje Rotación'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange', markersize=8, label='Mat 1'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Mat 2'),
            Line2D([0], [0], color='red', lw=3, label='Haz'),
            Line2D([0], [0], color='gold', lw=3, label='Detector')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    def update(val):
        idx_rot = int(slider_rot.val)
        idx_tilt = int(slider_tilt.val)
        
        # Traducir índices a grados físicos reales de la simulación
        omega_deg = angulos_disp[idx_rot]
        tilt_deg = tilteos_disp[idx_tilt]
        
        slider_rot.valtext.set_text(f"{omega_deg:.1f}°")
        slider_tilt.valtext.set_text(f"{tilt_deg}°")
        
        draw_scene(omega_deg, tilt_deg)
        fig.canvas.draw_idle()

    # Init display
    slider_tilt.valtext.set_text(f"{tilteos_disp[0]}°")
    slider_rot.valtext.set_text(f"{angulos_disp[0]:.1f}°")
    draw_scene(angulos_disp[0], tilteos_disp[0])

    slider_rot.on_changed(update)
    slider_tilt.on_changed(update)
    
    fig.slider_rot = slider_rot
    fig.slider_tilt = slider_tilt
    
    plt.show(block=True)

# %% [BLOQUE 6] ESPECTRO INTERACTIVO CON SLIDERS DISCRETOS (FÍSICA REAL)
def visualizar_espectro_interactivo_con_materiales(x_inicial=35, z_inicial=35):
    """
    CÁLCULO FÍSICO ESTRICTO Y EXACTO:
    Reconstruye la atenuación elástica coherente evaluando la contracción
    tensorial macroscópica píxel por píxel para cada material, asegurando
    la escala global para evitar distorsiones ópticas en Matplotlib.
    """
    cargar_datos_materiales()
    carpetas_materiales = detectar_materiales_disponibles()
    
    tilteos_disp = detectar_tilteos(RUTA_BASE)
    if not tilteos_disp: tilteos_disp = [0.0]
        
    data_global = {}
    for t in tilteos_disp:
        ruta_archivo = os.path.join(RUTA_BASE, f"Tilt_{t}deg", "Proyecciones_4D_Espectro.npy")
        if os.path.exists(ruta_archivo):
            data_global[t] = np.load(ruta_archivo, mmap_mode='r')
    
    if not data_global:
        print("[ERROR] No se encontraron archivos de Espectro 4D.")
        return

    primer_tilt = tilteos_disp[0]
    n_lambda, n_theta, n_z, n_x = data_global[primer_tilt].shape
    lambdas_4d_reales = np.linspace(1.0, 6.0, n_lambda)
    
    # --- MAPEADO DE ARCHIVOS A_lmu ---
    print("  [INFO] Mapeando archivos A_lmu del simulador y pre-interpolando B_lmu...")
    A_lmu_data = {}
    for mat in carpetas_materiales:
        A_lmu_data[mat] = {}
        for t in tilteos_disp:
            A_lmu_data[mat][t] = {}
            ruta_tilt = os.path.join(RUTA_ALMU_DETECTOR, mat, f"Tilt_{t}deg")
            if os.path.exists(ruta_tilt):
                for f in os.listdir(ruta_tilt):
                    if f.startswith("Proyecciones_Detector_Almu_L") and f.endswith(".npy"):
                        parts = f.replace("Proyecciones_Detector_Almu_L", "").replace(".npy", "").split("_mu")
                        l_val, mu_val = int(parts[0]), int(parts[1])
                        # Usamos mmap para no sobrecargar RAM
                        A_lmu_data[mat][t][(l_val, mu_val)] = np.load(os.path.join(ruta_tilt, f), mmap_mode='r')

    # --- PRE-INTERPOLACIÓN DE B_lmu A LA GRILLA DINÁMICA CON AJUSTE v0 ---
    B_interp_dict = {}
    for mat in carpetas_materiales:
        df_B = MATERIALES_DF_B.get(mat)
        v0 = MATERIALES_DATA.get(mat, {}).get('v0', 1.0)
        
        if df_B is not None:
            lambdas_B = df_B['lambda_A'].values
            B_interp_dict[mat] = {}
            for col in df_B.columns:
                if col.startswith('B_') and col.endswith('_Real'):
                    parts = col.split('_')
                    l_val, mu_val = int(parts[1]), int(parts[2])
                    col_imag = f'B_{l_val}_{mu_val}_Imag'
                    
                    b_real = df_B[col].values
                    b_imag = df_B[col_imag].values if col_imag in df_B.columns else np.zeros_like(lambdas_B)
                    
                    # MAGIA FÍSICA: Normalización por el volumen de la celda unitaria
                    # Esto garantiza que el Cobre y el Aluminio tengan proporciones físicamente reales
                    B_complex = (b_real + 1j * b_imag) / v0
                    
                    f_real = interp1d(lambdas_B, B_complex.real, kind='linear', fill_value='extrapolate')
                    f_imag = interp1d(lambdas_B, B_complex.imag, kind='linear', fill_value='extrapolate')
                    
                    B_interp_dict[mat][(l_val, mu_val)] = f_real(lambdas_4d_reales) + 1j * f_imag(lambdas_4d_reales)

    # --- INTERFAZ GRÁFICA ---
    estado = {'x': x_inicial, 'z': z_inicial}
    cache_max_y = {} # Memoria rápida para la escala global
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    ax_detector = fig.add_subplot(gs[0, 0])
    ax_spectrum = fig.add_subplot(gs[0, 1])
    ax_elastic = fig.add_subplot(gs[1, :])
    
    ax_rot_box = plt.axes([0.15, 0.92, 0.30, 0.02])
    slider_rot = Slider(ax_rot_box, 'Rotación $\\theta$', 0, n_theta - 1, valinit=0, valstep=1)
    
    ax_tilt_box = plt.axes([0.55, 0.92, 0.30, 0.02])
    slider_tilt = Slider(ax_tilt_box, 'Tilteo $\\alpha$', 0, len(tilteos_disp) - 1, valinit=0, valstep=1)
    
    def actualizar_graficos():
        idx_rot = int(slider_rot.val)
        idx_tilt = int(slider_tilt.val)
        x, z = estado['x'], estado['z']
        t_actual = tilteos_disp[idx_tilt]
        angulo_real = ANGULOS_DEFAULT[idx_rot] if idx_rot < len(ANGULOS_DEFAULT) else idx_rot
        
        slider_rot.valtext.set_text(f"{angulo_real:.1f}°")
        slider_tilt.valtext.set_text(f"{t_actual}°")
        
        # 1. Imagen Detector
        img_prom = np.mean(data_global[t_actual][:, idx_rot, :, :], axis=0)
        ax_detector.clear()
        ax_detector.imshow(img_prom, cmap='viridis', origin='lower')
        ax_detector.plot(x, z, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax_detector.text(x+2, z, f'({x}, {z})', color='white', fontsize=10, fontweight='bold')
        ax_detector.set_title(f'Detector (Promedio $\\lambda$) | Tilt={t_actual}° | $\\theta$={angulo_real:.1f}°')
        ax_detector.set_xlabel('X (píxel)')
        ax_detector.set_ylabel('Z (píxel)')
        
        # 2. Espectro Total (Transmisión real 4D)
        espectro = data_global[t_actual][:, idx_rot, z, x]
        ax_spectrum.clear()
        ax_spectrum.plot(lambdas_4d_reales, espectro, 'b-', lw=2)
        ax_spectrum.fill_between(lambdas_4d_reales, 0, espectro, color='blue', alpha=0.2)
        ax_spectrum.set_title(f'Transmisión Total en ({x},{z})')
        ax_spectrum.set_xlabel(r'$\lambda$ [$\AA$]')
        ax_spectrum.set_ylabel('Transmisión ($I/I_0$)')
        ax_spectrum.set_ylim(-0.05, 1.05)
        ax_spectrum.grid(True, alpha=0.3)
        
        # OBTENEMOS EL MÁXIMO GLOBAL DE LA IMAGEN PARA FIJAR LA ESCALA Y (Auto-Scaling Trap Fix)
        if (idx_rot, idx_tilt) not in cache_max_y:
            espectro_imagen_completa = data_global[t_actual][:, idx_rot, :, :]
            atenuacion_imagen = -np.log(np.clip(espectro_imagen_completa, 1e-10, 1.0))
            cache_max_y[(idx_rot, idx_tilt)] = max(0.01, np.max(atenuacion_imagen) * 1.1)
        global_max_y = cache_max_y[(idx_rot, idx_tilt)]
        
        # 3. CÁLCULO FÍSICO ESTRICTO DE ATENUACIÓN COHERENTE POR MATERIAL
        ax_elastic.clear()
        colores_mat = ['r-', 'g-', 'm-', 'c-', 'y-']
        colores_fill = ['red', 'green', 'magenta', 'cyan', 'yellow']
        
        for i, mat in enumerate(carpetas_materiales):
            mu_coh_mat = np.zeros(len(lambdas_4d_reales), dtype=np.float64)
            
            # Recorremos los A_lmu guardados en el disco
            if t_actual in A_lmu_data.get(mat, {}):
                for (l, mu), A_mmap in A_lmu_data[mat][t_actual].items():
                    A_val = A_mmap[idx_rot, z, x]
                    
                    if (l, mu) in B_interp_dict.get(mat, {}):
                        B_val_array = B_interp_dict[mat][(l, mu)]
                        mu_coh_mat += np.real(A_val * B_val_array)
            
            nombre_corto = mat.replace('Material_', '')
            ax_elastic.plot(lambdas_4d_reales, mu_coh_mat, colores_mat[i % len(colores_mat)], linewidth=2.5, label=f'Coh. {nombre_corto}')
            ax_elastic.fill_between(lambdas_4d_reales, 0, mu_coh_mat, color=colores_fill[i % len(colores_fill)], alpha=0.15)
            
        ax_elastic.set_title(r'Atenuación Elástica Coherente Estricta ($\Sigma_{coh} = \sum A_{l\mu} \cdot B_{l\mu}$)')
        ax_elastic.set_xlabel(r'$\lambda$ [$\AA$]')
        ax_elastic.set_ylabel('Atenuación ($\mu_{coh}$)')
        ax_elastic.grid(True, alpha=0.3)
        ax_elastic.legend()
        
        # APLICAMOS LA ESCALA FIJA PARA EVIDENCIAR FÍSICAMENTE EL ESPESOR DEL MATERIAL
        ax_elastic.set_ylim(-0.02 * global_max_y, global_max_y)
        
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax_detector:
            x_click = int(round(event.xdata))
            z_click = int(round(event.ydata))
            if 0 <= x_click < n_x and 0 <= z_click < n_z:
                estado['x'] = x_click
                estado['z'] = z_click
                actualizar_graficos()

    def on_close(event):
        data_global.clear()
        for mat in A_lmu_data.values():
            for t_dict in mat.values():
                t_dict.clear()
        A_lmu_data.clear()
        gc.collect()        

    fig.canvas.mpl_connect('close_event', on_close)
    slider_rot.on_changed(lambda val: actualizar_graficos())
    slider_tilt.on_changed(lambda val: actualizar_graficos())
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    actualizar_graficos()
    plt.suptitle('VISUALIZADOR DE ESPECTROS INTERACTIVO (Haz clic en la imagen para seleccionar píxel)', fontsize=14, fontweight='bold')
    plt.show(block=True)

# %% [BLOQUE 7] MENÚ INTERACTIVO PRINCIPAL
def menu_interactivo():
    while True:
        print("\n" + "="*50)
        print(" VISUALIZADOR TENSORIAL INTERACTIVO")
        print("="*50)
        print("1. Ver Geometría Experimental 3D (Discreto Real)")
        print("2. Navegar Proyecciones Detector A_lmu (Discreto Real)")
        print("3. Extraer Espectro 1D de un Píxel (Sliders)")
        print("4. Visualizador de Espectros Interactivo (Desplegables + Clic)")
        print("5. Salir")
        
        try:
            opcion_str = input("\nSelecciona una opción (1-5): ").strip()
            if not opcion_str: 
                continue
                
            opcion = int(opcion_str)
            
            if opcion == 1:
                visualizar_geometria_3d()
                
            elif opcion == 2:
                materiales = detectar_materiales_disponibles()
                if not materiales:
                    print("\n[AVISO] No se encontraron carpetas de materiales en Proyecciones_Almu_Detector.")
                    continue
                
                print("\nMateriales detectados:")
                for i, mat in enumerate(materiales):
                    print(f"  [{i+1}] {mat}")
                    
                sel_mat = int(input("Selecciona el material: ") or "1")
                if 1 <= sel_mat <= len(materiales):
                    carpeta_elegida = materiales[sel_mat - 1]
                    visualizar_proyecciones_Almu(carpeta_elegida)
                else:
                    print("Selección de material no válida.")
                
            elif opcion == 3:
                x_val = input("Posición X del píxel en detector [35]: ")
                x = int(x_val) if x_val.strip() else 35
                
                z_val = input("Posición Z del píxel en detector [35]: ")
                z = int(z_val) if z_val.strip() else 35
                
                visualizar_espectro_pixel_desde_4d(x, z)
                
            elif opcion == 4:
                x_val = input("Posición X inicial del píxel en detector [35]: ")
                x = int(x_val) if x_val.strip() else 35
                
                z_val = input("Posición Z inicial del píxel en detector [35]: ")
                z = int(z_val) if z_val.strip() else 35
                
                visualizar_espectro_interactivo_con_materiales(x, z)
                
            elif opcion == 5:
                print("\nCerrando visualizador...")
                break
                
            else:
                print("Opción no válida o Salir seleccionado.")
                
        except ValueError as ve:
            if "invalid literal" in str(ve):
                print("Entrada inválida. Por favor ingresa números enteros.")
            else:
                print(f"Error procesando la gráfica: {ve}")
                traceback.print_exc()
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    menu_interactivo()
    print("\n--- VISUALIZACIÓN TERMINADA ---")