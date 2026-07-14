# -*- coding: utf-8 -*-
"""
Visualizador de Resultados de Tomografía Tensorial de Neutrones (Multi-Material).
Entorno: texturaPy3.10
Estructurado en Bloques para uso interactivo en Spyder (# %%)

MODIFICACIÓN: Interfaz de Ventana Única con Doble Slider (Rotación y Tilteo).
Los sliders operan en pasos discretos atados a los datos reales simulados.
Selección interactiva de Coeficientes A_lmu con RadioButtons.
CORREGIDO: La barra de Tilteo ahora detecta y barre TODOS los ángulos simulados (0, 15, 30, 45, etc.).
"""
# %% [BLOQUE 1] IMPORTACIONES Y CONFIGURACIÓN
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from mpl_toolkits.mplot3d import Axes3D

# Configuración de rutas
RUTA_BASE = "Resultados_Tomografia"
RUTA_ALMU_DETECTOR = os.path.join(RUTA_BASE, "Proyecciones_Almu_Detector")

# Parámetros de reconstrucción (Fallback)
LAMBDAS_DEFAULT = np.linspace(1.0, 6.0, 50)
ANGULOS_DEFAULT = np.linspace(0, 180, 180, endpoint=False)

def detectar_materiales_disponibles():
    """Escanea el directorio de resultados para listar los materiales exportados."""
    if not os.path.exists(RUTA_ALMU_DETECTOR):
        return []
    carpetas = [d for d in os.listdir(RUTA_ALMU_DETECTOR) if os.path.isdir(os.path.join(RUTA_ALMU_DETECTOR, d)) and d.startswith("Material_")]
    return sorted(carpetas)

def detectar_tilteos(ruta_busqueda):
    """
    Busca carpetas con el patrón 'Tilt_Xdeg' y extrae los ángulos disponibles.
    Retorna una lista ordenada de floats.
    """
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
    """
    Busca recursivamente todos los archivos 'Proyecciones_Detector_Almu_L{l}_mu{mu}.npy'
    en la carpeta del material para listar los coeficientes generados.
    """
    coefs = set()
    for root, _, files in os.walk(ruta_material):
        for f in files:
            if f.startswith("Proyecciones_Detector_Almu_L") and f.endswith(".npy"):
                try:
                    # Extraer 'l' y 'mu' del nombre del archivo
                    partes = f.replace("Proyecciones_Detector_Almu_L", "").replace(".npy", "").split("_mu")
                    l_val = int(partes[0])
                    mu_val = int(partes[1])
                    coefs.add((l_val, mu_val))
                except:
                    pass
    # Ordenar por L y luego por mu
    return sorted(list(coefs))

# %% [BLOQUE 2] VISUALIZADOR DE PROYECCIONES 4D (ESPECTROS GLOBALES)
def visualizar_espectro_pixel_desde_4d(x, z):
    """Grafica el espectro de un píxel navegable por Rotación y Tilteo (Discreto)."""
    tilteos_disp = detectar_tilteos(RUTA_BASE)
    if not tilteos_disp:
        print("[ERROR] No se detectaron carpetas de Tilteo en los resultados.")
        return

    # Cargamos todos los volúmenes 4D en memoria mapeada
    data_dict = {}
    for t in tilteos_disp:
        ruta_archivo = os.path.join(RUTA_BASE, f"Tilt_{t}deg", "Proyecciones_4D_Espectro.npy")
        if os.path.exists(ruta_archivo):
            data_dict[t] = np.load(ruta_archivo, mmap_mode='r')
            
    tilteos_disp = list(data_dict.keys())
    if not tilteos_disp:
        print("[ERROR] No se encontraron archivos de Espectro 4D.")
        return

    # --- Configuración Gráfica ---
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.35) 

    t_init = tilteos_disp[0]
    idx_rot_init = 0
    
    line, = ax.plot(LAMBDAS_DEFAULT, data_dict[t_init][:, idx_rot_init, z, x], color='darkred', lw=2, marker='o', markersize=3)
    ax.set_xlabel(r"Longitud de Onda $\lambda$ [$\AA$]")
    ax.set_ylabel("Transmisión Total ($I/I_0$)")
    ax.grid(True, linestyle='--', alpha=0.7)

    # --- Sliders Discretos (Por Índices) ---
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

        # Forzamos el texto del slider para mostrar los grados reales simulados
        slider_tilt.valtext.set_text(f"{t_actual}°")
        slider_rot.valtext.set_text(f"{angulo_real:.1f}°")

        # Actualizamos curva y auto-escalamos
        line.set_ydata(data_dict[t_actual][:, idx_rot, z, x])
        ax.relim()
        ax.autoscale_view()

        ax.set_title(f"Espectro en [X={x}, Z={z}] | Tilt={t_actual}° | $\\theta \\approx {angulo_real:.1f}°$")
        fig.canvas.draw_idle()

    # Init display
    slider_tilt.valtext.set_text(f"{tilteos_disp[0]}°")
    slider_rot.valtext.set_text(f"{ANGULOS_DEFAULT[0]:.1f}°")
    update(0) 
    
    slider_rot.on_changed(update)
    slider_tilt.on_changed(update)
    
    fig.slider_rot = slider_rot
    fig.slider_tilt = slider_tilt

    plt.show(block=True)

# %% [BLOQUE 3] VISUALIZADOR DE PROYECCIONES DIRECTAS A_lmu (POR MATERIAL) - CORREGIDO
def visualizar_proyecciones_Almu(carpeta_material):
    """
    Carga los sinogramas directos del coeficiente A_lmu.
    Navegación estricta por índices de los ángulos simulados.
    Selección de coeficiente integrada en la UI.
    CORREGIDO: Detecta correctamente TODOS los ángulos de tilteo disponibles
    en la estructura de carpetas generada por el simulador.
    """
    ruta_material = os.path.join(RUTA_ALMU_DETECTOR, carpeta_material)
    
    # CORRECCIÓN: Buscar TODAS las carpetas Tilt_Xdeg dentro del material
    if not os.path.exists(ruta_material):
        print(f"[ERROR] No existe la carpeta: {ruta_material}")
        return

    # Listar todas las subcarpetas que empiezan con "Tilt_"
    tilteos_raw = [d for d in os.listdir(ruta_material) if os.path.isdir(os.path.join(ruta_material, d)) and d.startswith("Tilt_")]
    
    tilteos_disp = []
    for t in tilteos_raw:
        try:
            val = float(t.replace('Tilt_', '').replace('deg', ''))
            tilteos_disp.append(val)
        except ValueError:
            pass
    tilteos_disp = sorted(tilteos_disp)
    
    # Detectar coeficientes disponibles
    coefs_disp = detectar_coeficientes_disponibles(ruta_material)
    
    if not tilteos_disp:
        print(f"[ERROR] No se detectaron carpetas de Tilteo en {carpeta_material}.")
        return
    if not coefs_disp:
        print(f"[ERROR] No se encontraron coeficientes en {carpeta_material}.")
        return

    # Estructura para almacenar todos los datos cargados en memoria
    datos_globales = {}
    coefs_str = []
    
    print("Cargando datos en memoria...")
    for (l, mu) in coefs_disp:
        lbl = f"L={l}, mu={mu}"
        coefs_str.append(lbl)
        datos_globales[lbl] = {}
        for t in tilteos_disp:
            # CORRECCIÓN: Construir la ruta con la subcarpeta Tilt_ correcta
            ruta_archivo = os.path.join(ruta_material, f"Tilt_{t}deg", f"Proyecciones_Detector_Almu_L{l}_mu{mu}.npy")
            if os.path.exists(ruta_archivo):
                data = np.load(ruta_archivo)
                if np.iscomplexobj(data): 
                    data = np.real(data)
                datos_globales[lbl][t] = data

    coef_actual_lbl = coefs_str[0]
    
    # Validar que el coeficiente actual tenga datos
    if not datos_globales[coef_actual_lbl]:
        print("[ERROR] Falta datos para el primer coeficiente.")
        return
        
    t_init = tilteos_disp[0]
    num_angulos = datos_globales[coef_actual_lbl][t_init].shape[0]

    # --- Configuración Gráfica ---
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(left=0.25, bottom=0.30) 
    
    # Calcular límites de color para el coeficiente inicial
    vmin_global = min([np.min(d) for d in datos_globales[coef_actual_lbl].values()])
    vmax_global = max([np.max(d) for d in datos_globales[coef_actual_lbl].values()])
    
    im = ax.imshow(datos_globales[coef_actual_lbl][t_init][0, :, :], cmap='coolwarm', origin='lower', vmin=vmin_global, vmax=vmax_global)
    cbar = plt.colorbar(im, ax=ax, label=f'Amplitud Proyectada')
    ax.set_xlabel("Posición X (Píxel)")
    ax.set_ylabel("Posición Z (Píxel)")

    # --- Elementos de UI ---
    # Sliders
    ax_slider_rot = plt.axes([0.3, 0.15, 0.55, 0.03])
    slider_rot = Slider(ax_slider_rot, 'Rotación $\\theta$', 0, num_angulos - 1, valinit=0, valstep=1)

    ax_slider_tilt = plt.axes([0.3, 0.05, 0.55, 0.03])
    slider_tilt = Slider(ax_slider_tilt, 'Tilteo $\\alpha$', 0, len(tilteos_disp) - 1, valinit=0, valstep=1)

    # Botones Radiales para Coeficientes
    ax_radio = plt.axes([0.02, 0.3, 0.15, 0.5], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, coefs_str)

    # Diccionario mutable para mantener el estado
    estado = {'coef_lbl': coef_actual_lbl}

    def update_view():
        idx_rot = int(slider_rot.val)
        idx_tilt = int(slider_tilt.val)
        lbl = estado['coef_lbl']
        
        t_actual = tilteos_disp[idx_tilt]
        angulo_real = ANGULOS_DEFAULT[idx_rot] if idx_rot < len(ANGULOS_DEFAULT) else idx_rot
        
        slider_tilt.valtext.set_text(f"{t_actual}°")
        slider_rot.valtext.set_text(f"{angulo_real:.1f}°")
        
        if t_actual in datos_globales[lbl]:
            im.set_data(datos_globales[lbl][t_actual][idx_rot, :, :])
            ax.set_title(f"Tilt = {t_actual}° | $\\theta$ = {angulo_real:.1f}°")
        else:
            ax.set_title(f"Sin datos: Tilt = {t_actual}°")
            
        fig.canvas.draw_idle()

    def update_slider(val):
        update_view()
        
    def update_radio(label):
        estado['coef_lbl'] = label
        
        # Recalcular límites de color para el nuevo coeficiente
        if datos_globales[label]:
            vmin_new = min([np.min(d) for d in datos_globales[label].values()])
            vmax_new = max([np.max(d) for d in datos_globales[label].values()])
            im.set_clim(vmin=vmin_new, vmax=vmax_new)
            
            # Actualizar título principal
            l_val, mu_val = label.replace("L=","").replace("mu=","").split(", ")
            plt.suptitle(f"Textura: {carpeta_material.replace('Material_', '')} | Coeficiente $A_{{{l_val},{mu_val}}}$", fontweight='bold')
            
        update_view()

    # Callbacks
    slider_rot.on_changed(update_slider)
    slider_tilt.on_changed(update_slider)
    radio.on_clicked(update_radio)
    
    # Evitar recolección de basura
    fig.slider_rot = slider_rot
    fig.slider_tilt = slider_tilt
    fig.radio = radio
    
    # Init display
    slider_tilt.valtext.set_text(f"{tilteos_disp[0]}°")
    slider_rot.valtext.set_text(f"{ANGULOS_DEFAULT[0]:.1f}°")
    
    l_init, mu_init = coef_actual_lbl.replace("L=","").replace("mu=","").split(", ")
    plt.suptitle(f"Textura: {carpeta_material.replace('Material_', '')} | Coeficiente $A_{{{l_init},{mu_init}}}$", fontweight='bold')
    update_view()
    
    plt.show(block=True)

# %% [BLOQUE 4] VISUALIZADOR DE GEOMETRÍA 3D (VÓXEL A VÓXEL REAL)
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
    Los sliders de rotación y tilteo operan discretamente barriendo 
    únicamente los valores disponibles de la simulación.
    """
    # Recolectar datos simulados disponibles para el paso discreto
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

# %% [BLOQUE 5] MENÚ INTERACTIVO PRINCIPAL
def menu_interactivo():
    while True:
        print("\n" + "="*50)
        print(" VISUALIZADOR TENSORIAL INTERACTIVO")
        print("="*50)
        print("1. Ver Geometría Experimental 3D (Discreto Real)")
        print("2. Navegar Proyecciones Detector A_lmu (Discreto Real)")
        print("3. Extraer Espectro 1D de un Píxel (Discreto Real)")
        print("4. Salir")
        
        try:
            opcion = int(input("\nSelecciona una opción (1-4): ") or "0")
            
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
                    # Ya no pedimos l y mu aquí. Se carga todo en la UI.
                    visualizar_proyecciones_Almu(carpeta_elegida)
                else:
                    print("Selección de material no válida.")
                
            elif opcion == 3:
                x = int(input("Posición X del píxel en detector [35]: ") or "35")
                z = int(input("Posición Z del píxel en detector [35]: ") or "35")
                visualizar_espectro_pixel_desde_4d(x, z)
                
            elif opcion == 4:
                print("\nCerrando visualizador...")
                break
            else:
                print("Opción no válida.")
                
        except ValueError:
            print("Entrada inválida. Por favor ingresa números.")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")

if __name__ == '__main__':
    menu_interactivo()