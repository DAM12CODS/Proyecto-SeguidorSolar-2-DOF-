import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import customtkinter as ctk
from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timedelta
from pytz import timezone

# Configuración global de la UI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------------
# LÓGICA MATEMÁTICA Y GEOMETRÍA
# --------------------------------------------------------------------


def getSolarPosition(latitude=-0.2105367, longitude=-78.491614, date=None):
    """Obtiene la posición solar real usando pysolar."""
    if date is None:
        date = datetime.now(tz=timezone("America/Guayaquil"))
    az = get_azimuth(latitude, longitude, date)
    el = get_altitude(latitude, longitude, date)
    return az, el


def calculate_control_angles(azimuth_deg, elevation_deg):
    """Calcula los ángulos del seguidor 2-DOF."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    pitch = np.arcsin(-np.cos(el) * np.cos(az))
    roll = np.arctan2(np.cos(el) * np.sin(az), np.sin(el))
    return np.degrees(pitch), np.degrees(roll)


def build_detailed_panel(pitch_deg, roll_deg, pivot_z=0.8, thickness=0.025):
    """Genera la geometría 3D del panel (Cara frontal con celdas + Cara posterior oscura)."""
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    width, height = 1.0, 1.4
    cols, rows = 4, 6
    margin, gap = 0.05, 0.02

    polygons, colors = [], []

    # 1. CARA POSTERIOR / TRASERA (Color gris oscuro / negro metálico)
    # Se coloca en la parte inferior del panel (z = -thickness)
    back_face = np.array([
        [-height / 2, -width / 2, -thickness],
        [height / 2, -width / 2, -thickness],
        [height / 2, width / 2, -thickness],
        [-height / 2, width / 2, -thickness],
    ])
    polygons.append(back_face)
    colors.append("#1e293b")  # Gris oscuro mate para el reverso

    # 2. MARCO ESTRUCTURAL (Frontal)
    frame = np.array([
        [-height / 2, -width / 2, 0],
        [height / 2, -width / 2, 0],
        [height / 2, width / 2, 0],
        [-height / 2, width / 2, 0],
    ])
    polygons.append(frame)
    colors.append("#475569")  # Gris aluminio/marco

    # 3. CELDAS SOLARES (Frontales sobrepuestas)
    cell_h = (height - 2 * margin - (rows - 1) * gap) / rows
    cell_w = (width - 2 * margin - (cols - 1) * gap) / cols

    for r in range(rows):
        for c in range(cols):
            x0 = -height / 2 + margin + r * (cell_h + gap)
            y0 = -width / 2 + margin + c * (cell_w + gap)
            cell = np.array([
                [x0, y0, 0.005],
                [x0 + cell_h, y0, 0.005],
                [x0 + cell_h, y0 + cell_w, 0.005],
                [x0, y0 + cell_w, 0.005],
            ])
            polygons.append(cell)
            colors.append("#2563eb")  # Azul fotovoltaico brillante

    # MATRICES DE ROTACIÓN
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    Ry = np.array([
        [np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)],
    ])
    R = Ry @ Rx

    # Aplicar rotación y elevación Z a todos los componentes
    rotated_polygons = []
    for poly in polygons:
        rotated = (R @ poly.T).T
        rotated[:, 2] += pivot_z
        rotated_polygons.append(rotated)

    return rotated_polygons, colors


def project_shadow(polygons, az, el):
    """Proyecta matemáticamente la sombra del panel sobre el suelo (Z=0)."""
    if el <= 0:
        return []
    az_rad, el_rad = np.radians(az), np.radians(el)

    ux = np.cos(el_rad) * np.sin(az_rad)
    uy = np.cos(el_rad) * np.cos(az_rad)
    uz = np.sin(el_rad)

    shadow_polys = []
    for poly in polygons:
        shadow = poly.copy()
        shadow[:, 0] = poly[:, 0] - poly[:, 2] * (ux / uz)
        shadow[:, 1] = poly[:, 1] - poly[:, 2] * (uy / uz)
        shadow[:, 2] = 0.001
        shadow_polys.append(shadow)

    return shadow_polys


# --------------------------------------------------------------------
# APLICACIÓN CUSTOMTKINTER
# --------------------------------------------------------------------


class SolarDashboard(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Solar Tracker Simulation Pro - EPN")
        self.geometry("1300x850")
        self.minsize(1000, 700)

        self.animation = None
        self.is_running = False
        self.frame_index = 0

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_area()
        self.init_plot()

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(self.sidebar,
                     text="Parámetros",
                     font=ctk.CTkFont(size=20,
                                      weight="bold")).grid(row=0,
                                                           column=0,
                                                           padx=20,
                                                           pady=(30, 20),
                                                           sticky="w")

        ctk.CTkLabel(self.sidebar,
                     text="Fecha (YYYY-MM-DD):").grid(row=1,
                                                      column=0,
                                                      padx=20,
                                                      pady=(10, 0),
                                                      sticky="w")
        self.date_entry = ctk.CTkEntry(self.sidebar)
        self.date_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.date_entry.grid(row=2,
                             column=0,
                             padx=20,
                             pady=(5, 10),
                             sticky="ew")

        ctk.CTkLabel(self.sidebar,
                     text="Hora Inicio (HH:MM):").grid(row=3,
                                                       column=0,
                                                       padx=20,
                                                       pady=(10, 0),
                                                       sticky="w")
        self.time_entry = ctk.CTkEntry(self.sidebar)
        self.time_entry.insert(0, "06:00")
        self.time_entry.grid(row=4,
                             column=0,
                             padx=20,
                             pady=(5, 10),
                             sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Duración (Horas):").grid(row=5,
                                                                  column=0,
                                                                  padx=20,
                                                                  pady=(10, 0),
                                                                  sticky="w")
        self.duration_entry = ctk.CTkEntry(self.sidebar)
        self.duration_entry.insert(0, "12")
        self.duration_entry.grid(row=6,
                                 column=0,
                                 padx=20,
                                 pady=(5, 10),
                                 sticky="new")

        self.control_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.control_frame.grid(row=7, column=0, padx=20, pady=20, sticky="ew")

        self.btn_simulate = ctk.CTkButton(self.control_frame,
                                          text="▶ Iniciar Render",
                                          command=self.toggle_simulation,
                                          fg_color="#10b981",
                                          hover_color="#059669")
        self.btn_simulate.pack(fill="x", pady=5)

        self.btn_reset = ctk.CTkButton(self.control_frame,
                                       text="⏹ Reiniciar",
                                       command=self.reset_simulation,
                                       fg_color="#ef4444",
                                       hover_color="#dc2626")
        self.btn_reset.pack(fill="x", pady=5)

        ctk.CTkLabel(self.control_frame, text="Velocidad:").pack(anchor="w",
                                                                 pady=(15, 0))
        self.speed_slider = ctk.CTkSlider(self.control_frame, from_=1, to=100)
        self.speed_slider.set(80)
        self.speed_slider.pack(fill="x", pady=5)

        ctk.CTkLabel(self.sidebar, text="Progreso Diario:").grid(row=8,
                                                                 column=0,
                                                                 padx=20,
                                                                 sticky="w")
        self.progress_bar = ctk.CTkProgressBar(self.sidebar,
                                               progress_color="#38bdf8")
        self.progress_bar.grid(row=9,
                               column=0,
                               padx=20,
                               pady=(5, 30),
                               sticky="ew")
        self.progress_bar.set(0)

    def create_main_area(self):
        self.main_area = ctk.CTkFrame(self, fg_color="transparent")
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_area.grid_rowconfigure(1, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)

        self.kpi_frame = ctk.CTkFrame(self.main_area, fg_color="transparent")
        self.kpi_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        for i in range(5):
            self.kpi_frame.grid_columnconfigure(i, weight=1)

        self.lbl_time_val = self.create_kpi_card(self.kpi_frame, 0,
                                                 "HORA ACTUAL", "--:--")
        self.lbl_el_val = self.create_kpi_card(self.kpi_frame, 1, "ELEVACIÓN",
                                               "0.0°")
        self.lbl_az_val = self.create_kpi_card(self.kpi_frame, 2, "AZIMUTH",
                                               "0.0°")
        self.lbl_pitch_val = self.create_kpi_card(self.kpi_frame, 3,
                                                  "PITCH (X)", "0.0°")
        self.lbl_roll_val = self.create_kpi_card(self.kpi_frame, 4, "ROLL (Y)",
                                                 "0.0°")

        self.plot_frame = ctk.CTkFrame(self.main_area, corner_radius=15)
        self.plot_frame.grid(row=1, column=0, sticky="nsew")

        plt.style.use('dark_background')
        # Figura principal dividida (3D a la izquierda, 2D a la derecha)
        self.fig = plt.figure(figsize=(12, 6), facecolor="#242424")
        self.ax_3d = self.fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_pitch = self.fig.add_subplot(2, 2, 2)
        self.ax_roll = self.fig.add_subplot(2, 2, 4)

        self.ax_3d.set_facecolor('#242424')
        self.ax_pitch.set_facecolor('#1e1e1e')
        self.ax_roll.set_facecolor('#1e1e1e')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both",
                                         expand=True,
                                         padx=10,
                                         pady=10)

    def create_kpi_card(self, parent, col, title, initial_val):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color="#2b2b2b")
        card.grid(row=0, column=col, padx=5, sticky="ew")
        ctk.CTkLabel(card,
                     text=title,
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#9ca3af").pack(pady=(10, 0))
        val_label = ctk.CTkLabel(card,
                                 text=initial_val,
                                 font=ctk.CTkFont(size=24, weight="bold"),
                                 text_color="#38bdf8")
        val_label.pack(pady=(0, 10))
        return val_label

    def init_plot(self):
        self.ax_3d.clear()
        self.ax_pitch.clear()
        self.ax_roll.clear()

        self.ax_3d.set_axis_off()
        self.ax_3d.set_autoscale_on(False)

        r = 1.9
        self.ax_3d.set_xlim([-r, r])
        self.ax_3d.set_ylim([-r, r])
        self.ax_3d.set_zlim([0, r])

        # --- BRÚJULA BASE ---
        theta = np.linspace(0, 2 * np.pi, 120)
        self.ax_3d.plot(1.8 * np.cos(theta),
                        1.8 * np.sin(theta),
                        0,
                        color='#38bdf8',
                        alpha=0.6,
                        linewidth=2)

        for r_circle in [0.6, 1.2]:
            self.ax_3d.plot(r_circle * np.cos(theta),
                            r_circle * np.sin(theta),
                            0,
                            color='#4b5563',
                            alpha=0.5,
                            linestyle='--')

        self.ax_3d.plot([-1.8, 1.8], [0, 0], [0, 0],
                        color='#4b5563',
                        alpha=0.6,
                        linewidth=1.5)
        self.ax_3d.plot([0, 0], [-1.8, 1.8], [0, 0],
                        color='#4b5563',
                        alpha=0.6,
                        linewidth=1.5)

        offset = 2.0
        self.ax_3d.text(offset,
                        0,
                        0,
                        'ESTE',
                        color='#38bdf8',
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        va='center')
        self.ax_3d.text(-offset,
                        0,
                        0,
                        'OESTE',
                        color='#38bdf8',
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        va='center')
        self.ax_3d.text(0,
                        offset,
                        0,
                        'NORTE',
                        color='#38bdf8',
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        va='center')
        self.ax_3d.text(0,
                        -offset,
                        0,
                        'SUR',
                        color='#38bdf8',
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        va='center')

        # HUD en 3D
        self.hud_text = self.ax_3d.text2D(0.02,
                                          0.95,
                                          "",
                                          transform=self.ax_3d.transAxes,
                                          fontsize=9,
                                          bbox=dict(boxstyle='round',
                                                    facecolor='black',
                                                    alpha=0.6),
                                          color='white',
                                          verticalalignment='top')

        self.canvas.draw()

    def preprocess_data(self):
        try:
            date_str = self.date_entry.get()
            time_str = self.time_entry.get()
            hrs = float(self.duration_entry.get())
            start_dt = timezone("America/Guayaquil").localize(
                datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M"))
        except ValueError:
            return False

        self.times = [
            start_dt + timedelta(minutes=i) for i in range(int(hrs * 60))
        ]
        self.azimuths, self.elevations, self.pitches, self.rolls = [], [], [], []

        for t in self.times:
            az, el = getSolarPosition(date=t)
            self.azimuths.append(az)
            self.elevations.append(el)
            p, r = calculate_control_angles(az, el) if el > 0 else (0, 0)
            self.pitches.append(p)
            self.rolls.append(r)

        return True

    def toggle_simulation(self):
        if not hasattr(self, 'times') or len(self.times) == 0:
            if not self.preprocess_data(): return
            self.setup_3d_environment()

        if self.is_running:
            self.is_running = False
            self.btn_simulate.configure(text="▶ Reanudar", fg_color="#10b981")
            if self.animation: self.after_cancel(self.animation)
        else:
            self.is_running = True
            self.btn_simulate.configure(text="⏸ Pausar",
                                        fg_color="#f59e0b",
                                        hover_color="#d97706")
            self.animate()

    def reset_simulation(self):
        self.is_running = False
        if self.animation: self.after_cancel(self.animation)
        self.frame_index = 0
        self.progress_bar.set(0)
        self.btn_simulate.configure(text="▶ Iniciar Render",
                                    fg_color="#10b981")
        self.times = []
        self.init_plot()
        for label in [
                self.lbl_time_val, self.lbl_el_val, self.lbl_az_val,
                self.lbl_pitch_val, self.lbl_roll_val
        ]:
            label.configure(text="--")

    def setup_3d_environment(self):
        self.init_plot()
        self.pivot_z = 0.8

        # Configurar Gráficas 2D de Ángulos
        hours = [t.hour + t.minute / 60.0 for t in self.times]
        self.ax_pitch.plot(hours, self.pitches, color='#ef4444', linewidth=1.5)
        self.ax_pitch.set_ylabel('Pitch φ (°)', fontsize=8)
        self.ax_pitch.grid(True, alpha=0.3)
        self.ax_pitch.set_title('Ángulo Pitch vs. Hora',
                                fontsize=10,
                                color='white')

        self.ax_roll.plot(hours, self.rolls, color='#38bdf8', linewidth=1.5)
        self.ax_roll.set_xlabel('Hora Local (h)', fontsize=8)
        self.ax_roll.set_ylabel('Roll ψ (°)', fontsize=8)
        self.ax_roll.grid(True, alpha=0.3)
        self.ax_roll.set_title('Ángulo Roll vs. Hora',
                               fontsize=10,
                               color='white')

        self.line_p = self.ax_pitch.axvline(x=hours[0],
                                            color='white',
                                            linestyle='--',
                                            alpha=0.8)
        self.line_r = self.ax_roll.axvline(x=hours[0],
                                           color='white',
                                           linestyle='--',
                                           alpha=0.8)

        # Objetos 3D
        self.ax_3d.plot([0, 0], [0, 0], [0, self.pivot_z],
                        color='#94a3b8',
                        linewidth=6)
        self.ax_3d.plot([0], [0], [0],
                        marker='o',
                        color='#38bdf8',
                        markersize=8)

        polys, colors = build_detailed_panel(self.pitches[0], self.rolls[0],
                                             self.pivot_z)
        self.panel_collection = Poly3DCollection(polys,
                                                 facecolors=colors,
                                                 edgecolors='#020617',
                                                 linewidths=0.5,
                                                 alpha=0.95)
        self.ax_3d.add_collection3d(self.panel_collection)

        self.shadow_collection = Poly3DCollection([],
                                                  facecolors='#000000',
                                                  edgecolors='none',
                                                  alpha=0.5)
        self.ax_3d.add_collection3d(self.shadow_collection)

        self.sun_glow, = self.ax_3d.plot([], [], [],
                                         marker='o',
                                         color='#fef08a',
                                         markersize=40,
                                         alpha=0.15)
        self.sun_core, = self.ax_3d.plot([], [], [],
                                         marker='o',
                                         color='#fbbf24',
                                         markersize=10)
        self.sun_ray, = self.ax_3d.plot([], [], [],
                                        color='#fef08a',
                                        linestyle='--',
                                        linewidth=1.5,
                                        alpha=0.5)

        self.sun_dropline, = self.ax_3d.plot([], [], [],
                                             color='#4b5563',
                                             linestyle=':',
                                             alpha=0.8,
                                             linewidth=1)
        self.sun_ground_mark, = self.ax_3d.plot([], [], [],
                                                marker='+',
                                                color='#f59e0b',
                                                markersize=8,
                                                alpha=0.6)

        # Vector Normal del Panel (n̂)
        self.normal_quiver = None

        # Arco de Trayectoria
        sx = 1.7 * np.cos(np.radians(self.elevations)) * np.sin(
            np.radians(self.azimuths))
        sy = 1.7 * np.cos(np.radians(self.elevations)) * np.cos(
            np.radians(self.azimuths))
        sz = 1.7 * np.sin(np.radians(self.elevations))
        valid = sz > 0
        self.ax_3d.plot(sx[valid],
                        sy[valid],
                        sz[valid],
                        color='#fbbf24',
                        alpha=0.2,
                        linewidth=2)

    def animate(self):
        if not self.is_running: return
        if self.frame_index >= len(self.times):
            self.is_running = False
            self.btn_simulate.configure(text="▶ Iniciar Render",
                                        fg_color="#10b981")
            return

        i = self.frame_index
        el = self.elevations[i]
        curr_time = self.times[i]
        curr_hour = curr_time.hour + curr_time.minute / 60.0
        self.progress_bar.set((i + 1) / len(self.times))

        if el > 0:
            az, pitch, roll = self.azimuths[i], self.pitches[i], self.rolls[i]

            # Telemetría UI
            time_str = curr_time.strftime("%H:%M")
            self.lbl_time_val.configure(text=time_str)
            self.lbl_el_val.configure(text=f"{el:.1f}°")
            self.lbl_az_val.configure(text=f"{az:.1f}°")
            self.lbl_pitch_val.configure(text=f"{pitch:.1f}°")
            self.lbl_roll_val.configure(text=f"{roll:.1f}°")

            # Actualizar líneas verticales 2D
            self.line_p.set_xdata([curr_hour, curr_hour])
            self.line_r.set_xdata([curr_hour, curr_hour])

            # Coordenadas del Sol
            R_sun = 1.7
            sx = R_sun * np.cos(np.radians(el)) * np.sin(np.radians(az))
            sy = R_sun * np.cos(np.radians(el)) * np.cos(np.radians(az))
            sz = R_sun * np.sin(np.radians(el))

            # Actualizar Sol
            self.sun_core.set_data([sx], [sy])
            self.sun_core.set_3d_properties([sz])
            self.sun_glow.set_data([sx], [sy])
            self.sun_glow.set_3d_properties([sz])
            self.sun_ray.set_data([0, sx], [0, sy])
            self.sun_ray.set_3d_properties([self.pivot_z, sz])

            self.sun_dropline.set_data([sx, sx], [sy, sy])
            self.sun_dropline.set_3d_properties([0, sz])
            self.sun_ground_mark.set_data([sx], [sy])
            self.sun_ground_mark.set_3d_properties([0])

            # Actualizar Panel y Sombra
            polys, _ = build_detailed_panel(pitch, roll, self.pivot_z)
            self.panel_collection.set_verts(polys)

            shadows = project_shadow(polys, az, el)
            self.shadow_collection.set_verts(shadows)
            alpha_shadow = max(0, 0.5 * (el / 10.0)) if el < 10 else 0.5
            self.shadow_collection.set_alpha(alpha_shadow)

            # --- DIBUJAR VECTOR NORMAL (n̂) ---
            if self.normal_quiver:
                self.normal_quiver.remove()

            pitch_r, roll_r = np.radians(pitch), np.radians(roll)
            Rx = np.array([[1, 0, 0], [0, np.cos(pitch_r), -np.sin(pitch_r)],
                           [0, np.sin(pitch_r),
                            np.cos(pitch_r)]])
            Ry = np.array([[np.cos(roll_r), 0,
                            np.sin(roll_r)], [0, 1, 0],
                           [-np.sin(roll_r), 0,
                            np.cos(roll_r)]])
            n_vec = Ry @ Rx @ np.array([0, 0, 1])

            self.normal_quiver = self.ax_3d.quiver(0,
                                                   0,
                                                   self.pivot_z,
                                                   n_vec[0] * 0.8,
                                                   n_vec[1] * 0.8,
                                                   n_vec[2] * 0.8,
                                                   color='#06b6d4',
                                                   linewidth=2,
                                                   arrow_length_ratio=0.25)

            # HUD 3D
            self.hud_text.set_text(
                f"Hora: {time_str}\n"
                f"Sol   → Elev: {el:.1f}° | Az: {az:.1f}°\n"
                f"Panel → Pitch: {pitch:.1f}° | Roll: {roll:.1f}°")

            self.canvas.draw_idle()

        self.frame_index += 1
        delay = int(150 - (self.speed_slider.get() * 1.4))
        self.animation = self.after(delay, self.animate)


if __name__ == "__main__":
    app = SolarDashboard()
    app.mainloop()
