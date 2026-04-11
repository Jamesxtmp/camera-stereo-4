import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

try:
    from PIL import Image, ImageTk
except Exception as exc:
    raise SystemExit(f"Pillow es requerido para la vista de camara en Tkinter: {exc}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_utils import DROIDCAM_FIXED_INDEX, abrir_camara_con_fallback
from src.app_config import CAMERA_WIDTH, CAMERA_HEIGHT, LEFT_FOLDER


BAUDRATE = 115200
DEFAULT_PORT = "COM7"
SERVO_COUNT = 6
SERVO_LABELS = [
    "Servo 1",
    "Servo 2",
    "Servo 3",
    "Servo 4",
    "Servo 5",
    "Servo 6",
]
WINDOW_TITLE = "Control Garra/Orugas + Camara"


class ControlCamUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1180x860")
        self.root.minsize(980, 760)

        self.serial_port = None
        self.serial_lock = threading.Lock()
        self.current_port = tk.StringVar(value=DEFAULT_PORT)
        self.status_text = tk.StringVar(value="Desconectado")
        self.port_status = tk.StringVar(value="Puerto: no conectado")

        self.sliders = []
        self.value_labels = []
        self.command_text = tk.StringVar(value="1:90 2:90 3:90 4:90 5:90 6:90")

        self.cap = None
        self.camera_backend_name = ""
        self.camera_status = tk.StringVar(value="Camara: desconectada")
        self.last_frame = None
        self.camera_running = False
        self.flip_180 = False
        self.current_photo_path = tk.StringVar(value="Sin capturas")

        left_folder = Path(LEFT_FOLDER)
        if not left_folder.is_absolute():
            left_folder = PROJECT_ROOT / left_folder
        self.left_folder = left_folder
        self.left_folder.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(200, self.refresh_ports)
        self.start_camera()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        # Layout principal: video a la izquierda, controles a la derecha.
        body = ttk.Frame(main)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Panel camara
        camera_frame = ttk.LabelFrame(body, text="Camara en vivo", padding=10)
        camera_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.video_label = tk.Label(
            camera_frame,
            text="Iniciando camara...",
            bg="#111111",
            fg="#cfcfcf",
            width=78,
            height=32,
            anchor="center",
        )
        self.video_label.pack(fill="both", expand=True)

        camera_info = ttk.Frame(camera_frame)
        camera_info.pack(fill="x", pady=(8, 0))
        ttk.Label(camera_info, textvariable=self.camera_status).pack(side="left")
        ttk.Button(camera_info, text="Voltear 180°", command=self.toggle_flip_180).pack(side="right", padx=(8, 0))
        ttk.Button(camera_info, text="Tomar foto", command=self.take_photo).pack(side="right")

        ttk.Label(camera_frame, textvariable=self.current_photo_path, foreground="#555555").pack(anchor="w", pady=(6, 0))

        # Panel controles
        controls = ttk.Frame(body)
        controls.grid(row=0, column=1, sticky="nsew")

        title = ttk.Label(controls, text="Control de Garra + Orugas", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w")

        conn_frame = ttk.LabelFrame(controls, text="Conexion serie", padding=10)
        conn_frame.pack(fill="x", pady=(10, 0))

        port_row = ttk.Frame(conn_frame)
        port_row.pack(fill="x")

        ttk.Label(port_row, text="Puerto:").pack(side="left")
        self.port_combo = ttk.Combobox(port_row, textvariable=self.current_port, width=16, state="readonly")
        self.port_combo.pack(side="left", padx=(8, 8))

        ttk.Button(port_row, text="Actualizar", command=self.refresh_ports).pack(side="left")
        ttk.Button(port_row, text="Conectar", command=self.connect).pack(side="left", padx=(8, 0))
        ttk.Button(port_row, text="Desconectar", command=self.disconnect).pack(side="left", padx=(8, 0))

        ttk.Label(conn_frame, textvariable=self.port_status).pack(anchor="w", pady=(8, 0))
        ttk.Label(conn_frame, textvariable=self.status_text).pack(anchor="w")

        tracks_frame = ttk.LabelFrame(controls, text="Orugas", padding=10)
        tracks_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(tracks_frame, text="Comandos: F=Adelante B=Atras L=Izquierda R=Derecha S=Stop").pack(anchor="w")

        track_grid = ttk.Frame(tracks_frame)
        track_grid.pack(pady=(8, 0), fill="x")
        self._create_track_button(track_grid, "Adelante", "F", 0, 1)
        self._create_track_button(track_grid, "Izquierda", "L", 1, 0)
        self._create_track_button(track_grid, "STOP", "S", 1, 1, is_stop=True)
        self._create_track_button(track_grid, "Derecha", "R", 1, 2)
        self._create_track_button(track_grid, "Atras", "B", 2, 1)

        manual_frame = ttk.LabelFrame(controls, text="Entrada manual", padding=10)
        manual_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(manual_frame, text="Formato: 1:50 2:100 3:140 4:30 5:30 6:110").pack(anchor="w")

        entry_row = ttk.Frame(manual_frame)
        entry_row.pack(fill="x", pady=(8, 0))

        self.command_entry = ttk.Entry(entry_row, textvariable=self.command_text)
        self.command_entry.pack(side="left", fill="x", expand=True)
        self.command_entry.bind("<Return>", lambda event: self.send_manual_commands())

        ttk.Button(entry_row, text="Enviar", command=self.send_manual_commands).pack(side="left", padx=(8, 0))

        sliders_frame = ttk.LabelFrame(controls, text="Servos", padding=10)
        sliders_frame.pack(fill="both", expand=True, pady=(10, 0))

        for index in range(SERVO_COUNT):
            row = ttk.Frame(sliders_frame)
            row.pack(fill="x", pady=5)

            ttk.Label(row, text=SERVO_LABELS[index], width=8).pack(side="left")

            slider = tk.Scale(
                row,
                from_=0,
                to=180,
                orient="horizontal",
                length=220,
                resolution=1,
                showvalue=False,
                command=lambda value, servo_index=index: self.on_slider_change(servo_index, value),
            )
            slider.set(90)
            slider.pack(side="left", padx=(8, 8), fill="x", expand=True)

            value_label = ttk.Label(row, text="90", width=4)
            value_label.pack(side="left")

            self.sliders.append(slider)
            self.value_labels.append(value_label)

        actions = ttk.Frame(controls)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Centro (90)", command=self.center_all).pack(side="left")
        ttk.Button(actions, text="Enviar todos", command=self.send_all).pack(side="left", padx=(8, 0))

    def _create_track_button(self, parent, text, command, row, col, is_stop=False) -> None:
        width = 12 if not is_stop else 14
        button = ttk.Button(parent, text=text, width=width, command=lambda cmd=command: self.send_track_command(cmd))
        button.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

        if command != "S":
            button.bind("<ButtonPress-1>", lambda _event, cmd=command: self.send_track_command(cmd))
            button.bind("<ButtonRelease-1>", lambda _event: self.send_track_command("S"))

        parent.grid_columnconfigure(col, weight=1)

    def refresh_ports(self) -> None:
        if serial is None:
            self.port_combo["values"] = []
            self.port_status.set("pyserial no esta instalado")
            return

        ports = [port.device for port in serial.tools.list_ports.comports()]
        if ports:
            self.port_combo["values"] = ports
            if self.current_port.get() not in ports:
                self.current_port.set(ports[0])
            self.port_status.set(f"Puertos detectados: {', '.join(ports)}")
        else:
            self.port_combo["values"] = []
            self.port_status.set("No se detectaron puertos serie")

    def connect(self) -> None:
        if serial is None:
            messagebox.showerror("pyserial requerido", "Instala pyserial para usar el controlador serial")
            return

        port = self.current_port.get().strip()
        if not port:
            messagebox.showwarning("Puerto requerido", "Selecciona un puerto serie")
            return

        self.disconnect()

        try:
            self.serial_port = serial.Serial(port, BAUDRATE, timeout=0.1)
            self.status_text.set(f"Conectado a {port} @ {BAUDRATE}")
            self.port_status.set(f"Puerto activo: {port}")
        except Exception as exc:
            self.serial_port = None
            messagebox.showerror("Error de conexion", f"No se pudo abrir {port}\n\n{exc}")
            self.status_text.set("Desconectado")

    def disconnect(self) -> None:
        with self.serial_lock:
            if self.serial_port is not None:
                try:
                    self.serial_port.close()
                except Exception:
                    pass
                self.serial_port = None
        self.status_text.set("Desconectado")

    def send_command(self, servo_number: int, angle: int) -> None:
        if self.serial_port is None or not self.serial_port.is_open:
            return

        payload = f"{servo_number}:{angle}\n".encode("utf-8")
        with self.serial_lock:
            try:
                self.serial_port.write(payload)
                self.serial_port.flush()
            except Exception as exc:
                self.status_text.set("Error al enviar")
                messagebox.showerror("Error serial", f"No se pudo enviar comando {servo_number}:{angle}\n\n{exc}")

    def send_track_command(self, command: str) -> None:
        if self.serial_port is None or not self.serial_port.is_open:
            return

        normalized = command.strip().upper()
        if normalized not in {"F", "B", "L", "R", "S"}:
            return

        payload = f"M:{normalized}\n".encode("utf-8")
        with self.serial_lock:
            try:
                self.serial_port.write(payload)
                self.serial_port.flush()
                self.status_text.set(f"Comando orugas: {normalized}")
            except Exception as exc:
                self.status_text.set("Error al enviar")
                messagebox.showerror("Error serial", f"No se pudo enviar comando de orugas ({normalized})\n\n{exc}")

    def on_slider_change(self, servo_index: int, value: str) -> None:
        angle = int(float(value))
        self.value_labels[servo_index].config(text=str(angle))
        self.send_command(servo_index + 1, angle)

    def center_all(self) -> None:
        for index, slider in enumerate(self.sliders):
            slider.set(90)
            self.value_labels[index].config(text="90")
            self.send_command(index + 1, 90)

    def send_all(self) -> None:
        for index, slider in enumerate(self.sliders):
            angle = int(slider.get())
            self.send_command(index + 1, angle)

    def send_manual_commands(self) -> None:
        raw = self.command_text.get().strip()
        if not raw:
            return

        parts = raw.replace(",", " ").split()
        sent_any = False

        for part in parts:
            if ":" not in part:
                continue

            servo_text, angle_text = part.split(":", 1)

            try:
                servo_number = int(servo_text)
                angle = int(angle_text)
            except ValueError:
                continue

            if 1 <= servo_number <= SERVO_COUNT and 0 <= angle <= 180:
                self.sliders[servo_number - 1].set(angle)
                self.value_labels[servo_number - 1].config(text=str(angle))
                self.send_command(servo_number, angle)
                sent_any = True

        if not sent_any:
            messagebox.showwarning("Entrada invalida", "Usa el formato: 1:50 2:100 3:140 4:30 5:30 6:110")

    def start_camera(self) -> None:
        if self.camera_running:
            return

        try:
            cap, backend_name, _ = abrir_camara_con_fallback(
                indice=DROIDCAM_FIXED_INDEX,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                validate_frames=True,
                frame_attempts=8,
            )
        except RuntimeError as exc:
            self.camera_status.set(f"Camara no disponible: {exc}")
            return

        self.cap = cap
        self.camera_backend_name = backend_name
        self.camera_running = True
        self.camera_status.set(
            f"Camara OK | idx={DROIDCAM_FIXED_INDEX} | backend={backend_name} | {CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )
        self._update_camera_frame()

    def _update_camera_frame(self) -> None:
        if not self.camera_running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.last_frame = frame
            display_frame = cv2.rotate(frame, cv2.ROTATE_180) if self.flip_180 else frame
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            label_w = max(self.video_label.winfo_width(), 640)
            label_h = max(self.video_label.winfo_height(), 360)
            img.thumbnail((label_w, label_h), Image.Resampling.BILINEAR)

            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
        else:
            self.camera_status.set("Camara conectada pero sin frames")

        self.root.after(15, self._update_camera_frame)

    def take_photo(self) -> None:
        if self.last_frame is None:
            messagebox.showwarning("Sin frame", "Aun no hay frame disponible para guardar")
            return

        ts = int(time.time() * 1000)
        path = self.left_folder / f"left_{ts}.jpg"
        frame_to_save = cv2.rotate(self.last_frame, cv2.ROTATE_180) if self.flip_180 else self.last_frame
        ok = cv2.imwrite(str(path), frame_to_save)
        if ok:
            self.current_photo_path.set(f"Foto guardada: {path.name}")
        else:
            messagebox.showerror("Error", "No se pudo guardar la foto")

    def toggle_flip_180(self) -> None:
        self.flip_180 = not self.flip_180

        if self.camera_running and self.cap is not None:
            estado = "activado" if self.flip_180 else "desactivado"
            self.camera_status.set(
                f"Camara OK | idx={DROIDCAM_FIXED_INDEX} | backend={self.camera_backend_name} | {CAMERA_WIDTH}x{CAMERA_HEIGHT} | giro 180 {estado}"
            )

    def on_close(self) -> None:
        self.disconnect()

        self.camera_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = ControlCamUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
