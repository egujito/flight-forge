from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional

os.environ.setdefault("MPLBACKEND", "QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PySide6.QtCore import QDate, QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from flightForge import Environment, Motor, Parachute, Rocket, Simulation
from flightForge.extras import Campaign
from flightForge.extras.analysis import apogee_histogram, landing_scatter, sensitivity_tornado
from flightForge.extras.param import Param
from flightForge.extras.results import CampaignResults
from flightForge.extras.runner import BaseObjects, execute_run
from flightForge.utils import logarithmic_thrust

UPDATE_EVERY = 50

CHANNELS = [
    "t", "x", "y", "z", "vx", "vy", "vz", "speed",
    "ax", "ay", "az", "acceleration",
    "mass", "thrust", "drag", "total_mdot", "grain_mdot", "mach",
]

EVENT_COLORS = {
    "rail_departure": "#4ec9b0",
    "burn_out": "#f59a42",
    "apogee": "#c586c0",
    "impact": "#f44747",
}


# ---------------------------------------------------------------------------
# Dark palette
# ---------------------------------------------------------------------------

def apply_dark_palette(app: QApplication) -> None:
    pal = QPalette()
    dark = QColor("#1e1e1e")
    base = QColor("#252526")
    mid = QColor("#3c3c3c")
    text = QColor("#d4d4d4")
    dim_text = QColor("#888888")
    highlight = QColor("#0d7acc")
    R = QPalette.ColorRole
    pal.setColor(R.Window, dark)
    pal.setColor(R.WindowText, text)
    pal.setColor(R.Base, base)
    pal.setColor(R.AlternateBase, mid)
    pal.setColor(R.ToolTipBase, mid)
    pal.setColor(R.ToolTipText, text)
    pal.setColor(R.Text, text)
    pal.setColor(R.Button, mid)
    pal.setColor(R.ButtonText, text)
    pal.setColor(R.BrightText, QColor("#ffffff"))
    pal.setColor(R.Highlight, highlight)
    pal.setColor(R.HighlightedText, QColor("#ffffff"))
    pal.setColor(R.PlaceholderText, dim_text)
    pal.setColor(QPalette.ColorGroup.Disabled, R.Text, dim_text)
    pal.setColor(QPalette.ColorGroup.Disabled, R.ButtonText, dim_text)
    app.setPalette(pal)
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            font-weight: bold;
            color: #aaaaaa;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QDateEdit {
            background: #2d2d2d; border: 1px solid #555; border-radius: 3px;
            padding: 2px 4px; color: #d4d4d4;
        }
        QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #0d7acc;
        }
        QPushButton {
            background: #3c3c3c; border: 1px solid #555; border-radius: 4px;
            padding: 4px 12px; color: #d4d4d4;
        }
        QPushButton:hover { background: #4a4a4a; }
        QPushButton:pressed { background: #0d7acc; color: white; }
        QPushButton:disabled { color: #666; }
        QTabBar::tab { background: #2d2d2d; color: #aaa; padding: 6px 14px; border: 1px solid #444; }
        QTabBar::tab:selected { background: #1e1e1e; color: #d4d4d4; border-bottom: 2px solid #0d7acc; }
        QListWidget { background: #252526; border: 1px solid #555; }
        QTableWidget { background: #252526; gridline-color: #444; }
        QHeaderView::section { background: #3c3c3c; color: #aaa; border: 1px solid #444; padding: 4px; }
        QTextEdit { background: #1a1a1a; border: 1px solid #555; color: #c8c8c8; }
        QSplitter::handle { background: #3c3c3c; }
        QScrollBar:vertical { background: #252526; width: 10px; }
        QScrollBar::handle:vertical { background: #555; border-radius: 5px; min-height: 20px; }
    """)


# ---------------------------------------------------------------------------
# Matplotlib canvas helpers
# ---------------------------------------------------------------------------

def _styled_fig(nrows=1, ncols=1, proj3d=False) -> tuple[Figure, object]:
    fig = Figure(facecolor="#1e1e1e", tight_layout=True)
    ax = fig.add_subplot(111, projection="3d" if proj3d else None)
    ax.set_facecolor("#2d2d2d")
    if proj3d:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("#444444")
        ax.tick_params(colors="#888888", labelsize=7)
    else:
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors="#888888", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#d4d4d4")
        ax.grid(True, alpha=0.2, color="#555555")
    return fig, ax


class LiveCanvas2D(FigureCanvasQTAgg):
    def __init__(self, xlabel: str, ylabel: str, title: str, color: str):
        fig, ax = _styled_fig()
        super().__init__(fig)
        self.fig = fig
        self.ax = ax
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self.line, = ax.plot([], [], color=color, linewidth=1.5)
        self._vlines: dict[str, object] = {}
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_data(self, xs: list, ys: list) -> None:
        self.line.set_data(xs, ys)
        if xs:
            self.ax.relim()
            self.ax.autoscale_view()
        self.draw_idle()

    def add_vline(self, name: str, x: float) -> None:
        if name in self._vlines:
            return
        color = EVENT_COLORS.get(name, "#ffffff")
        vl = self.ax.axvline(x=x, color=color, linestyle="--", linewidth=1.0, alpha=0.8, label=name)
        self._vlines[name] = vl
        self.ax.legend(loc="upper right", facecolor="#2d2d2d", labelcolor="#d4d4d4", fontsize=7)
        self.draw_idle()

    def clear_plot(self) -> None:
        self.ax.cla()
        self._vlines.clear()
        xlabel = self.ax.get_xlabel()
        self.ax.set_facecolor("#2d2d2d")
        for spine in self.ax.spines.values():
            spine.set_color("#555555")
        self.ax.tick_params(colors="#888888", labelsize=8)
        self.ax.xaxis.label.set_color("#aaaaaa")
        self.ax.yaxis.label.set_color("#aaaaaa")
        self.ax.grid(True, alpha=0.2, color="#555555")
        self.line, = self.ax.plot([], [], color=self.line.get_color(), linewidth=1.5)
        self.draw_idle()


class LiveCanvas3D(FigureCanvasQTAgg):
    def __init__(self):
        fig, ax = _styled_fig(proj3d=True)
        super().__init__(fig)
        self.fig = fig
        self.ax = ax
        ax.set_xlabel("X (m)", color="#888", fontsize=8)
        ax.set_ylabel("Y (m)", color="#888", fontsize=8)
        ax.set_zlabel("Alt (m)", color="#888", fontsize=8)
        ax.set_title("Trajectory", color="#d4d4d4", fontsize=9)
        self.line, = ax.plot([], [], [], color="#4a9eff", linewidth=1.5)
        self.current_pt, = ax.plot([], [], [], "ro", markersize=5)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def update_data(self, xs: list, ys: list, zs: list) -> None:
        self.line.set_data(xs, ys)
        self.line.set_3d_properties(zs)
        if xs:
            self.current_pt.set_data([xs[-1]], [ys[-1]])
            self.current_pt.set_3d_properties([zs[-1]])
            self.ax.set_xlim(min(xs) - 1, max(xs) + 1)
            self.ax.set_ylim(min(ys) - 1, max(ys) + 1)
            self.ax.set_zlim(0, max(zs) * 1.1 + 1)
        self.draw_idle()

    def clear_plot(self) -> None:
        self.line.set_data([], [])
        self.line.set_3d_properties([])
        self.current_pt.set_data([], [])
        self.current_pt.set_3d_properties([])
        self.draw_idle()

    def plot_final(self, x, y, z) -> None:
        self.ax.cla()
        self.ax.set_facecolor("#1e1e1e")
        self.ax.set_xlabel("X (m)", color="#888", fontsize=8)
        self.ax.set_ylabel("Y (m)", color="#888", fontsize=8)
        self.ax.set_zlabel("Alt (m)", color="#888", fontsize=8)
        self.ax.set_title("3D Trajectory", color="#d4d4d4", fontsize=9)
        self.ax.plot(x, y, z, color="#4a9eff", linewidth=1.5)
        self.ax.scatter([x[0]], [y[0]], [z[0]], color="#4ec9b0", s=40, zorder=5, label="Launch")
        self.ax.scatter([x[-1]], [y[-1]], [z[-1]], color="#f44747", s=40, marker="x", zorder=5, label="Impact")
        self.ax.set_zlim(0, max(z) * 1.1 + 1)
        self.ax.legend(loc="upper right", facecolor="#2d2d2d", labelcolor="#d4d4d4", fontsize=7)
        self.draw_idle()


# ---------------------------------------------------------------------------
# GUILogHandler
# ---------------------------------------------------------------------------

class _LogEmitter(QObject):
    sig_log = Signal(str)


class GUILogHandler(logging.Handler):
    _ansi = re.compile(r"\x1b\[[0-9;]*[mK]")

    def __init__(self) -> None:
        super().__init__()
        self.emitter = _LogEmitter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self._ansi.sub("", self.format(record))
        self.emitter.sig_log.emit(msg)


# ---------------------------------------------------------------------------
# SimulationWorker
# ---------------------------------------------------------------------------

class SimulationWorker(QThread):
    step_update = Signal(object)
    event_fired = Signal(str, float)
    run_finished = Signal(object)
    run_error = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def _build_objects(self):
        cfg = self._config

        env = Environment()
        if cfg["env"]["use_api"]:
            if not cfg["env"]["api_key"]:
                raise ValueError("Windy API key is empty.")
            env.set_model(
                cfg["env"]["api_key"],
                cfg["env"]["lat"],
                cfg["env"]["lon"],
                cfg["env"]["model"],
                (cfg["env"]["day"], cfg["env"]["month"], cfg["env"]["year"]),
            )

        m = cfg["motor"]
        if m["thrust_type"] == "csv":
            if not os.path.isfile(m["csv_path"]):
                raise ValueError(f"Thrust CSV not found: {m['csv_path']}")
            thrust_src = m["csv_path"]
        else:
            thrust_src = logarithmic_thrust(m["burn_time"], m["peak_thrust"], m["ramp_time"])

        motor = Motor(
            thrust_src,
            m["burn_time"],
            ox_mdot=m["ox_mdot"],
            initial_ox_mass=m["initial_ox_mass"],
            initial_grain_mass=m["initial_grain_mass"],
        )

        r = cfg["rocket"]
        if not os.path.isfile(r["drag_csv"]):
            raise ValueError(f"Drag CSV not found: {r['drag_csv']}")
        rocket = Rocket(r["dry_mass"], r["drag_csv"], r["dim"])
        rocket.add_motor(motor)

        parachutes = [
            Parachute(p["name"], p["cd_s"], p["lag"], p["trigger"])
            for p in cfg["parachutes"]
        ]
        for p in parachutes:
            rocket.add_parachute(p)

        s = cfg["sim"]
        sim_kwargs = {
            "rail_length": s["rail_length"],
            "inclination": s["inclination"],
            "heading": s["heading"],
        }
        run_kwargs: dict = {"terminate_on": s["terminate_on"], "method": s["method"]}
        if s["method"] == "RK45":
            run_kwargs.update({"rtol": s["rtol"], "atol": s["atol"],
                               "max_step": s["max_step"], "t_max": s["t_max"]})
        else:
            run_kwargs.update({"dt": s["dt"], "t_max": s["t_max"]})

        return env, rocket, sim_kwargs, run_kwargs

    def run(self) -> None:
        try:
            env, rocket, sim_kwargs, run_kwargs = self._build_objects()
        except Exception as exc:
            self.run_error.emit(str(exc))
            return

        try:
            sim = Simulation(env, rocket, **sim_kwargs)
        except Exception as exc:
            self.run_error.emit(str(exc))
            return

        # --- stop hook ---
        original_rhs = sim._ode_rhs
        worker = self

        def _stoppable_rhs(t, state):
            if worker._stop_requested:
                raise InterruptedError("stopped by user")
            return original_rhs(t, state)

        sim._ode_rhs = _stoppable_rhs

        # --- live-data hook ---
        original_event_check = sim._event_check
        step_count = [0]
        known_events: set[str] = set()
        t_buf: list[float] = []
        z_buf: list[float] = []
        speed_buf: list[float] = []
        x_buf: list[float] = []
        y_buf: list[float] = []

        def _hooked_event_check(t, t_prev, state, state_prev):
            original_event_check(t, t_prev, state, state_prev)
            t_buf.append(t)
            z_buf.append(float(state[2]))
            speed_buf.append(float(np.linalg.norm(state[3:6])))
            x_buf.append(float(state[0]))
            y_buf.append(float(state[1]))
            step_count[0] += 1
            for ename, edata in sim.events.items():
                if edata is not None and ename not in known_events:
                    known_events.add(ename)
                    worker.event_fired.emit(ename, float(edata[0]))
            if step_count[0] % UPDATE_EVERY == 0:
                worker.step_update.emit({
                    "t": list(t_buf), "z": list(z_buf),
                    "speed": list(speed_buf),
                    "x": list(x_buf), "y": list(y_buf),
                })

        sim._event_check = _hooked_event_check

        try:
            flight = sim.run(**run_kwargs)
        except InterruptedError:
            self.run_error.emit("Simulation stopped by user.")
            return
        except Exception as exc:
            self.run_error.emit(str(exc))
            return

        # final emit with all data
        self.step_update.emit({
            "t": list(t_buf), "z": list(z_buf),
            "speed": list(speed_buf), "x": list(x_buf), "y": list(y_buf),
        })
        self.run_finished.emit((flight, sim))


# ---------------------------------------------------------------------------
# Tab: Environment
# ---------------------------------------------------------------------------

class EnvironmentTab(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        self.use_api_cb = QCheckBox("Use Windy API (live weather)")
        root.addWidget(self.use_api_cb)

        self.api_group = QGroupBox("Windy API Configuration")
        form = QFormLayout(self.api_group)
        self.api_key = QLineEdit()
        self.api_key.setPlaceholderText("Windy API key")
        self.lat = QDoubleSpinBox(); self.lat.setRange(-90, 90); self.lat.setDecimals(6); self.lat.setValue(39.3897)
        self.lon = QDoubleSpinBox(); self.lon.setRange(-180, 180); self.lon.setDecimals(6); self.lon.setValue(-8.2890)
        self.model = QComboBox(); self.model.addItems(["gfs", "iconEu", "ecmwf", "nam", "icon"])
        self.model.setCurrentText("iconEu")
        self.date_edit = QDateEdit(QDate.currentDate().addDays(1))
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("dd/MM/yyyy")
        form.addRow("API Key:", self.api_key)
        form.addRow("Latitude:", self.lat)
        form.addRow("Longitude:", self.lon)
        form.addRow("Model:", self.model)
        form.addRow("Date:", self.date_edit)
        root.addWidget(self.api_group)

        note = QLabel("When Windy API is disabled, ISA standard atmosphere is used.")
        note.setStyleSheet("color: #888; font-style: italic;")
        root.addWidget(note)
        root.addStretch()

        self.api_group.setVisible(False)
        self.use_api_cb.toggled.connect(self.api_group.setVisible)

    def get_config(self) -> dict:
        d = self.date_edit.date()
        return {
            "use_api": self.use_api_cb.isChecked(),
            "api_key": self.api_key.text().strip(),
            "lat": self.lat.value(),
            "lon": self.lon.value(),
            "model": self.model.currentText(),
            "day": d.day(),
            "month": d.month(),
            "year": d.year(),
        }


# ---------------------------------------------------------------------------
# Tab: Motor
# ---------------------------------------------------------------------------

class MotorTab(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        src_box = QGroupBox("Thrust Source")
        src_layout = QVBoxLayout(src_box)
        self.rb_csv = QRadioButton("CSV File")
        self.rb_log = QRadioButton("Logarithmic Curve")
        self.rb_csv.setChecked(True)
        src_layout.addWidget(self.rb_csv)

        self.csv_widget = QWidget()
        csv_row = QHBoxLayout(self.csv_widget)
        csv_row.setContentsMargins(20, 0, 0, 0)
        self.thrust_csv = QLineEdit("curves/thrust(2).csv")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_thrust)
        csv_row.addWidget(QLabel("File:"))
        csv_row.addWidget(self.thrust_csv)
        csv_row.addWidget(btn_browse)
        src_layout.addWidget(self.csv_widget)

        src_layout.addWidget(self.rb_log)
        self.log_widget = QWidget()
        log_form = QFormLayout(self.log_widget)
        log_form.setContentsMargins(20, 0, 0, 0)
        self.peak_thrust = QDoubleSpinBox(); self.peak_thrust.setRange(0, 1e6); self.peak_thrust.setValue(4000); self.peak_thrust.setSuffix(" N")
        self.ramp_time = QDoubleSpinBox(); self.ramp_time.setRange(0, 10); self.ramp_time.setDecimals(3); self.ramp_time.setValue(0.2); self.ramp_time.setSuffix(" s")
        log_form.addRow("Peak Thrust:", self.peak_thrust)
        log_form.addRow("Ramp Time:", self.ramp_time)
        self.log_widget.setVisible(False)
        src_layout.addWidget(self.log_widget)

        self.rb_csv.toggled.connect(lambda on: (self.csv_widget.setVisible(on), self.log_widget.setVisible(not on)))
        root.addWidget(src_box)

        prop_box = QGroupBox("Propellant")
        prop_form = QFormLayout(prop_box)
        self.burn_time = QDoubleSpinBox(); self.burn_time.setRange(0.01, 1000); self.burn_time.setDecimals(3); self.burn_time.setValue(4.2); self.burn_time.setSuffix(" s")
        self.ox_mdot = QDoubleSpinBox(); self.ox_mdot.setRange(0, 1000); self.ox_mdot.setDecimals(4); self.ox_mdot.setValue(1.5); self.ox_mdot.setSuffix(" kg/s")
        self.ox_mass = QDoubleSpinBox(); self.ox_mass.setRange(0, 10000); self.ox_mass.setDecimals(3); self.ox_mass.setValue(7.33); self.ox_mass.setSuffix(" kg")
        self.grain_mass = QDoubleSpinBox(); self.grain_mass.setRange(0.001, 10000); self.grain_mass.setDecimals(3); self.grain_mass.setValue(3.0); self.grain_mass.setSuffix(" kg")
        self.type_label = QLabel("Type: Hybrid")
        self.type_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        prop_form.addRow("Burn Time:", self.burn_time)
        prop_form.addRow("Ox Mass Flow (ox_mdot):", self.ox_mdot)
        prop_form.addRow("Initial Ox Mass:", self.ox_mass)
        prop_form.addRow("Initial Grain Mass:", self.grain_mass)
        prop_form.addRow("", self.type_label)
        self.ox_mass.valueChanged.connect(self._update_type_label)
        root.addWidget(prop_box)
        root.addStretch()

    def _browse_thrust(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Thrust CSV", "curves", "CSV (*.csv)")
        if path:
            self.thrust_csv.setText(path)

    def _update_type_label(self, val: float):
        self.type_label.setText("Type: " + ("Hybrid" if val > 0 else "Solid"))

    def get_config(self) -> dict:
        return {
            "thrust_type": "csv" if self.rb_csv.isChecked() else "log",
            "csv_path": self.thrust_csv.text().strip(),
            "peak_thrust": self.peak_thrust.value(),
            "ramp_time": self.ramp_time.value(),
            "burn_time": self.burn_time.value(),
            "ox_mdot": self.ox_mdot.value(),
            "initial_ox_mass": self.ox_mass.value(),
            "initial_grain_mass": self.grain_mass.value(),
        }


# ---------------------------------------------------------------------------
# Tab: Rocket
# ---------------------------------------------------------------------------

class RocketTab(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        box = QGroupBox("Rocket Properties")
        form = QFormLayout(box)
        self.dry_mass = QDoubleSpinBox(); self.dry_mass.setRange(0.1, 100000); self.dry_mass.setDecimals(3); self.dry_mass.setValue(40.8); self.dry_mass.setSuffix(" kg")
        self.dim = QDoubleSpinBox(); self.dim.setRange(0.001, 100); self.dim.setDecimals(4); self.dim.setValue(0.163); self.dim.setSuffix(" m")
        drag_row = QWidget()
        drag_layout = QHBoxLayout(drag_row)
        drag_layout.setContentsMargins(0, 0, 0, 0)
        self.drag_csv = QLineEdit("curves/MaCd.csv")
        btn = QPushButton("Browse…"); btn.clicked.connect(self._browse_drag)
        drag_layout.addWidget(self.drag_csv); drag_layout.addWidget(btn)
        form.addRow("Dry Mass:", self.dry_mass)
        form.addRow("Body Diameter:", self.dim)
        form.addRow("Drag Curve CSV:", drag_row)

        ref_area_label = QLabel()
        ref_area_label.setStyleSheet("color: #888; font-style: italic;")
        form.addRow("Ref. Area:", ref_area_label)
        def _update_area(v):
            ref_area_label.setText(f"{np.pi * (v / 2) ** 2:.6f} m²")
        self.dim.valueChanged.connect(_update_area)
        _update_area(self.dim.value())

        root.addWidget(box)
        root.addStretch()

    def _browse_drag(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Drag CSV", "curves", "CSV (*.csv)")
        if path:
            self.drag_csv.setText(path)

    def get_config(self) -> dict:
        return {
            "dry_mass": self.dry_mass.value(),
            "dim": self.dim.value(),
            "drag_csv": self.drag_csv.text().strip(),
        }


# ---------------------------------------------------------------------------
# Tab: Parachutes
# ---------------------------------------------------------------------------

class ParachutesTab(QWidget):
    _defaults = [
        {"name": "drogue", "cd_s": 0.7354, "lag": 1.0, "trigger": "apogee"},
        {"name": "main",   "cd_s": 13.8991, "lag": 1.0, "trigger": 450.0},
    ]

    def __init__(self):
        super().__init__()
        self._chutes: list[dict] = [dict(d) for d in self._defaults]
        self._selected: Optional[int] = None

        root = QHBoxLayout(self)

        # --- left panel ---
        left = QWidget(); left.setMaximumWidth(260)
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Parachutes:"))
        self.lst = QListWidget()
        self.lst.currentRowChanged.connect(self._on_select)
        lv.addWidget(self.lst)
        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add"); btn_add.clicked.connect(self._add)
        btn_rem = QPushButton("Remove"); btn_rem.clicked.connect(self._remove)
        btn_row.addWidget(btn_add); btn_row.addWidget(btn_rem)
        lv.addLayout(btn_row)
        root.addWidget(left)

        # --- right edit panel ---
        self.edit_box = QGroupBox("Parachute Configuration")
        form = QFormLayout(self.edit_box)
        self.e_name = QLineEdit()
        self.e_cds = QDoubleSpinBox(); self.e_cds.setRange(0.0001, 10000); self.e_cds.setDecimals(4); self.e_cds.setSuffix(" m²")
        self.e_lag = QDoubleSpinBox(); self.e_lag.setRange(0, 100); self.e_lag.setDecimals(3); self.e_lag.setSuffix(" s")
        self.rb_apogee = QRadioButton("Apogee"); self.rb_apogee.setChecked(True)
        self.rb_alt = QRadioButton("Altitude (m)")
        self.e_alt = QDoubleSpinBox(); self.e_alt.setRange(0, 100000); self.e_alt.setDecimals(1); self.e_alt.setValue(450); self.e_alt.setEnabled(False)
        self.rb_alt.toggled.connect(self.e_alt.setEnabled)
        trig_row = QWidget(); tr = QHBoxLayout(trig_row); tr.setContentsMargins(0,0,0,0)
        tr.addWidget(self.rb_apogee); tr.addWidget(self.rb_alt); tr.addWidget(self.e_alt)
        form.addRow("Name:", self.e_name)
        form.addRow("CdS:", self.e_cds)
        form.addRow("Lag:", self.e_lag)
        form.addRow("Trigger:", trig_row)
        btn_apply = QPushButton("Apply Changes"); btn_apply.clicked.connect(self._apply)
        form.addRow("", btn_apply)
        root.addWidget(self.edit_box)
        self.edit_box.setEnabled(False)

        self._refresh_list()

    def _refresh_list(self):
        self.lst.clear()
        for c in self._chutes:
            trig = "apogee" if c["trigger"] == "apogee" else f"{c['trigger']} m"
            self.lst.addItem(f"{c['name']}  |  CdS={c['cd_s']:.4f}  |  lag={c['lag']}s  |  {trig}")

    def _on_select(self, row: int):
        if row < 0 or row >= len(self._chutes):
            self.edit_box.setEnabled(False); return
        self._selected = row
        c = self._chutes[row]
        self.e_name.setText(c["name"])
        self.e_cds.setValue(c["cd_s"])
        self.e_lag.setValue(c["lag"])
        if c["trigger"] == "apogee":
            self.rb_apogee.setChecked(True)
        else:
            self.rb_alt.setChecked(True)
            self.e_alt.setValue(float(c["trigger"]))
        self.edit_box.setEnabled(True)

    def _add(self):
        self._chutes.append({"name": f"chute{len(self._chutes)+1}", "cd_s": 1.0, "lag": 1.0, "trigger": "apogee"})
        self._refresh_list()
        self.lst.setCurrentRow(len(self._chutes) - 1)

    def _remove(self):
        row = self.lst.currentRow()
        if 0 <= row < len(self._chutes):
            self._chutes.pop(row)
            self._refresh_list()
            self.edit_box.setEnabled(False)

    def _apply(self):
        if self._selected is None: return
        trigger = "apogee" if self.rb_apogee.isChecked() else self.e_alt.value()
        self._chutes[self._selected] = {
            "name": self.e_name.text().strip() or f"chute{self._selected}",
            "cd_s": self.e_cds.value(),
            "lag": self.e_lag.value(),
            "trigger": trigger,
        }
        self._refresh_list()
        self.lst.setCurrentRow(self._selected)

    def get_config(self) -> list[dict]:
        return [dict(c) for c in self._chutes]


# ---------------------------------------------------------------------------
# Tab: Simulation Settings
# ---------------------------------------------------------------------------

class SimSettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        integ = QGroupBox("Integration")
        form = QFormLayout(integ)
        self.method = QComboBox(); self.method.addItems(["RK45", "RK4"])
        self.rtol = QDoubleSpinBox(); self.rtol.setRange(1e-12, 1); self.rtol.setDecimals(10); self.rtol.setValue(1e-6)
        self.atol = QDoubleSpinBox(); self.atol.setRange(1e-15, 1); self.atol.setDecimals(12); self.atol.setValue(1e-9)
        self.max_step = QDoubleSpinBox(); self.max_step.setRange(0, 1e9); self.max_step.setDecimals(3); self.max_step.setValue(0); self.max_step.setSpecialValueText("Auto (∞)")
        self.dt = QDoubleSpinBox(); self.dt.setRange(1e-5, 10); self.dt.setDecimals(5); self.dt.setValue(0.01); self.dt.setSuffix(" s")
        self.t_max = QDoubleSpinBox(); self.t_max.setRange(1, 1e6); self.t_max.setDecimals(1); self.t_max.setValue(1000); self.t_max.setSuffix(" s")
        self.terminate = QComboBox(); self.terminate.addItems(["impact", "apogee", "burn_out", "rail_departure"])

        self._rk45_fields: list[tuple] = []
        for label, widget in [("rtol:", self.rtol), ("atol:", self.atol), ("max_step:", self.max_step)]:
            row_w = QWidget()
            row_l = QFormLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.addRow(label, widget)
            form.addRow(row_w)
            self._rk45_fields.append(row_w)

        self._rk4_row = QWidget()
        rk4_l = QFormLayout(self._rk4_row)
        rk4_l.setContentsMargins(0, 0, 0, 0)
        rk4_l.addRow("dt:", self.dt)
        form.addRow(self._rk4_row)
        self._rk4_row.setVisible(False)

        form.insertRow(0, "Method:", self.method)
        form.addRow("t_max:", self.t_max)
        form.addRow("Terminate on:", self.terminate)
        self.method.currentTextChanged.connect(self._on_method_change)
        root.addWidget(integ)

        rail = QGroupBox("Launch Rail")
        rail_form = QFormLayout(rail)
        self.rail_length = QDoubleSpinBox(); self.rail_length.setRange(0.1, 1000); self.rail_length.setDecimals(2); self.rail_length.setValue(12); self.rail_length.setSuffix(" m")
        self.inclination = QDoubleSpinBox(); self.inclination.setRange(0, 90); self.inclination.setDecimals(2); self.inclination.setValue(84); self.inclination.setSuffix(" °")
        self.heading = QDoubleSpinBox(); self.heading.setRange(0, 360); self.heading.setDecimals(2); self.heading.setValue(144); self.heading.setSuffix(" °")
        rail_form.addRow("Rail Length:", self.rail_length)
        rail_form.addRow("Inclination:", self.inclination)
        rail_form.addRow("Heading:", self.heading)
        root.addWidget(rail)
        root.addStretch()

    def _on_method_change(self, method: str):
        for w in self._rk45_fields:
            w.setVisible(method == "RK45")
        self._rk4_row.setVisible(method == "RK4")

    def get_config(self) -> dict:
        max_step_val = self.max_step.value()
        return {
            "method": self.method.currentText(),
            "rtol": self.rtol.value(),
            "atol": self.atol.value(),
            "max_step": np.inf if max_step_val == 0 else max_step_val,
            "dt": self.dt.value(),
            "t_max": self.t_max.value(),
            "terminate_on": self.terminate.currentText(),
            "rail_length": self.rail_length.value(),
            "inclination": self.inclination.value(),
            "heading": self.heading.value(),
        }


# ---------------------------------------------------------------------------
# Tab: Run & Monitor
# ---------------------------------------------------------------------------

class RunTab(QWidget):
    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self)

        # control bar
        bar = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run Simulation")
        self.run_btn.setStyleSheet("font-size: 13px; font-weight: bold; padding: 6px 20px; background: #1a6e1a; border-color: #2a8e2a;")
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-size: 13px; padding: 6px 16px;")
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #888; margin-left: 12px;")
        bar.addWidget(self.run_btn)
        bar.addWidget(self.stop_btn)
        bar.addWidget(self.status_lbl)
        bar.addStretch()
        root.addLayout(bar)

        splitter = QSplitter(Qt.Horizontal)

        # left: altitude + velocity
        left_split = QSplitter(Qt.Vertical)
        self.alt_canvas = LiveCanvas2D("Time (s)", "Altitude (m)", "Altitude vs Time", "#4ec9b0")
        self.vel_canvas = LiveCanvas2D("Time (s)", "Speed (m/s)", "Speed vs Time", "#f59a42")
        left_split.addWidget(self.alt_canvas)
        left_split.addWidget(self.vel_canvas)
        left_split.setSizes([300, 300])
        splitter.addWidget(left_split)

        # right: 3D trajectory + event log
        right_split = QSplitter(Qt.Vertical)
        self.traj_canvas = LiveCanvas3D()
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setFont(QFont("Monospace", 8))
        self.event_log.setMaximumHeight(180)
        right_split.addWidget(self.traj_canvas)
        right_split.addWidget(self.event_log)
        right_split.setSizes([400, 180])
        splitter.addWidget(right_split)
        splitter.setSizes([600, 600])
        root.addWidget(splitter)

    def append_log(self, msg: str) -> None:
        self.event_log.append(msg)
        self.event_log.verticalScrollBar().setValue(self.event_log.verticalScrollBar().maximum())

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.status_lbl.setText("Running…" if running else "Ready")
        self.status_lbl.setStyleSheet("color: #4ec9b0; margin-left: 12px;" if running else "color: #888; margin-left: 12px;")

    def clear_plots(self) -> None:
        self.alt_canvas.clear_plot()
        self.vel_canvas.clear_plot()
        self.traj_canvas.clear_plot()
        self.event_log.clear()

    def on_step_update(self, data: dict) -> None:
        t, z = data["t"], data["z"]
        self.alt_canvas.update_data(t, z)
        self.vel_canvas.update_data(t, data["speed"])
        self.traj_canvas.update_data(data["x"], data["y"], z)

    def on_event_fired(self, name: str, t: float) -> None:
        color = EVENT_COLORS.get(name, "#ffffff")
        self.event_log.append(
            f'<span style="color:{color}; font-weight:bold;">[EVENT]</span>'
            f' <span style="color:#d4d4d4;">{name}</span>'
            f' <span style="color:#888;">at t = {t:.3f} s</span>'
        )
        self.alt_canvas.add_vline(name, t)
        self.vel_canvas.add_vline(name, t)


# ---------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------

class ResultsTab(QWidget):
    _METRIC_NAMES = [
        "Apogee (m)", "Apogee Time (s)", "Max Speed (m/s)", "Max Mach",
        "Max Acceleration (m/s²)", "Out-of-rail Velocity (m/s)",
        "Impact Time (s)", "Impact Range (m)",
    ]

    def __init__(self):
        super().__init__()
        self._flight = None

        splitter = QSplitter(Qt.Vertical)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # --- top pane: metrics table ---
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(6, 6, 6, 4)
        tl.addWidget(QLabel("Key Metrics"))
        self.metrics_table = QTableWidget(len(self._METRIC_NAMES), 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        for i, name in enumerate(self._METRIC_NAMES):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem("—"))
        tl.addWidget(self.metrics_table)
        splitter.addWidget(top)

        # --- bottom pane: plot sub-tabs ---
        bottom_tabs = QTabWidget()

        plot_widget = QWidget()
        pl = QVBoxLayout(plot_widget)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("X:"))
        self.x_combo = QComboBox()
        self.x_combo.addItems(CHANNELS)
        self.x_combo.setCurrentText("t")
        ctrl.addWidget(self.x_combo)
        ctrl.addWidget(QLabel("Y:"))
        self.y_combo = QComboBox()
        self.y_combo.addItems(CHANNELS)
        self.y_combo.setCurrentText("z")
        ctrl.addWidget(self.y_combo)
        self.plot_btn = QPushButton("Plot")
        self.plot_btn.clicked.connect(self._replot)
        ctrl.addWidget(self.plot_btn)
        ctrl.addStretch()
        pl.addLayout(ctrl)
        self.custom_canvas = LiveCanvas2D("x", "y", "Custom Plot", "#4a9eff")
        pl.addWidget(self.custom_canvas)
        bottom_tabs.addTab(plot_widget, "Plot")

        traj_widget = QWidget()
        tw = QVBoxLayout(traj_widget)
        self.traj_canvas = LiveCanvas3D()
        tw.addWidget(self.traj_canvas)
        bottom_tabs.addTab(traj_widget, "3D Trajectory")

        splitter.addWidget(bottom_tabs)
        splitter.setSizes([220, 600])

    def populate(self, flight, sim) -> None:
        self._flight = flight
        apogee = sim.linear_params.get("apogee") or float(np.max(flight.z))
        apogee_t_event = sim.events.get("apogee")
        apogee_t = (
            apogee_t_event[0] if apogee_t_event
            else float(flight.t[int(np.argmax(flight.z))])
        )
        v_rail = sim.linear_params.get("out_of_rail_velocity")
        impact_event = sim.events.get("impact")
        impact_t = impact_event[0] if impact_event else float(flight.t[-1])

        values = [
            apogee,
            apogee_t,
            float(np.max(flight.speed)),
            float(np.max(flight.mach)),
            float(np.max(flight.acceleration)),
            v_rail if v_rail is not None else float("nan"),
            impact_t,
            float(np.hypot(flight.x[-1], flight.y[-1])),
        ]
        for i, v in enumerate(values):
            item = QTableWidgetItem(f"{v:.3f}" if v == v else "—")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.metrics_table.setItem(i, 1, item)

        self._replot()
        self.traj_canvas.plot_final(flight.x, flight.y, flight.z)

    def _replot(self) -> None:
        if self._flight is None:
            return
        xc = self.x_combo.currentText()
        yc = self.y_combo.currentText()
        xs = getattr(self._flight, xc)
        ys = getattr(self._flight, yc)
        ax = self.custom_canvas.ax
        ax.cla()
        ax.set_facecolor("#2d2d2d")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors="#888888", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.grid(True, alpha=0.2, color="#555555")
        ax.plot(xs, ys, color="#4a9eff", linewidth=1.5)
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)
        ax.set_title(f"{yc} vs {xc}", color="#d4d4d4")
        # keep line attr in sync so LiveCanvas2D.update_data still works if called
        self.custom_canvas.line, = ax.plot([], [], alpha=0)
        self.custom_canvas.draw_idle()


# ---------------------------------------------------------------------------
# Campaign helpers
# ---------------------------------------------------------------------------

_SWEEP_PATHS = [
    # Rocket structure
    "rocket.dry_mass",
    "rocket.ref_area",
    # Motor
    "rocket.motor.burn_time",
    "rocket.motor.initial_grain_mass",
    "rocket.motor.initial_ox_mass",
    "rocket.motor.ox_mdot",
    # Launch parameters (routed via sim.* → sim_kwargs)
    "sim.inclination",
    "sim.heading",
    "sim.rail_length",
    # Wind — constant surface speed in m/s (sets env.wind_profile)
    "env.wind_u",
    "env.wind_v",
]

_CAMPAIGN_METRICS = [
    "apogee_m", "apogee_t", "max_speed_ms", "max_mach",
    "max_accel_ms2", "final_t", "final_range_m",
]


def _parse_sweep_values(text: str) -> list[float]:
    text = text.strip()
    if text.startswith("linspace("):
        inner = text[9:].rstrip(")")
        parts = [float(x) for x in inner.split(",")]
        return list(np.linspace(*parts))
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _build_base_objects(cfg: dict):
    """Build Environment + Rocket (with motor and parachutes) from a config dict."""
    env = Environment()
    if cfg["env"]["use_api"]:
        if not cfg["env"]["api_key"]:
            raise ValueError("Windy API key is empty.")
        env.set_model(
            cfg["env"]["api_key"],
            cfg["env"]["lat"],
            cfg["env"]["lon"],
            cfg["env"]["model"],
            (cfg["env"]["day"], cfg["env"]["month"], cfg["env"]["year"]),
        )

    m = cfg["motor"]
    if m["thrust_type"] == "csv":
        if not os.path.isfile(m["csv_path"]):
            raise ValueError(f"Thrust CSV not found: {m['csv_path']}")
        thrust_src = m["csv_path"]
    else:
        thrust_src = logarithmic_thrust(m["burn_time"], m["peak_thrust"], m["ramp_time"])

    motor = Motor(
        thrust_src,
        m["burn_time"],
        ox_mdot=m["ox_mdot"],
        initial_ox_mass=m["initial_ox_mass"],
        initial_grain_mass=m["initial_grain_mass"],
    )

    r = cfg["rocket"]
    if not os.path.isfile(r["drag_csv"]):
        raise ValueError(f"Drag CSV not found: {r['drag_csv']}")
    rocket = Rocket(r["dry_mass"], r["drag_csv"], r["dim"])
    rocket.add_motor(motor)
    for p in cfg["parachutes"]:
        rocket.add_parachute(Parachute(p["name"], p["cd_s"], p["lag"], p["trigger"]))

    return env, rocket


# ---------------------------------------------------------------------------
# SweepRowWidget
# ---------------------------------------------------------------------------

class SweepRowWidget(QWidget):
    removed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)

        self.path_combo = QComboBox()
        self.path_combo.setEditable(True)
        self.path_combo.addItems(_SWEEP_PATHS)
        self.path_combo.setMinimumWidth(220)

        self.type_combo = QComboBox()
        self.type_combo.addItems(["Sweep", "Normal", "Uniform"])

        self._stack = QStackedWidget()

        # page 0 — Sweep values
        sweep_w = QWidget()
        sweep_l = QHBoxLayout(sweep_w)
        sweep_l.setContentsMargins(0, 0, 0, 0)
        self.sweep_edit = QLineEdit()
        self.sweep_edit.setPlaceholderText("e.g. 38, 40, 42  or  linspace(35,50,8)")
        sweep_l.addWidget(self.sweep_edit)
        self._stack.addWidget(sweep_w)

        # page 1 — Normal(mu, sigma)
        norm_w = QWidget()
        norm_l = QHBoxLayout(norm_w)
        norm_l.setContentsMargins(0, 0, 0, 0)
        self.norm_mu = QDoubleSpinBox()
        self.norm_mu.setRange(-1e9, 1e9)
        self.norm_mu.setDecimals(4)
        self.norm_mu.setPrefix("μ ")
        self.norm_sigma = QDoubleSpinBox()
        self.norm_sigma.setRange(0, 1e9)
        self.norm_sigma.setDecimals(4)
        self.norm_sigma.setPrefix("σ ")
        norm_l.addWidget(self.norm_mu)
        norm_l.addWidget(self.norm_sigma)
        self._stack.addWidget(norm_w)

        # page 2 — Uniform(lo, hi)
        uni_w = QWidget()
        uni_l = QHBoxLayout(uni_w)
        uni_l.setContentsMargins(0, 0, 0, 0)
        self.uni_lo = QDoubleSpinBox()
        self.uni_lo.setRange(-1e9, 1e9)
        self.uni_lo.setDecimals(4)
        self.uni_lo.setPrefix("lo ")
        self.uni_hi = QDoubleSpinBox()
        self.uni_hi.setRange(-1e9, 1e9)
        self.uni_hi.setDecimals(4)
        self.uni_hi.setPrefix("hi ")
        uni_l.addWidget(self.uni_lo)
        uni_l.addWidget(self.uni_hi)
        self._stack.addWidget(uni_w)

        self.type_combo.currentIndexChanged.connect(self._stack.setCurrentIndex)

        btn_del = QPushButton("×")
        btn_del.setFixedWidth(28)
        btn_del.setStyleSheet("QPushButton { color: #f44747; font-weight: bold; }")
        btn_del.clicked.connect(lambda: self.removed.emit(self))

        row.addWidget(self.path_combo)
        row.addWidget(self.type_combo)
        row.addWidget(self._stack)
        row.addWidget(btn_del)

    def get_param(self) -> tuple[str, object]:
        path = self.path_combo.currentText().strip()
        if not path:
            raise ValueError("Sweep path is empty.")
        kind = self.type_combo.currentText()
        if kind == "Sweep":
            vals = _parse_sweep_values(self.sweep_edit.text())
            if not vals:
                raise ValueError(f"No values for path '{path}'.")
            return path, Param.sweep(vals)
        if kind == "Normal":
            return path, Param.normal(self.norm_mu.value(), self.norm_sigma.value())
        # Uniform
        lo, hi = self.uni_lo.value(), self.uni_hi.value()
        if hi <= lo:
            raise ValueError(f"Uniform hi must be > lo for path '{path}'.")
        return path, Param.uniform(lo, hi)


# ---------------------------------------------------------------------------
# CampaignWorker
# ---------------------------------------------------------------------------

class CampaignWorker(QThread):
    progress = Signal(int, int)
    run_finished = Signal(object)
    run_error = Signal(str)

    def __init__(self, campaign: Campaign, n_workers: int) -> None:
        super().__init__()
        self._campaign = campaign
        self._n_workers = n_workers

    def run(self) -> None:
        specs = self._campaign.specs
        total = len(specs)
        base = BaseObjects(
            env=self._campaign.environment,
            rocket=self._campaign.rocket,
        )
        try:
            self.progress.emit(0, total)
            if self._n_workers <= 1:
                # Sequential — emit progress after every run
                pairs: list = []
                for i, spec in enumerate(specs):
                    pairs.append(execute_run(base, spec))
                    self.progress.emit(i + 1, total)
                pairs.sort(key=lambda r: specs.index(r[0]))
                self.run_finished.emit(CampaignResults(pairs))
            else:
                # Parallel via campaign.run — progress only at finish
                results = self._campaign.run(
                    n_workers=self._n_workers, show_progress=False
                )
                self.progress.emit(total, total)
                self.run_finished.emit(results)
        except Exception as exc:
            self.run_error.emit(str(exc))


# ---------------------------------------------------------------------------
# Tab: Campaign
# ---------------------------------------------------------------------------

class CampaignTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._sweep_rows: list[SweepRowWidget] = []
        self._results = None
        self._worker: Optional[CampaignWorker] = None

        splitter = QSplitter(Qt.Vertical)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # ---- top: configuration ----
        cfg_widget = QWidget()
        cfg_widget.setMinimumHeight(260)
        cfg_layout = QVBoxLayout(cfg_widget)
        cfg_layout.setContentsMargins(6, 6, 6, 4)

        # sweep rows area
        sweep_box = QGroupBox("Sweep Parameters")
        sweep_vl = QVBoxLayout(sweep_box)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(180)
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setAlignment(Qt.AlignTop)
        self._rows_layout.setSpacing(2)
        scroll.setWidget(self._rows_container)
        sweep_vl.addWidget(scroll)
        btn_add_row = QPushButton("+ Add Parameter")
        btn_add_row.clicked.connect(self._add_row)
        sweep_vl.addWidget(btn_add_row)
        cfg_layout.addWidget(sweep_box)

        # campaign options
        opts_box = QGroupBox("Campaign Options")
        opts_form = QFormLayout(opts_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["grid", "zip", "random", "lhs"])
        self.n_runs_spin = QSpinBox()
        self.n_runs_spin.setRange(2, 10000)
        self.n_runs_spin.setValue(50)
        self.n_runs_spin.setEnabled(False)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(1)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(0)
        self.seed_spin.setSpecialValueText("No seed")
        self.label_edit = QLineEdit("campaign")
        opts_form.addRow("Mode:", self.mode_combo)
        opts_form.addRow("N runs (random/lhs):", self.n_runs_spin)
        opts_form.addRow("Workers:", self.workers_spin)
        opts_form.addRow("Seed:", self.seed_spin)
        opts_form.addRow("Label:", self.label_edit)
        self.mode_combo.currentTextChanged.connect(
            lambda m: self.n_runs_spin.setEnabled(m in ("random", "lhs"))
        )
        cfg_layout.addWidget(opts_box)

        # run bar
        run_bar = QHBoxLayout()
        self.run_btn = QPushButton("Run Campaign")
        self.run_btn.setStyleSheet(
            "font-weight: bold; padding: 5px 16px; background: #1a4a8a; border-color: #2a6abf;"
        )
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #888; margin-left: 8px;")
        run_bar.addWidget(self.run_btn)
        run_bar.addWidget(self.progress_bar, 1)
        run_bar.addWidget(self.status_lbl)
        cfg_layout.addLayout(run_bar)
        splitter.addWidget(cfg_widget)

        # ---- bottom: results ----
        results_widget = QWidget()
        res_layout = QVBoxLayout(results_widget)
        res_layout.setContentsMargins(6, 4, 6, 6)

        # summary table
        summary_box = QGroupBox("Summary")
        summary_vl = QVBoxLayout(summary_box)
        self.summary_table = QTableWidget(0, 0)
        self.summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        summary_vl.addWidget(self.summary_table)
        res_layout.addWidget(summary_box)

        # analysis plots
        analysis_box = QGroupBox("Analysis")
        analysis_vl = QVBoxLayout(analysis_box)
        plot_ctrl = QHBoxLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(
            ["Envelope", "Landing Scatter", "Apogee Histogram", "Sensitivity Tornado"]
        )
        self.plot_channel_combo = QComboBox()
        self.plot_channel_combo.addItems(CHANNELS)
        self.plot_channel_combo.setCurrentText("z")
        self.plot_metric_combo = QComboBox()
        self.plot_metric_combo.addItems(_CAMPAIGN_METRICS)
        self.plot_metric_combo.hide()
        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self._do_analysis_plot)
        plot_ctrl.addWidget(QLabel("Plot:"))
        plot_ctrl.addWidget(self.plot_type_combo)
        plot_ctrl.addWidget(self.plot_channel_combo)
        plot_ctrl.addWidget(self.plot_metric_combo)
        plot_ctrl.addWidget(btn_plot)
        plot_ctrl.addStretch()
        self.plot_type_combo.currentTextChanged.connect(self._on_plot_type_change)
        analysis_vl.addLayout(plot_ctrl)

        fig = Figure(facecolor="#1e1e1e", tight_layout=True)
        self.campaign_canvas = FigureCanvasQTAgg(fig)
        self.campaign_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._camp_fig = fig
        analysis_vl.addWidget(self.campaign_canvas)
        res_layout.addWidget(analysis_box)
        splitter.addWidget(results_widget)
        splitter.setSizes([300, 600])

    def _add_row(self) -> None:
        row = SweepRowWidget()
        row.removed.connect(self._remove_row)
        self._sweep_rows.append(row)
        self._rows_layout.addWidget(row)

    def _remove_row(self, row: SweepRowWidget) -> None:
        self._sweep_rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()

    def _on_plot_type_change(self, ptype: str) -> None:
        self.plot_channel_combo.setVisible(ptype == "Envelope")
        self.plot_metric_combo.setVisible(ptype == "Sensitivity Tornado")

    def build_campaign(self, base_cfg: dict) -> Campaign:
        env, rocket = _build_base_objects(base_cfg)
        s = base_cfg["sim"]
        sim_kwargs = {
            "rail_length": s["rail_length"],
            "inclination": s["inclination"],
            "heading": s["heading"],
        }
        run_kwargs: dict = {"terminate_on": s["terminate_on"], "method": s["method"]}
        if s["method"] == "RK45":
            run_kwargs.update(
                {"rtol": s["rtol"], "atol": s["atol"],
                 "max_step": s["max_step"], "t_max": s["t_max"]}
            )
        else:
            run_kwargs.update({"dt": s["dt"], "t_max": s["t_max"]})

        camp = Campaign(
            env, rocket,
            sim_kwargs=sim_kwargs,
            run_kwargs=run_kwargs,
            label=self.label_edit.text().strip() or "campaign",
        )

        if not self._sweep_rows:
            raise ValueError("Add at least one sweep parameter before running a campaign.")

        params = {}
        for row in self._sweep_rows:
            path, param = row.get_param()
            params[path] = param

        mode = self.mode_combo.currentText()
        seed_val = self.seed_spin.value()
        seed = seed_val if seed_val > 0 else None
        n = self.n_runs_spin.value() if mode in ("random", "lhs") else None
        camp.sweep_multiple(params, mode=mode, n=n, seed=seed)
        return camp

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.status_lbl.setText("Running…" if running else "Ready")
        self.status_lbl.setStyleSheet(
            "color: #4ec9b0; margin-left: 8px;"
            if running else "color: #888; margin-left: 8px;"
        )

    def on_progress(self, done: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.status_lbl.setText(f"{done}/{total}")

    def populate_results(self, results) -> None:
        self._results = results
        df = results.summary()
        self.summary_table.setRowCount(len(df))
        self.summary_table.setColumnCount(len(df.columns))
        self.summary_table.setHorizontalHeaderLabels(list(df.columns))
        for i, row in df.iterrows():
            for j, val in enumerate(row):
                text = f"{val:.4g}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(i, j, item)

    def _do_analysis_plot(self) -> None:
        if self._results is None:
            QMessageBox.information(self, "No Results", "Run a campaign first.")
            return
        ptype = self.plot_type_combo.currentText()
        self._camp_fig.clear()
        ax = self._camp_fig.add_subplot(111)
        ax.set_facecolor("#2d2d2d")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.tick_params(colors="#888888", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#d4d4d4")
        ax.grid(True, alpha=0.2, color="#555555")
        try:
            if ptype == "Envelope":
                self._results.plot_envelope(
                    channel=self.plot_channel_combo.currentText(),
                    x_channel="t",
                    ax=ax,
                )
            elif ptype == "Landing Scatter":
                landing_scatter(self._results, ax=ax)
            elif ptype == "Apogee Histogram":
                apogee_histogram(self._results, ax=ax)
            else:
                sensitivity_tornado(
                    self._results,
                    metric=self.plot_metric_combo.currentText(),
                    ax=ax,
                )
        except Exception as exc:
            ax.text(
                0.5, 0.5, str(exc),
                ha="center", va="center", color="#f44747",
                transform=ax.transAxes, wrap=True,
            )
        self.campaign_canvas.draw_idle()


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("flightForge  —  3DOF Simulator")
        self.resize(1400, 900)
        self._worker: Optional[SimulationWorker] = None
        self._campaign_worker: Optional[CampaignWorker] = None

        tabs = QTabWidget()
        self.env_tab = EnvironmentTab()
        self.motor_tab = MotorTab()
        self.rocket_tab = RocketTab()
        self.chutes_tab = ParachutesTab()
        self.sim_tab = SimSettingsTab()
        self.run_tab = RunTab()
        self.results_tab = ResultsTab()
        self.campaign_tab = CampaignTab()

        tabs.addTab(self.env_tab,      "Environment")
        tabs.addTab(self.motor_tab,    "Motor")
        tabs.addTab(self.rocket_tab,   "Rocket")
        tabs.addTab(self.chutes_tab,   "Parachutes")
        tabs.addTab(self.sim_tab,      "Simulation")
        tabs.addTab(self.run_tab,      "Run & Monitor")
        tabs.addTab(self.results_tab,  "Results")
        tabs.addTab(self.campaign_tab, "Campaign")
        self.setCentralWidget(tabs)
        self._tabs = tabs

        self.run_tab.run_btn.clicked.connect(self._on_run)
        self.run_tab.stop_btn.clicked.connect(self._on_stop)
        self.campaign_tab.run_btn.clicked.connect(self._on_campaign_run)

        self._log_handler = GUILogHandler()
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        
        # Connect the thread-safe signal to your UI update function
        self._log_handler.emitter.sig_log.connect(self.run_tab.append_log)
        
        logging.getLogger("flightForge").addHandler(self._log_handler)

    def _collect_config(self) -> dict:
        return {
            "env": self.env_tab.get_config(),
            "motor": self.motor_tab.get_config(),
            "rocket": self.rocket_tab.get_config(),
            "parachutes": self.chutes_tab.get_config(),
            "sim": self.sim_tab.get_config(),
        }

    def _on_run(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        try:
            config = self._collect_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Configuration Error", str(exc))
            return

        self.run_tab.clear_plots()
        self.run_tab.set_running(True)
        self._tabs.setCurrentWidget(self.run_tab)

        self._worker = SimulationWorker(config)
        self._worker.step_update.connect(self.run_tab.on_step_update)
        self._worker.event_fired.connect(self.run_tab.on_event_fired)
        self._worker.run_finished.connect(self._on_run_finished)
        self._worker.run_error.connect(self._on_run_error)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
            self.run_tab.status_lbl.setText("Stopping…")

    def _on_run_finished(self, payload) -> None:
        flight, sim = payload
        self.run_tab.set_running(False)
        self.run_tab.status_lbl.setText("Complete")
        self.run_tab.status_lbl.setStyleSheet("color: #4ec9b0; margin-left: 12px; font-weight: bold;")
        self.results_tab.populate(flight, sim)
        self._tabs.setCurrentWidget(self.results_tab)
        self.setWindowTitle(
            f"flightForge  —  Apogee: {sim.linear_params.get('apogee', 0):.0f} m  |  "
            f"Impact: {sim.events.get('impact', [0])[0]:.1f} s"
        )

    def _on_run_error(self, msg: str) -> None:
        self.run_tab.set_running(False)
        self.run_tab.status_lbl.setText("Error")
        self.run_tab.status_lbl.setStyleSheet("color: #f44747; margin-left: 12px;")
        QMessageBox.warning(self, "Simulation Error", msg)

    def _on_campaign_run(self) -> None:
        if self._campaign_worker and self._campaign_worker.isRunning():
            return
        try:
            base_cfg = self._collect_config()
            campaign = self.campaign_tab.build_campaign(base_cfg)
        except ValueError as exc:
            QMessageBox.critical(self, "Campaign Configuration Error", str(exc))
            return

        self.campaign_tab.set_running(True)
        self.campaign_tab.progress_bar.setValue(0)
        self._campaign_worker = CampaignWorker(campaign, self.campaign_tab.workers_spin.value())
        self._campaign_worker.progress.connect(self.campaign_tab.on_progress)
        self._campaign_worker.run_finished.connect(self._on_campaign_finished)
        self._campaign_worker.run_error.connect(self._on_campaign_error)
        self._campaign_worker.start()

    def _on_campaign_finished(self, results) -> None:
        self.campaign_tab.set_running(False)
        self.campaign_tab.status_lbl.setText(f"Done — {len(results)} runs")
        self.campaign_tab.status_lbl.setStyleSheet("color: #4ec9b0; margin-left: 8px; font-weight: bold;")
        self.campaign_tab.populate_results(results)

    def _on_campaign_error(self, msg: str) -> None:
        self.campaign_tab.set_running(False)
        self.campaign_tab.status_lbl.setText("Error")
        self.campaign_tab.status_lbl.setStyleSheet("color: #f44747; margin-left: 8px;")
        QMessageBox.warning(self, "Campaign Error", msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
