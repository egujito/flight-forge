from __future__ import annotations

import datetime
import time
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import requests

from .logger import logger


class Environment:
    """Atmospheric model supporting ISA, custom wind profiles, and Windy API data."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        model: str = "gfs",
        wind_u: Optional[Callable[[float], float]] = None,
        wind_v: Optional[Callable[[float], float]] = None,
        rho_profile: Optional[Callable[[float], float]] = None,
    ) -> None:
        """Initialise environment with optional API credentials and custom profiles.

        Args:
            api_key:     Windy.com API key. Triggers API fetch when provided with lat/lon.
            lat:         Launch site latitude in decimal degrees.
            lon:         Launch site longitude in decimal degrees.
            model:       Windy forecast model (e.g. 'gfs', 'iconEu').
            wind_u:      Optional callable f(h) -> east wind component in m/s.
            wind_v:      Optional callable f(h) -> north wind component in m/s.
            rho_profile: Optional callable f(h) -> air density in kg/m³.
        """
        self.g = 9.80665
        self.R = 287.05
        self.gamma = 1.4
        self.beta = 1.458e-6
        self.S = 110.4

        self.h_vals = np.array([0.0])
        self.rho_vals = np.array([1.225])
        self.u_vals = np.array([0.0])
        self.v_vals = np.array([0.0])

        self.lat = lat
        self.lon = lon
        self.model = "Default" if api_key is None else model

        self.wind_profile: Callable = self._build_wind_profile(wind_u, wind_v)
        self.rho_profile: Callable = rho_profile if rho_profile is not None else self._def_rho_profile

        if api_key and lat is not None and lon is not None:
            self.set_model(api_key, lat, lon, model)

    def set_model(
        self,
        api_key: str,
        lat: float,
        lon: float,
        model: str = "gfs",
        date: Optional[tuple[int, int, int]] = None,
    ) -> None:
        """Fetch atmospheric data from the Windy API and override wind/density profiles.

        Args:
            api_key: Windy.com API key.
            lat:     Latitude in decimal degrees.
            lon:     Longitude in decimal degrees.
            model:   Forecast model identifier.
            date:    Optional (day, month, year) tuple; defaults to current time.
        """
        self.lat = lat
        self.lon = lon
        self.model = model

        target_ts_ms: Optional[float] = None
        if date is not None:
            try:
                day, month, year = date
                dt_obj = datetime.datetime(
                    year, month, day, 12, 0, 0, tzinfo=datetime.timezone.utc
                )
                target_ts_ms = dt_obj.timestamp() * 1000
            except Exception as e:
                raise ValueError(f"Invalid date format: {e}") from e

        self._fetch_data(api_key, float(lat), float(lon), model, target_ts_ms)
        self.wind_profile = self._api_wind_profile
        self.rho_profile = self._api_rho_profile

        u_surf = self.u_vals[0]
        v_surf = self.v_vals[0]
        v_mag = float(np.sqrt(u_surf**2 + v_surf**2))
        logger.info("-------ENVIRONMENT INFO --------")
        logger.info(f"Coordinates:   {self.lat}, {self.lon}")
        logger.info(f"Model Used:    {self.model}")
        logger.info(f"Surface Wind:  U={u_surf:.2f} m/s, V={v_surf:.2f} m/s  Mag={v_mag:.2f} m/s")
        logger.info("--------------------------------")

    def density(self, h: float) -> float:
        """Return air density in kg/m³ at altitude h (metres)."""
        return self.rho_profile(h)

    def wind(self, h: float) -> tuple[float, float]:
        """Return (u, v) wind components in m/s at altitude h (metres)."""
        return self.wind_profile(h)

    def speed_of_sound(self, h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return speed of sound in m/s at altitude h using ISA temperature."""
        T = self._get_isa_temperature(h)
        return np.sqrt(self.gamma * self.R * T)

    def dynamic_viscosity(self, h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Return dynamic viscosity in Pa·s at altitude h using Sutherland's formula."""
        T = self._get_isa_temperature(h)
        return (self.beta * T**1.5) / (T + self.S)

    def _build_wind_profile(
        self,
        wind_u: Optional[Callable[[float], float]],
        wind_v: Optional[Callable[[float], float]],
    ) -> Callable[[float], tuple[float, float]]:
        if wind_u is None and wind_v is None:
            return self._def_wind_profile
        _u = wind_u if wind_u is not None else lambda _h: 0.0
        _v = wind_v if wind_v is not None else lambda _h: 0.0
        return lambda h: (float(_u(h)), float(_v(h)))

    def _get_isa_temperature(self, h: Union[float, np.ndarray]) -> np.ndarray:
        h = np.array(h, dtype=float)
        T0, L, T_trop, h_trop = 288.15, 0.0065, 216.65, 11000.0
        return np.where(h <= h_trop, T0 - L * h, T_trop)

    def _def_rho_profile(self, h: Union[float, np.ndarray]) -> np.ndarray:
        h = np.array(h, dtype=float)
        P0, T0, L = 101325.0, 288.15, 0.0065
        h_trop, T_trop, P_trop = 11000.0, 216.65, 22632.10
        T = self._get_isa_temperature(h)
        press_trop = P0 * (1 - (L * h) / T0) ** (self.g / (self.R * L))
        press_strat = P_trop * np.exp(-self.g * (h - h_trop) / (self.R * T_trop))
        P = np.where(h <= h_trop, press_trop, press_strat)
        return P / (self.R * T)

    @staticmethod
    def _def_wind_profile(h: Union[float, np.ndarray]) -> tuple:
        h = np.asarray(h)
        if np.ndim(h) > 0:
            return (np.zeros_like(h), np.zeros_like(h))
        return (0.0, 0.0)

    def _api_wind_profile(self, h: Union[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return (np.interp(h, self.h_vals, self.u_vals), np.interp(h, self.h_vals, self.v_vals))

    def _api_rho_profile(self, h: Union[float, np.ndarray]) -> np.ndarray:
        return np.interp(h, self.h_vals, self.rho_vals)

    def _fetch_data(
        self,
        key: str,
        lat: float,
        lon: float,
        model: str,
        target_ts: Optional[float] = None,
    ) -> None:
        levels = [
            "1000h", "950h", "925h", "900h", "850h", "800h",
            "700h", "600h", "500h", "400h", "300h", "200h", "150h",
        ]
        payload = {
            "lat": lat, "lon": lon, "model": model,
            "parameters": ["wind", "temp", "gh"],
            "levels": levels, "key": key,
        }
        try:
            response = requests.post(
                "https://api.windy.com/api/point-forecast/v2", json=payload
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(
                f"Windy API Error {response.status_code}: {response.text}"
            ) from e

        data = response.json()
        target_ts = target_ts if target_ts is not None else time.time() * 1000
        idx = int((np.abs(np.array(data["ts"]) - target_ts)).argmin())

        h_list: list[float] = []
        rho_list: list[float] = []
        u_list: list[float] = []
        v_list: list[float] = []

        for lvl in levels:
            val_h = data.get(f"gh-{lvl}", [None])[idx]
            val_temp = data.get(f"temp-{lvl}", [None])[idx]
            val_u = data.get(f"wind_u-{lvl}", [None])[idx]
            val_v = data.get(f"wind_v-{lvl}", [None])[idx]
            if all(v is not None for v in [val_h, val_temp, val_u, val_v]):
                pressure_pa = int(lvl.replace("h", "")) * 100.0
                h_list.append(val_h)
                rho_list.append(pressure_pa / (self.R * val_temp))
                u_list.append(val_u)
                v_list.append(val_v)

        if h_list:
            order = np.argsort(h_list)
            self.h_vals = np.array(h_list)[order]
            self.rho_vals = np.array(rho_list)[order]
            self.u_vals = np.array(u_list)[order]
            self.v_vals = np.array(v_list)[order]

    def plot_profiles(self) -> None:
        """Plot atmospheric wind, density, speed of sound, and viscosity profiles."""
        if self.model == "Default":
            h_plot = np.linspace(0, 11000, 100)
            winds = [self.wind(h) for h in h_plot]
            u_plot = np.array([w[0] for w in winds])
            v_plot = np.array([w[1] for w in winds])
        else:
            h_plot = self.h_vals
            u_plot = self.u_vals
            v_plot = self.v_vals

        speed_plot = np.sqrt(u_plot**2 + v_plot**2)
        rho_plot = self.density(h_plot)
        sound_plot = self.speed_of_sound(h_plot)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(u_plot, h_plot, label="U", color="blue")
        axs[0, 0].plot(v_plot, h_plot, label="V", color="red")
        axs[0, 0].set_title("Wind Components (m/s)")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(speed_plot, h_plot, color="black")
        axs[0, 1].set_title("Wind Speed (m/s)")
        axs[0, 1].grid(True)

        axs[1, 0].plot(rho_plot, h_plot, color="green")
        axs[1, 0].set_title("Density (kg/m³)")
        axs[1, 0].grid(True)

        axs[1, 1].plot(sound_plot, h_plot, color="orange")
        axs[1, 1].set_title("Speed of Sound (m/s)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
