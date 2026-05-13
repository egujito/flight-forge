import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import requests

from .logger import logger

class Environment:
    def __init__(
        self,
        api_key=None,
        lat=None,
        lon=None,
        model="gfs",
        wind_profile=None,
        rho_profile=None,
        e_log=False,
    ):
        self.g = 9.80665
        self.R = 287.05
        self.gamma = 1.4
        self.beta = 1.458e-6
        self.S = 110.4

        self.wind_profile = self._def_wind_profile
        self.rho_profile = self._def_rho_profile

        self.h_vals = np.array([0.0])
        self.rho_vals = np.array([1.225])
        self.u_vals = np.array([0.0])
        self.v_vals = np.array([0.0])

        self.e_log = e_log
        self.lat = lat
        self.lon = lon
        self.model = "Default" if api_key is None else model

        if wind_profile is not None:
            self.wind_profile = wind_profile
        if rho_profile is not None:
            self.rho_profile = rho_profile

        if api_key and lat is not None and lon is not None:
            self.set_model(api_key, lat, lon, model)

    def set_model(self, api_key, lat, lon, model="gfs", date=None):
        self.lat = lat
        self.lon = lon
        self.model = model

        target_ts_ms = None
        if date is not None:
            try:
                day, month, year = date
                dt_obj = datetime.datetime(
                    year, month, day, 12, 0, 0, tzinfo=datetime.timezone.utc
                )
                target_ts_sec = dt_obj.timestamp()
                target_ts_ms = target_ts_sec * 1000
            except Exception as e:
                raise ValueError(f"Invalid date format: {e}")

        self._fetch_data(api_key, float(lat), float(lon), model, target_ts_ms)
        
        self.wind_profile = self._api_wind_profile
        self.rho_profile = self._api_rho_profile
        
        if self.e_log:
            self._cmd_log()

    def density(self, h):
        return self.rho_profile(h)
    
    def wind(self, h):
        return self.wind_profile(h)

    def speed_of_sound(self, h):
        T = self._get_isa_temperature(h)
        return np.sqrt(self.gamma * self.R * T)

    def dynamic_viscosity(self, h):
        T = self._get_isa_temperature(h)
        return (self.beta * T**(1.5)) / (T + self.S)

    def _get_isa_temperature(self, h):
        h = np.array(h, dtype=float)
        
        T0 = 288.15
        L = 0.0065
        
        T_trop = 216.65
        h_trop = 11000.0

        T = np.where(h <= h_trop, T0 - (L * h), T_trop)
        return T

    def _def_rho_profile(self, h):
        h = np.array(h, dtype=float)
        
        P0 = 101325.0
        T0 = 288.15
        L = 0.0065
        
        h_trop = 11000.0
        T_trop = 216.65
        P_trop = 22632.10
        
        T = self._get_isa_temperature(h)
        
        press_trop = P0 * (1 - (L * h) / T0) ** (self.g / (self.R * L))
        
        exponent = -self.g * (h - h_trop) / (self.R * T_trop)
        press_strat = P_trop * np.exp(exponent)
        
        P = np.where(h <= h_trop, press_trop, press_strat)
        
        return P / (self.R * T)

    @staticmethod
    def _def_wind_profile(h):
        h = np.array(h)
        if h.shape:
             return (np.zeros_like(h), np.zeros_like(h))
        return (0.0, 0.0)

    def _cmd_log(self):
        logger.info("-------ENVIRONMENT INFO --------")
        if self.lat is not None and self.lon is not None:
            logger.info(f"Coordinates:   {self.lat}, {self.lon}")
        else:
            logger.info("Coordinates:   Not Defined")
            
        logger.info(f"Model Used:    {self.model}")
        
        u_surf = self.u_vals[0]
        v_surf = self.v_vals[0]
        v_mag = np.sqrt(u_surf**2 + v_surf**2)
        
        logger.info(f"Surface Wind:  U={u_surf:.2f} m/s, V={v_surf:.2f} m/s")
        logger.info(f"               Mag={v_mag:.2f} m/s")
        logger.info("--------------------------------")

        self._plot_profiles()

    def _plot_profiles(self):
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
        math_angle = np.degrees(np.arctan2(v_plot, u_plot))
        
        rho_plot = self.density(h_plot)
        sound_plot = self.speed_of_sound(h_plot)
        visc_plot = self.dynamic_viscosity(h_plot)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        
        axs[0, 0].plot(u_plot, h_plot, label='U', color='blue')
        axs[0, 0].plot(v_plot, h_plot, label='V', color='red')
        axs[0, 0].set_title('Wind Components (m/s)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(speed_plot, h_plot, color='black')
        axs[0, 1].set_title('Wind Speed (m/s)')
        axs[0, 1].grid(True)
        
        axs[0, 2].plot(math_angle, h_plot, color='purple')
        axs[0, 2].set_title('Wind Direction (deg)')
        axs[0, 2].grid(True)

        axs[1, 0].plot(rho_plot, h_plot, color='green')
        axs[1, 0].set_title('Density (kg/m^3)')
        axs[1, 0].set_xlabel('Density')
        axs[1, 0].grid(True)

        axs[1, 1].plot(sound_plot, h_plot, color='orange')
        axs[1, 1].set_title('Speed of Sound (m/s)')
        axs[1, 1].set_xlabel('Speed')
        axs[1, 1].grid(True)

        axs[1, 2].plot(visc_plot, h_plot, color='brown')
        axs[1, 2].set_title('Dyn. Viscosity (Pa*s)')
        axs[1, 2].set_xlabel('Viscosity')
        axs[1, 2].grid(True)

        plt.tight_layout()
        plt.show()

    def _api_wind_profile(self, h):
        u_interp = np.interp(h, self.h_vals, self.u_vals)
        v_interp = np.interp(h, self.h_vals, self.v_vals)
        return (u_interp, v_interp)

    def _api_rho_profile(self, h):
        return np.interp(h, self.h_vals, self.rho_vals)

    def _fetch_data(self, key, lat, lon, model, target_ts=None):
        levels = ["1000h", "950h", "925h", "900h", "850h", "800h", 
                  "700h", "600h", "500h", "400h", "300h", "200h", "150h"]
        
        payload = {
            "lat": lat,
            "lon": lon,
            "model": model,
            "parameters": ["wind", "temp", "gh"],
            "levels": levels,
            "key": key
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
        ts_arr = np.array(data["ts"])
        idx = (np.abs(ts_arr - target_ts)).argmin()

        h_list = []
        rho_list = []
        u_list = []
        v_list = []

        for lvl in levels:
            key_gh = f"gh-{lvl}"
            key_temp = f"temp-{lvl}"
            key_u = f"wind_u-{lvl}"
            key_v = f"wind_v-{lvl}"

            if (key_gh in data and key_temp in data and 
                key_u in data and key_v in data):
                
                val_h = data[key_gh][idx]
                val_temp = data[key_temp][idx]
                val_u = data[key_u][idx]
                val_v = data[key_v][idx]

                if all(v is not None for v in [val_h, val_temp, val_u, val_v]):
                    pressure_pa = int(lvl.replace("h", "")) * 100.0
                    rho = pressure_pa / (self.R * val_temp)

                    h_list.append(val_h)
                    rho_list.append(rho)
                    u_list.append(val_u)
                    v_list.append(val_v)

        if h_list:
            self.h_vals = np.array(h_list)
            self.rho_vals = np.array(rho_list)
            self.u_vals = np.array(u_list)
            self.v_vals = np.array(v_list)

            sort_order = np.argsort(self.h_vals)
            self.h_vals = self.h_vals[sort_order]
            self.rho_vals = self.rho_vals[sort_order]
            self.u_vals = self.u_vals[sort_order]
            self.v_vals = self.v_vals[sort_order]
