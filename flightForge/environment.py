import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import datetime

class Environment:
    def __init__(self, api_key=None, lat=None, lon=None, model='gfs', wind_profile=None, rho_profile=None, e_log=False):
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
        
    def set_model(self, api_key, lat, lon, model='gfs', date=None):
        self.lat = lat
        self.lon = lon
        self.model = model
        
        target_ts_ms = None
        if date is not None:
            try:
                day, month, year = date
                dt_obj = datetime.datetime(year, month, day, 12, 0, 0, tzinfo=datetime.timezone.utc)
                
                target_ts_sec = dt_obj.timestamp()
                
                target_ts_ms = target_ts_sec * 1000
                
            except ValueError as e:
                raise ValueError(f"Invalid date tuple provided: {e}")
            except Exception:
                raise ValueError("Date argument must be a tuple (day, month, year).")


        self._fetch_data(api_key, float(lat), float(lon), model, target_ts_ms)
        self.wind_profile = self._api_wind_profile
        self.rho_profile = self._api_rho_profile
        if self.e_log:
            self._cmd_log()

    def density(self, h):
        return self.rho_profile(h)
    
    def wind(self, h):
        return self.wind_profile(h)

    def _cmd_log(self):

        print(f"-------ENVIRONMENT INFO --------")
        if self.lat is not None and self.lon is not None:
            print(f"Coordinates:   {self.lat}, {self.lon}")
        else:
            print(f"Coordinates:   Not Defined")
            
        print(f"Model Used:    {self.model}")
        
        u_surf = self.u_vals[0]
        v_surf = self.v_vals[0]
        v_mag = np.sqrt(u_surf**2 + v_surf**2)
        
        print(f"Surface Wind:  U={u_surf:.2f} m/s, V={v_surf:.2f} m/s")
        print(f"               Mag={v_mag:.2f} m/s")
        print("--------------------------------")

        self._plot_profiles()

    def _plot_profiles(self):
        if self.model == "Default":
            h_plot = np.linspace(0, 10000, 100)
            winds = [self.wind(h) for h in h_plot]
            u_plot = np.array([w[0] for w in winds])
            v_plot = np.array([w[1] for w in winds])
        else:
            h_plot = self.h_vals
            u_plot = self.u_vals
            v_plot = self.v_vals

        speed_plot = np.sqrt(u_plot**2 + v_plot**2)
        math_angle = np.degrees(np.arctan2(v_plot, u_plot))
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        axs[0].plot(u_plot, h_plot, label='U (East)', color='blue')
        axs[0].plot(v_plot, h_plot, label='V (North)', color='red')
        axs[0].set_xlabel('Velocity (m/s)')
        axs[0].set_ylabel('Altitude (m)')
        axs[0].set_title('Wind Components')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(speed_plot, h_plot, color='black')
        axs[1].set_xlabel('Speed (m/s)')
        axs[1].set_ylabel('Altitude (m)')
        axs[1].set_title('Wind Speed Magnitude')
        axs[1].grid(True)
        
        axs[2].plot(math_angle, h_plot, color='purple')
        axs[2].set_xlabel('Direction (deg, 0=East, 90=North)')
        axs[2].set_ylabel('Altitude (m)')
        axs[2].set_title('Wind Direction vector')
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

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
            response = requests.post("https://api.windy.com/api/point-forecast/v2", json=payload)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Windy API Error {response.status_code}: {response.text}") from e

        data = response.json()

        target_ts = target_ts if target_ts is not None else time.time() * 1000
        
        ts_arr = np.array(data['ts'])
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

            if key_gh in data and key_temp in data and key_u in data and key_v in data:
                val_h = data[key_gh][idx]
                val_temp = data[key_temp][idx]
                val_u = data[key_u][idx]
                val_v = data[key_v][idx]

                if val_h is not None and val_temp is not None and val_u is not None and val_v is not None:
                    pressure_pa = int(lvl.replace('h', '')) * 100.0
                    rho = pressure_pa / (287.05 * val_temp)

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

    def _api_wind_profile(self, h):
        u_interp = np.interp(h, self.h_vals, self.u_vals)
        v_interp = np.interp(h, self.h_vals, self.v_vals)
        return (v_interp, u_interp)

    def _api_rho_profile(self, h):
        return np.interp(h, self.h_vals, self.rho_vals)

    @staticmethod
    def _def_wind_profile(h):
        speed = 3 + 0.01 * h
        angle = np.radians(10 + 0.02*h)
        wx = speed * np.cos(angle)
        wy = speed * np.sin(angle)
        return (wx, wy)

    @staticmethod
    def _def_rho_profile(h):
        rho0 = 1.225
        H = 8500.0
        return rho0 * np.exp(-h / H)