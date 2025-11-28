import importlib

import numpy as np
import pandas as pd
from astropy import coordinates as coord
from astropy import units as u
from astropy.coordinates import AltAz
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from hipercam.core import (
    BAD_TIME,
    CLOUDS,
    JUNK,
    NO_DATA,
    NO_EXTRACTION,
    NO_FWHM,
    NO_SKY,
    TARGET_NONLINEAR,
    TARGET_SATURATED,
)
from hipercam.hlog import Hlog
from thefuzz import fuzz

ucam_dates = dict(
    CUBE=Time("2019-09-24T12:00"),
    SUPERSDSS=Time("2017-05-01T12:00"),
)

STD_PATH = importlib.resources.files("hcam_obsutils") / "data"
std_tables = dict(
    hipercam="hcam_flux_stds.csv",
    ultracam_precube="ucam_flux_stds.csv",
    ultracam="ucam_cube_flux_stds.csv",
    ultraspec="sdss_flux_stds.csv",
)


class Calibrator:
    """
    A class to simplify the photometric calibration of data.

    Can be used to calculate zeropoints from standard star observations,
    or to calculate magnitudes of comparison stars in the field.

    By default, uses standard atmospheric extinction coefficients,
    but these can be changed using the set_atm_extinction method.

    Parameters
    ----------
    instrument : str
        The instrument used, either 'ultracam', 'ultraspec' or 'hipercam'
    std_name : str
        The name of the standard star to use for calibration. Should be in the standard
        star tables provided.
    std_logfile : str
        The logfile containing the standard star observations.
    observatory : str
        The name of the observatory where the data were taken.
    comp_logfile : str | None
        The logfile containing the comparison star observations.
        If None, only zeropoints will be calculated.
    coords : astropy.coordinates.SkyCoord | None
        The coordinates of the comparison star(s). Required if comp_logfile is given.

    Examples
    --------
        >>> coords = coord.SkyCoord("23 17 29.7 +52 20 20.7", unit=(u.hourangle, u.deg))
        >>> observatory = "lapalma"
        >>> std_name = "GD153"
        >>> std_logfile = "2023_07_27/run0006.log"
        >>> comp_logfile = "2023_07_27/run0019_cal.log"
        >>> bands = ["u", "g", "r", "i", "z"]
        >>> calibrator = Calibrator(
        ...     "hipercam", std_name, std_logfile, comp_logfile, coords, observatory
        ... )

        >>> for band in bands:
        >>>    mean, median, std = calibrator.comparison_mags(band, "2")
        >>>    print(f"Comparison magnitudes ({band}): {mean:.3f} {median:.3f} {std:.3f}")
        >>>    print("")
    """

    # table of default atmospheric extinction coefficients
    # can be overridden if needed
    atm_extinction = {
        "us": 0.48,
        "gs": 0.24,
        "rs": 0.18,
        "is": 0.15,
        "zs": 0.10,
        "u": 0.48,
        "g": 0.24,
        "r": 0.18,
        "i": 0.15,
        "z": 0.10,
    }

    def __init__(
        self,
        instrument: str,
        std_name: str,
        std_logfile: str,
        observatory: str | coord.EarthLocation,
        comp_logfile: str | None = None,
        coords: coord.SkyCoord | None = None,
    ):
        if instrument.lower() not in ["ultracam", "hipercam", "ultraspec"]:
            raise ValueError(f"Unknown instrument: {instrument}")

        self.instrument = instrument.lower()
        if self.instrument == "ultracam":
            self.band_to_ccd = {"u": "3", "g": "2", "r": "1", "i": "1", "z": "1"}
        elif self.instrument == "ultraspec":
            self.band_to_ccd = {"u": "1", "g": "1", "r": "1", "i": "1", "z": "1"}
        else:
            self.band_to_ccd = {"u": "1", "g": "2", "r": "3", "i": "4", "z": "5"}

        self.std_name = std_name
        self.std_logfile = std_logfile
        self.comp_logfile = comp_logfile
        self.coords = coords
        if isinstance(observatory, str):
            self.observatory = coord.EarthLocation.of_site(observatory)
        else:
            self.observatory = observatory

        if self.comp_logfile is not None and self.coords is None:
            raise ValueError("If comp_logfile is given, coords must also be provided.")

        # get a date from the logfile (used to see which UCAM standard star table to use)
        lf = Hlog.rascii(self.std_logfile)
        ts = lf.tseries("1", "1")
        self.date = Time(ts.t[0], format="mjd")

        self.get_std_info()

    def set_atm_extinction(self, band, value):
        """
        Sets the atmospheric extinction coefficient for a given band
        """
        self.atm_extinction[band] = value

    def get_std_info(self):
        if self.instrument == "ultracam":
            if self.date < ucam_dates["CUBE"]:
                filename = STD_PATH / std_tables["ultracam_precube"]
                self.prefix = "ucam_"
            else:
                filename = STD_PATH / std_tables["ultracam"]
                self.prefix = "ucam_cb_"
            if self.date > ucam_dates["SUPERSDSS"]:
                self.postfix = "s"
            else:
                self.postfix = ""

        elif self.instrument == "hipercam":
            filename = STD_PATH / std_tables["hipercam"]
            self.prefix = "hcam_"
            self.postfix = "s"
        else:  # ultraspec
            filename = STD_PATH / std_tables["ultraspec"]
            self.prefix = "sdss_"
            self.postfix = ""

        bands = ["u", "g", "r", "i", "z"]
        df = pd.read_csv(filename)
        row = df.query(f"Name == '{self.std_name}'")
        if row.empty:
            # no match - try fuzzy matching to give a useful error message
            closeness = df.Name.apply(lambda x: fuzz.ratio(x, self.std_name))
            matches = ",".join(df.Name[closeness >= 80])
            raise ValueError(
                f"Unknown standard star: {self.std_name}. Did you mean one of: {matches}?"
            )

        self.std_coo = coord.SkyCoord(row["RA"], row["DEC"], unit=u.deg)
        self.std_mags = {
            band: row[f"{self.prefix}{band}{self.postfix}"].values[0] for band in bands
        }

    def inst_mags(self, which, band, aperture):
        """
        Calculates the instrumental magnitudes, corrected for atmospheric extinction
        """
        if which not in ["std", "comp"]:
            raise ValueError(f"Unknown value for 'which': {which}")
        if which == "std":
            logfile = self.std_logfile
            coords = self.std_coo
        else:
            if self.comp_logfile is None:
                raise ValueError("comp_logfile is not set")
            logfile = self.comp_logfile
            coords = self.coords

        ccd = self.band_to_ccd[band]

        # first we create a "Tseries". This is the "raw" photometry.
        lf = Hlog.rascii(logfile)
        ts = lf.tseries(ccd, aperture)

        # now we get the time, counts and error from this Tseries, ignoring
        # any data points flagged as bad for some reason
        ANY_FLAG = (
            NO_FWHM
            | NO_SKY
            | TARGET_NONLINEAR
            | TARGET_SATURATED
            | NO_EXTRACTION
            | NO_DATA
            | BAD_TIME
            | CLOUDS
            | JUNK
        )
        t, expt, y, ey = ts.get_data(ANY_FLAG)
        t = Time(t, format="mjd")
        # airmass from the time and coordinates
        aa = AltAz(obstime=t, location=self.observatory)
        altaz = coords.transform_to(aa)
        airmass = altaz.secz.value
        # instrumental magnitude, corrected for atmospheric extinction
        k = self.atm_extinction[band]
        m_inst = -2.5 * np.log10(y / expt / 86400) - k * airmass
        return m_inst

    def get_zeropoint(self, band):
        """
        Calculate zeropoint for a given band based on std star observations
        """
        g_inst_std = self.inst_mags("std", band, "1")
        zp_mean, zp_median, zp_err = sigma_clipped_stats(
            self.std_mags[band] - g_inst_std
        )
        return zp_mean, zp_median, zp_err

    def comparison_mags(self, band, aperture):
        """
        Calculates the comparison magnitudes
        """
        if self.comp_logfile is None:
            raise ValueError("comp_logfile is not set")
        inst_comp = self.inst_mags("comp", band, aperture)
        zp_mean, zp_median, zp_err = self.get_zeropoint(band)
        print(f"Zeropoint ({band}): {zp_mean:.3f} {zp_median:.3f} {zp_err:.3f}")
        comp_mags = inst_comp + zp_median
        return sigma_clipped_stats(comp_mags)
