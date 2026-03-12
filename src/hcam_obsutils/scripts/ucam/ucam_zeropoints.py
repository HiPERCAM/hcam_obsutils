import os
import re
from pathlib import Path

from hcam_obsutils.dbutils import (
    add_zeropoint_data,
    create_zeropoint_table,
    get_zeropoint_data,
)
from hcam_obsutils.qcutils import plot_zeropoint_data
from hcam_obsutils.throughput import Calibrator

DB_LOCATION = Path(os.getenv("HCAM_QC_DBLOC", "/home/observer/qc"))
DBFILE = DB_LOCATION / "ultracam" / "ucam_qc.sqlite"


def ucam_zeropoints(args=None):
    """
    Calculate the zeropoint for a standard star observation.

    uspec_zeropoints uses a reduced logfile from a standard star observation
    to calculate the zeropoint. Default values of atmospheric extinction 
    are assumed. 

    This routine is intended to be used with observations of one of the standard
    stars listed in the photometric standards compiled by Alex Brown, a database
    of which is listed in the `data` directory of this repository. 

    The zeropoints are plotted against a historical database of measurements
    and you can optionally add this measurement to this database. The location 
    of the database is set by the HCAM_QC_DBLOC environment variable. If not
    set it will be nested inside /home/observer/qc.
    
    Parameters
    ----------
    logfile: str
        Logfile containing standard star observations. You should use a 
        large aperture to be sure to capture all the flux from the star. 

    stdname: str
        Name of the standard star. This should match one of the names in the
        database of standards compiled by Alex Brown. 
    
    bands: str
        Bands in which the standard star was observed.
        These should be space seperated (e.g. 'u g r').
        Don't use subscripts or primes - we assume these
        are super SDSS filters.  
    """ 
    import warnings

    from sigfig import round as sigfig_round
    from trm import cline
    from trm.cline import Cline

    # get inputs
    command, args = cline.script_args(args)
    with Cline("HIPERCAM_ENV", ".hipercam", command, args) as cl:
        cl.register("logfile", Cline.LOCAL, Cline.PROMPT)
        cl.register("stdname", Cline.LOCAL, Cline.PROMPT)
        cl.register("bands", Cline.LOCAL, Cline.PROMPT)

        logfile = cl.get_value(
            "logfile",
            "Logfile containing standard star observations:",
            cline.Fname("logfile", ".log"),
        )
        stdname = cl.get_value("stdname", "Name of the standard star:", "stdname")

        bands = cl.get_default("bands")
        if bands is not None and len(bands) != 3:
            cl.set_default("bands", "u g r")

        bands = cl.get_value(
            "bands",
            "bands used (space separated, e.g. 'u g r'):",
            "u g r",
        ).split()

    calibrator = Calibrator("ultracam", stdname, logfile, "lasilla")

    date = calibrator.date.isot.split("T")[0]
    results = []
    for band in bands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_zp, median_zp, std_zp = calibrator.get_zeropoint(band)
            print(f"Band {band}: ZP = {sigfig_round(mean_zp, std_zp)}")
            results.append(
                dict(date=date, stdname=stdname, band=band, mean=mean_zp, err=std_zp)
            )

    try:
        df = get_zeropoint_data(DBFILE)
    except Exception:
        # create table
        initial_row = results[0]
        create_zeropoint_table(DBFILE, initial_row)
        df = get_zeropoint_data(DBFILE)

    resp = input("do you want to compare these results with archival values?: ")
    if re.match("Y", resp.upper()):
        plot_zeropoint_data(df, bands, results)

    resp = input("do you want to add these results to the quality control database?: ")
    if re.match("Y", resp.upper()):
        for row in results:
            add_zeropoint_data(DBFILE, df, row)
