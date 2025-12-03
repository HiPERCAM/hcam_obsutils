import re

from astropy.coordinates import EarthLocation

from hcam_obsutils.dbutils import (
    add_zeropoint_data,
    create_zeropoint_table,
    get_zeropoint_data,
)
from hcam_obsutils.qcutils import plot_zeropoint_data
from hcam_obsutils.throughput import Calibrator

DBFILE = "/home/observer/qc/ultraspec/uspec_qc.sqlite"


def main(args=None):
    import warnings

    from sigfig import round as sigfig_round
    from trm import cline
    from trm.cline import Cline

    # get inputs
    command, args = cline.script_args(args)
    with Cline("HIPERCAM_ENV", ".hipercam", command, args) as cl:
        cl.register("logfile", Cline.LOCAL, Cline.PROMPT)
        cl.register("stdname", Cline.LOCAL, Cline.PROMPT)
        cl.register("band", Cline.LOCAL, Cline.PROMPT)

        logfile = cl.get_value(
            "logfile",
            "Logfile containing standard star observations:",
            cline.Fname("logfile", ".log"),
        )
        stdname = cl.get_value("stdname", "Name of the standard star:", "stdname")
        band = cl.get_value("band", "band", "g")

    if band.lower() == "kg5":
        print("Zeropoints for KG5 band not currently supported.")
        return

    TNT = EarthLocation(lat=18.574, lon=98.482, height=2449.0)
    calibrator = Calibrator("ultraspec", stdname, logfile, TNT)

    date = calibrator.date.isot.split("T")[0]
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_zp, median_zp, std_zp = calibrator.get_zeropoint(band)
        print(f"Band {band}: ZP = {sigfig_round(mean_zp, std_zp)}")
        results.append(dict(date=date, band=band, mean=mean_zp, err=std_zp))

    try:
        df = get_zeropoint_data(DBFILE)
    except Exception:
        # create table
        initial_row = results[0]
        create_zeropoint_table(DBFILE, initial_row)
        df = get_zeropoint_data(DBFILE)

    resp = input("do you want to compare these results with archival values?: ")
    if re.match("Y", resp.upper()):
        plot_zeropoint_data(df, [band], results)

    resp = input("do you want to add these results to the quality control database?: ")
    if re.match("Y", resp.upper()):
        for row in results:
            add_zeropoint_data(DBFILE, df, row)
