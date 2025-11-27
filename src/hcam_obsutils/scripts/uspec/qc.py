import dataclasses
import re

from hipercam import HCAM, MCCD

from hcam_obsutils.dbutils import add_bias_data, get_bias_data
from hcam_obsutils.qcutils import (
    ReadoutMode,
    bias_measurement_to_dataframe_row,
    calc_and_plot,
    plot_qc_bias_archive,
)

DBFILE = "/home/observer/qc/ultraspec/uspec_qc.sqlite"
ccd_lut = {"1": "ccd"}
win_lut = {"1": "1"}


@dataclasses.dataclass
class UspecReadoutMode(ReadoutMode):
    binning: str
    speed: str
    output: str
    hvgain: int


def main(args=None):
    from trm import cline
    from trm.cline import Cline

    # get inputs
    command, args = cline.script_args(args)
    with Cline("HIPERCAM_ENV", ".hipercam", command, args) as cl:
        cl.register("fname", Cline.LOCAL, Cline.PROMPT)
        fname = cl.get_value(
            "fname", "hcam file to analyse:", cline.Fname("bias", HCAM)
        )

    if not fname.endswith(".hcm"):
        fname = fname + ".hcm"

    # read file
    mccd = MCCD.read(fname)
    # now determine the mean, median and standard deviation of each half of
    # each CCD

    # let's have some dictionaries
    means = dict()
    sigmas = dict()

    # get metadata on read noise, time etc.
    date = mccd.head["TIMSTAMP"].split("T")[0]
    binning = "%dx%d" % (mccd["1"]["1"].xbin, mccd["1"]["1"].ybin)
    output = mccd.head["OUTPUT"]
    speed = mccd.head["SPEED"]
    if output == "N":
        coutput = "NORMAL OUTPUT"
    else:
        coutput = "AVALANCHE OUTPUT"

    if speed == "F":
        cspeed = "FAST READOUT SPEED"
    elif speed == "M":
        cspeed = "MEDIUM READOUT SPEED"
    else:
        cspeed = "SLOW READOUT SPEED"
    hvgain = mccd.head["HVGAIN"]

    if output == "N":
        title = f"ULTRASPEC, {cspeed}, {coutput}"
    else:
        title = f"ULTRASPEC, {cspeed}, {coutput}, HVGAIN={hvgain}"

    mode = UspecReadoutMode(binning=binning, speed=speed, output=output, hvgain=hvgain)
    print(title)

    # define statistics region
    ylow = 100
    ystep = 400
    ysize = 200
    xleft = 200
    xstep = 400
    xsize = 200
    for nccd, ccd in mccd.items():
        if nccd not in means:
            means[nccd] = dict()
        if nccd not in sigmas:
            sigmas[nccd] = dict()

        for nwin, win in ccd.items():
            window_name = win_lut[nwin]
            mean, sd = calc_and_plot(
                ccd, nccd, nwin, window_name, xleft, xstep, xsize, ylow, ystep, ysize
            )
            means[nccd][nwin] = mean
            sigmas[nccd][nwin] = sd

    bias_df = get_bias_data(DBFILE, mode).sort_values(by=["date"])
    resp = input("do you want to compare these results with archival values?: ")
    if re.match("Y", resp.upper()):
        plot_qc_bias_archive(date, mode, ccd_lut, win_lut, means, sigmas, bias_df)

    resp = input("do you want to add these results to the quality control database?: ")
    if re.match("Y", resp.upper()):
        row = bias_measurement_to_dataframe_row(
            date, mode, ccd_lut, win_lut, means, sigmas
        )
        add_bias_data(DBFILE, bias_df, row)
