import dataclasses
import re
import sys

from hipercam import HCAM, MCCD

from hcam_obsutils.dbutils import add_bias_data, get_bias_data
from hcam_obsutils.qcutils import (
    ReadoutMode,
    bias_measurement_to_dataframe_row,
    calc_and_plot,
    plot_qc_bias_archive,
)

DBFILE = "/home/observer/qc/ultracam/ucam_qc.sqlite"
ccd_lut = {"1": "red", "2": "grn", "3": "blu"}
win_lut = {"1": "Left", "2": "Right"}


@dataclasses.dataclass
class UCAMReadoutMode(ReadoutMode):
    binning: str
    readout: str


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

    # define box for average and sigma
    ylow = 250
    ystep = 400
    ysize = 100
    xleft = 100
    xstep = 200
    xsize = 100

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

    # get metadata on read noise, time etc.
    # get metadata from data
    if mccd.head["GAINSPED"] == "cdd":
        readout = "SLOW"
    elif mccd.head["GAINSPED"] == "fbb":
        readout = "FAST"
    else:
        readout = "TURBO"

    date = mccd.head["TIMSTAMP"].split("T")[0]
    binning = "%dx%d" % (mccd["1"]["1"].xbin, mccd["1"]["1"].ybin)
    mode = UCAMReadoutMode(binning=binning, readout=readout)

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
