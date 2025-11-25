import hipercam as hcam
import numpy as np

from hcam_obsutils.qcutils import block_stats
from hcam_obsutils.qcutils.gain import gain

HELP = """
Python script to measure ULTRACAM gain in a quick and dirty manner.

The two flats should have different mean count levels.
Flat fields will be bias subtracted, so you also have to supply a bias frame
"""


def metadata(mccd):
    # get metadata from data
    if mccd.head["GAINSPED"] == "cdd":
        readout = "SLOW"
    elif mccd.head["GAINSPED"] == "fbb":
        readout = "FAST"
    else:
        readout = "TURBO"

    date = mccd.head["TIMSTAMP"].split("T")[0]
    binning = "%dx%d" % (mccd["1"]["1"].xbin, mccd["1"]["1"].ybin)
    return date, readout, binning


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "flat1",
        type=str,
        help="First flat field hcm file (should not be bias subtracted)",
    )
    parser.add_argument("flat2", type=str, help="Second flat field hcm file")
    parser.add_argument("bias", type=str, help="Bias frame hcm file")
    args = parser.parse_args()

    flat1_name = args.flat1 if args.flat1.endswith(".hcm") else args.flat1 + ".hcm"
    flat1 = hcam.MCCD.read(flat1_name)
    flat2_name = args.flat2 if args.flat2.endswith(".hcm") else args.flat2 + ".hcm"
    flat2 = hcam.MCCD.read(flat2_name)
    bias_name = args.bias if args.bias.endswith(".hcm") else args.bias + ".hcm"
    bias = hcam.MCCD.read(bias_name)

    date, readout, binning = metadata(flat1)
    for nccd, ccd in flat1.items():
        for nwin, win in ccd.items():
            g = gain(
                flat1,
                flat2,
                bias,
                nccd,
                nwin,
                xmin=200,
                xmax=300,
                ymin=300,
                ymax=400,
            )
            # _, bias_level, rno = block_stats(bias[nccd][nwin].data)

            print("CCD{}, WIN{} ({} {})".format(nccd, nwin, speed, binning))
            print("======================================================")
            print("")
            # print("  Bias level:    {:4.0f} ADU".format(bias_level))
            # print("  Read noise:    {:4.1f} e-".format(rno))
            print("  Gain:           {:4.1f} e-/ADU".format(g))
            print("")
