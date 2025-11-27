import hipercam as hcam

from hcam_obsutils.qcutils import block_stats
from hcam_obsutils.qcutils.gain import gain


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


def main(args=None):
    """
    Python script to measure ULTRACAM gain in a quick and dirty manner.

    The two flats should have different mean count levels.
    Flat fields will be bias subtracted, so you also have to supply a bias frame

    Parameters
    ----------
    flat1 : str
        First flat field hcm file (should not be bias subtracted)
    flat2 : str
        Second flat field hcm file
    bias : str
        Bias frame hcm file
    """
    from trm import cline
    from trm.cline import Cline

    # get inputs
    command, args = cline.script_args(args)
    with Cline("HIPERCAM_ENV", ".hipercam", command, args) as cl:
        cl.register("flat1", Cline.LOCAL, Cline.PROMPT)
        cl.register("flat2", Cline.LOCAL, Cline.PROMPT)
        cl.register("bias", Cline.LOCAL, Cline.PROMPT)

        flat1_name = cl.get_value(
            "flat1",
            "First flat field hcm file (should not be bias subtracted):",
            cline.Fname("flat1", hcam.HCAM),
        )
        flat2_name = cl.get_value(
            "flat2", "Second flat field hcm file:", cline.Fname("flat2", hcam.HCAM)
        )
        bias_name = cl.get_value(
            "bias", "Bias frame hcm file:", cline.Fname("bias", hcam.HCAM)
        )

    flat1 = hcam.MCCD.read(flat1_name)
    flat2 = hcam.MCCD.read(flat2_name)
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
            _, bias_level, rno = block_stats(bias[nccd][nwin].data)

            print("CCD{}, WIN{} ({} {})".format(nccd, nwin, readout, binning))
            print("======================================================")
            print("")
            print("  Bias level:    {:4.0f} ADU".format(bias_level))
            print("  Read noise:    {:4.1f} ADU".format(rno))
            print("  Read noise:    {:4.1f} e-".format(rno / g))
            print("  Gain:           {:4.1f} e-/ADU".format(g))
            print("")
