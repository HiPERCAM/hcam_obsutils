from hipercam.hlog import Hlog

from hcam_obsutils.throughput import Calibrator


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

    for band in bands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean_zp, median_zp, std_zp = calibrator.get_zeropoint(band)
            print(f"Band {band}: ZP = {sigfig_round(mean_zp, std_zp)}")
