import hipercam as hcam
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from hcam_obsutils.qcutils.gain import gain_simple


def main(args=None):
    from trm import cline
    from trm.cline import Cline

    # get inputs
    command, args = cline.script_args(args)
    with Cline("HIPERCAM_ENV", ".hipercam", command, args) as cl:
        cl.register("flat", Cline.LOCAL, Cline.PROMPT)
        cl.register("bias", Cline.LOCAL, Cline.PROMPT)

        flat_name = cl.get_value(
            "flat",
            "Flat field hcm file (should not be bias subtracted):",
            cline.Fname("flat1", hcam.HCAM),
        )
        bias_name = cl.get_value(
            "bias", "Bias frame hcm file:", cline.Fname("bias", hcam.HCAM)
        )

    # read files
    flat = hcam.MCCD.read(flat_name)
    bias = hcam.MCCD.read(bias_name)

    # now subtract the bias, determine the mean and standard deviation and then
    # the gain
    flat -= bias.crop(flat)

    for nccd, ccd in flat.items():
        for nwin, win in ccd.items():
            sub_win = win.window(
                win.llx + 200, win.llx + 300, win.lly + 450, win.lly + 550
            )

            mean = sub_win.mean()
            sigma = sub_win.std()

            gain = gain_simple(mean, sigma)
            print("CCD%s %s mean   = %5.0f" % (nccd, nwin, mean))
            print("CCD%s %s sigma  = %4.1f" % (nccd, nwin, sigma))
            print("CCD%s %s gain  =  %4.1f" % (nccd, nwin, gain))
            print("")

            # and plot them at high contrast to check for readout noise
            _, axes = plt.subplots()
            plo = 0.999
            phi = 1.001
            hcam.mpl.pWind(axes, win, plo * win.median(), phi * win.median())
            axes.grid(False)
            # plot CCD boundary
            axes.plot(
                [0.5, ccd.nxtot + 0.5, ccd.nxtot + 0.5, 0.5, 0.5],
                [0.5, 0.5, ccd.nytot + 0.5, ccd.nytot + 0.5, 0.5],
            )
            p = Rectangle(
                (win.llx + 200, win.lly + 450), 100, 100, color="r", fill=False, lw=2
            )
            axes.add_patch(p)
            plt.show()
