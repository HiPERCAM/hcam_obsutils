import argparse
from pathlib import Path
from typing import Iterable

from hipercam.hcam import Rhead as Hhead
from hipercam.ucam import Rhead as Uhead

HELP = """
missbias reads all the runs in the directories specified and tries to work out if there
are any non-biases without corresponding biases. This is a crude test and does not verify that
runs identified as 'Bias' are what they say they are or that they are any good. As well as the
directories specified, the script also looks for subdirectories called 'data'
"""

UCAM_RE = "run[0-9][0-9][0-9].xml"
HCAM_RE = "run[0-9][0-9][0-9][0-9].fits"


def is_bias(header: Hhead | Uhead) -> bool:
    try:
        # assume ucam first
        target = header.header["TARGET"].lower()
    except KeyError:
        target = header.header["OBJECT"].lower()
    return "bias" in target


def headers(dirpath: str, hcam: bool = False) -> Iterable[Hhead | Uhead]:
    """
    Generator yielding header objects from all runs in dirpath.

    ULTRACAM/ULTRASPEC Power ON/OFF runs are skipped.

    Parameters
    ----------
    dirpath : str
        Path to directory to search for runs.
    hcam : bool
        If True, process HiPERCAM runs, otherwise ULTRASPEC/ULTRACAM runs.

    Yields
    ------
    header : Hhead or Uhead
        Header object for each run found.
    """
    dirpath = Path(dirpath)
    if dirpath.is_dir():
        header_files = dirpath.glob(HCAM_RE) if hcam else dirpath.glob(UCAM_RE)
        for fn in header_files:
            fn = fn.with_suffix("")
            header = Hhead(str(fn)) if hcam else Uhead(str(fn))
            if not hcam and header.isPonoff():
                continue
            yield header


def uhead_equal(h1: Uhead, h2: Uhead, fussy: bool = False) -> bool:
    ok = (
        (h1.xbin == h2.xbin)
        and (h1.instrument == h2.instrument)
        and (h1.ybin == h2.ybin)
        and (len(h1.win) == len(h2.win))
        and (h1.gainSpeed == h2.gainSpeed)
        and (
            h1.header.get("HVGAIN", None) == h2.header.get("HVGAIN", None)
            if fussy
            else True
        )
    )
    if ok:
        for window in h1.win:
            if not any(w == window for w in h2.win):
                ok = False
                break
    return ok


def main():
    parser = argparse.ArgumentParser(description=HELP)
    parser.add_argument(
        "-f",
        "--fussy",
        action="store_true",
        default=False,
        help="fussy tests ensure difference in avalanche gains are picked up, only important for ULTRASPEC",
    )
    parser.add_argument(
        "-i",
        "--include-caution",
        default=False,
        action="store_true",
        help="include runs marked 'data caution' when listing runs without biasses",
    )
    parser.add_argument(
        "--hcam",
        action="store_true",
        default=False,
        help="process HiPERCAM runs rather than ULTRASPEC and/or ULTRACAMruns",
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        help="directories to search for runs, subdirectories called 'data' will also be searched",
    )
    args = parser.parse_args()

    # accumulate a list of unique biases and non-biases
    nonbiases = {}
    biases = {}
    dirs = set(["data"] + args.dirs)
    for dirpath in sorted(dirs):
        # all headers in this directory
        for header in headers(dirpath, hcam=args.hcam):
            # which dictionary to store in?
            if is_bias(header):
                destination = biases
            else:
                destination = nonbiases

            # compare with already stored formats
            new_format = True
            for _, rold in destination.items():
                if uhead_equal(header, rold, fussy=args.fussy):
                    new_format = False
                    break
            if new_format:
                destination[header.run] = header

    # now see if each non-bias has a matching bias
    for run, nhead in nonbiases.items():
        if not args.include_caution and nhead.header["DTYPE"].lower() == "data caution":
            continue

        has_bias = False
        for _, bhead in biases.items():
            if uhead_equal(nhead, bhead, fussy=args.fussy):
                has_bias = True
                break
        if not has_bias:
            print(f"No bias found for run {run} in format:")
