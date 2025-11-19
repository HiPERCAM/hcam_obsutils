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
    """
    Determine if a header corresponds to a bias frame.

    Parameters
    ----------
    header : Hhead or Uhead
        Header object to check.

    Returns
    -------
    bool
        True if the header corresponds to a bias frame, False otherwise.
    """
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
    """
    Determine if two Uhead objects correspond to the same format, for the purposes of calibration.

    Parameters
    ----------
    h1 : Uhead
        First header object.
    h2 : Uhead
        Second header object.
    fussy : bool
        If True, include avalanche gain in the comparison (only relevant for ULTRASPEC).

    Returns
    -------
    bool
        True if the two headers correspond to the same format, False otherwise.
    """
    # binning, gain, instrument, etc
    ok = (
        (h1.xbin == h2.xbin)
        and (h1.ybin == h2.ybin)
        and (h1.instrument == h2.instrument)
        and (len(h1.win) == len(h2.win))
        and (h1.gainSpeed == h2.gainSpeed)
        and (
            h1.header.get("HVGAIN", None) == h2.header.get("HVGAIN", None)
            if fussy
            else True
        )
    )
    # check windows are the same
    if ok:
        for window in h1.win:
            if not any(w == window for w in h2.win):
                ok = False
                break
    return ok


def hhead_equal(h1: Hhead, h2: Hhead, **kwargs) -> bool:
    """
    Determine if two Hhead objects correspond to the same format, for the purposes of calibration.

    Parameters
    ----------
    h1 : Hhead
        First header object.
    h2 : Hhead
        Second header object.

    Returns
    -------
    bool
        True if the two headers correspond to the same format, False otherwise.
    """
    ok = (
        (h1.xbin == h2.xbin)
        and (h1.ybin == h2.ybin)
        and len(h1.windows) == len(h2.windows)
        # mode
        and h1.header.get("ESO DET READ CURNAME", None)
        == h2.header.get("ESO DET READ CURNAME", None)
        # readout speed
        and h1.header.get("ESO DET SPEED", None) == h2.header.get("ESO DET SPEED", None)
    )
    if ok:
        # check windows are the same in all CCDs
        ok = sorted(h1.wforms) == sorted(h2.wforms)
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

    # choose comparison function
    compare = hhead_equal if args.hcam else uhead_equal

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
                if compare(header, rold, fussy=args.fussy):
                    new_format = False
                    break
            if new_format:
                key = header.fname if args.hcam else header.run
                destination[key] = header

    # now see if each non-bias has a matching bias
    for run, nhead in nonbiases.items():
        # skip data caution runs unless requested
        if not args.include_caution and nhead.header["DTYPE"].lower() == "data caution":
            continue

        has_bias = False
        # loop over all unique bias formats looking for a match
        for _, bhead in biases.items():
            if compare(nhead, bhead, fussy=args.fussy):
                has_bias = True
                break

        # no match for this run, report
        if not has_bias:
            print(f"No bias found for run {run} in format:")
