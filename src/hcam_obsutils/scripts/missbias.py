import argparse

from hcam_obsutils.formats import headers, hhead_equal, is_bias, uhead_equal

HELP = """
missbias reads all the runs in the directories specified and tries to work out if there
are any non-biases without corresponding biases. This is a crude test and does not verify that
runs identified as 'Bias' are what they say they are or that they are any good. As well as the
directories specified, the script also looks for subdirectories called 'data'
"""


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
        help="process HiPERCAM runs rather than ULTRASPEC and/or ULTRACAM runs",
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
            print(
                f"No bias found for {run} in format: {nhead.mode} {nhead.xbin}x{nhead.ybin} {nhead.wforms}"
            )
