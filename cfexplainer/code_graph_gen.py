import os
import sys
import numpy as np
from helpers import utils
from helpers import joern
from data_pre import bigvul


def preprocess(row):
    savedir_before = utils.get_dir(utils.processed_dir() / row["dataset"] / "before")
    savedir_after = utils.get_dir(utils.processed_dir() / row["dataset"] / "after")

    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])

    fpath2 = savedir_after / f"{row['id']}.c"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])

    if not os.path.exists(f"{fpath1}.edges.json"):
        joern.full_run_joern(fpath1, verbose=3)

    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        joern.full_run_joern(fpath2, verbose=3)


if __name__ == "__main__":
    NUM_JOBS = 5
    JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(sys.argv[1]) - 1

    df = bigvul()
    df = df.iloc[::-1]

    splits = np.array_split(df, NUM_JOBS)

    utils.dfmp(
        splits[JOB_ARRAY_NUMBER],
        preprocess,
        ordr=False,
        workers=8
    )