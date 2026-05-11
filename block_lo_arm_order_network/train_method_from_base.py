#!/usr/bin/env python3
import sys

from train_clean_aogpt import main


if __name__ == "__main__":
    if "--run-kind" not in sys.argv:
        sys.argv.extend(["--run-kind", "graph_rw"])
    main(default_run_kind="graph_rw")
