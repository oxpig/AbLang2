"""Command-line interface to AbLang2.

Input is a CSV file with `heavy` and `light` columns (either may be blank
for a single chain) and an optional `id` column.  Per-sequence results
(`seqcoding`, `pseudo_log_likelihood`, `confidence`, `restore`) are written
as CSV; per-residue results (`rescoding`, `likelihood`, `probability`) are
written as a NumPy `.npz` archive with one array per sequence, keyed by id.
"""

import argparse
import csv
import sys

import numpy as np

MODES = [
    "seqcoding",
    "rescoding",
    "likelihood",
    "probability",
    "pseudo_log_likelihood",
    "confidence",
    "restore",
]
PER_SEQUENCE_MODES = {"seqcoding", "pseudo_log_likelihood", "confidence", "restore"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="ablang2", description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("input", help="CSV file with `heavy` and `light` columns")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output file: CSV for per-sequence modes, .npz for per-residue modes",
    )
    parser.add_argument(
        "--model",
        default="ablang2-paired",
        choices=["ablang2-paired", "tcrlang-paired", "ablang1-heavy", "ablang1-light"],
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="number the sequences with ANARCI and align the output "
        "(needs the `align` extra and ANARCI, which must be installed "
        "from Bioconda)",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--ncpu", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="e.g. `cpu` or `cuda`")
    return parser.parse_args(argv)


def read_input(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    missing = {"heavy", "light"} - set(columns)
    if missing:
        sys.exit(f"ablang2: input lacks column(s): {', '.join(sorted(missing))}")
    ids = [row.get("id") or str(n) for n, row in enumerate(rows)]
    sequences = [(row["heavy"], row["light"]) for row in rows]
    return ids, sequences


def write_csv(path, mode, ids, values):
    values = np.asarray(values)
    if values.ndim == 1:
        header = [mode]
        values = values.reshape(-1, 1)
    else:
        header = list(range(values.shape[1]))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *header])
        for i, row in zip(ids, values):
            writer.writerow([i, *row])


def main(argv=None):
    args = parse_args(argv)
    ids, sequences = read_input(args.input)

    # Imported late: loading torch is slow, and `--help` shouldn't wait.
    from .pretrained import pretrained

    model = pretrained(model_to_use=args.model, ncpu=args.ncpu, device=args.device)
    result = model(
        sequences, mode=args.mode, align=args.align, batch_size=args.batch_size
    )

    if args.mode in PER_SEQUENCE_MODES:
        write_csv(args.output, args.mode, ids, result)
    else:
        np.savez(args.output, **{str(i): np.asarray(r) for i, r in zip(ids, result)})


if __name__ == "__main__":
    main()
