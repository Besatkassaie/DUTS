"""Single entry point for running the released experiments.

    python main.py                                   # prompts for everything it needs
    python main.py --system duts --dataset santos3 \
        --dataset-path data/santos3 \
        --index-path artifacts/santos3/index \
        --embedding-path data/santos3/vectors --defaults
    python main.py --help                            # every option

Missing prerequisites (embeddings, value-distribution synopsis, HNSW index) are detected and,
after confirmation, built; then rerun the printed command. See experiments/entry/app.py.
"""
import sys

from experiments.entry.app import main

if __name__ == "__main__":
    sys.exit(main())
