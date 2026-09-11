"""Compatibility shim for the FlagScale retrieval trainer.

The supported user entrypoint is now ``flagscale/train/train_retrieval.py``;
this module remains only for older local commands that referenced the original
example path.
"""

from flagscale.train.train_retrieval import main


if __name__ == "__main__":
    main()
