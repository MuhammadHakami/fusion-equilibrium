"""Fusion test fold switch (Final phase readiness): QW_FOLD names the test-fold directory (data/<fold>/*.parquet and
data/sophelio_proc/<fold>/*.npz); TAG is the suffix of every per-fold feature file ("public" for diii_d_public_test, else
"private"), so the Development files stay byte-identical by default.

    QW_FOLD=diii_d_private_test .venv/bin/python paper/strat/fusion/<feature script> ...
"""
from __future__ import annotations

import os

FOLD = os.environ.get("QW_FOLD", "diii_d_public_test")
TAG = "public" if FOLD == "diii_d_public_test" else "private"
