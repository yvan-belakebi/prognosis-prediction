"""Check that index_wsis resolves duplicate slide names deterministically.

Run: python test_index_wsis.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stain_feats_staged import index_wsis


def test_duplicate_prefers_clean_dir():
    """Same slide in 2016_anon and 2016_anon_problem -> the clean one wins."""
    with tempfile.TemporaryDirectory() as root:
        for sub in ("2016_anon", "2016_anon_problem/2016_missing_info"):
            os.makedirs(os.path.join(root, sub))
            open(os.path.join(root, sub, "2016_110873_ANON.ndpi"), "wb").close()

        index = index_wsis(root)

        assert index["2016_110873_ANON"] == os.path.join(
            "2016_anon", "2016_110873_ANON.ndpi"
        ), index["2016_110873_ANON"]


if __name__ == "__main__":
    test_duplicate_prefers_clean_dir()
    print("ok")
