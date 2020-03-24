"""
Python file containing methods used for testing outer modules
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import preprocessing as pp


class TestPreProcessing:
    def test_complex_preprocessing(self):
        assert pp.complex_processing("@hellparade what? whyyyy?") == "whyy"

    def test_simple_preprocessing(self):
        # TODO
        pass
