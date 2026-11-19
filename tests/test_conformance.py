"""The adapter `lerobot_smolvla` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "lerobot_smolvla"


if __name__ == "__main__":
    unittest.main()
