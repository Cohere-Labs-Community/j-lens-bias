"""Import alias for the submitted steering harness.

``cataluna84_week_3_followups.py`` and ``cataluna84_week_3_steering.py``
import the pre-submission module name ``week3_steering``.
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("cataluna84_week_3_steering")
