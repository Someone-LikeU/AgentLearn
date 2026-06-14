# encoding: utf-8
import logging
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import external_mcp_manager  # noqa: F401


class ExternalMCPManagerTest(unittest.TestCase):
    def test_streamable_http_transport_logger_is_suppressed(self):
        logger = logging.getLogger("mcp.client.streamable_http")

        self.assertFalse(logger.propagate)
        self.assertGreaterEqual(logger.level, logging.CRITICAL)


if __name__ == "__main__":
    unittest.main()
