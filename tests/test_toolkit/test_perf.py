import unittest

from core.toolkit.perf import format_num_bytes
from core.toolkit.perf import MemoryMonitor


class TestPerf(unittest.TestCase):
    def test_format_num_bytes(self):
        self.assertEqual(format_num_bytes(1023), "1023.00B")
        self.assertEqual(format_num_bytes(1024), "1.00KB")
        self.assertEqual(format_num_bytes(1024**2), "1.00MB")

    def test_memory_monitor(self):
        MemoryMonitor().table()


if __name__ == "__main__":
    unittest.main()
