import unittest
import subprocess

from pathlib import Path

from ddp_titanic_task import main


ddp_task_path = str(Path(__file__).parent / "ddp_titanic_task.py")


class TestTitanic(unittest.TestCase):
    def test_titanic(self) -> None:
        main()

    def test_titanic_ddp(self) -> None:
        cmd = ["accelerate", "launch", "--num_processes=2", ddp_task_path]
        # subprocess.run(cmd, check=True)  # uncomment this line to run the test


if __name__ == "__main__":
    unittest.main()
