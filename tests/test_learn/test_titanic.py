import unittest
import subprocess

from pathlib import Path


ddp_task_path = str(Path(__file__).parent / "ddp_titanic_task.py")


class TestTitanic(unittest.TestCase):
    def test_titanic(self) -> None:
        cmd = ["python", ddp_task_path]
        subprocess.run(cmd, check=True)

    def test_titanic_ddp(self) -> None:
        cmd = ["accelerate", "launch", "--num_processes=2", ddp_task_path]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    unittest.main()
