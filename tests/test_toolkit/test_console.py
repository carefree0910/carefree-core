import unittest

from unittest.mock import patch
from core.toolkit.console import *


class TestConsole(unittest.TestCase):
    def test_functions(self):
        log("Hello World!")
        debug("Hello World!")
        warn("Hello World!")
        deprecated("Hello World!")
        error("Hello World!")
        print("Hello World!")
        rule("Hello World!")
        status("Hello World!")

    @patch("rich.prompt.Prompt.ask")
    def test_ask_without_default(self, mock_ask):
        mock_ask.return_value = "yes"
        question = "Continue?"
        choices = ["yes", "no"]
        result = ask(question, choices)
        mock_ask.assert_called_once_with(question, choices=choices)
        self.assertEqual(result, "yes")

    @patch("rich.prompt.Prompt.ask")
    def test_ask_with_default(self, mock_ask):
        mock_ask.return_value = "no"
        question = "Continue?"
        choices = ["yes", "no"]
        default = "no"
        result = ask(question, choices, default=default)
        mock_ask.assert_called_once_with(question, choices=choices, default=default)
        self.assertEqual(result, "no")

    @patch("rich.prompt.Prompt.ask")
    def test_ask_with_kwargs(self, mock_ask):
        mock_ask.return_value = "yes"
        question = "Continue?"
        choices = ["yes", "no"]
        kwargs = {"show_default": True}
        result = ask(question, choices, **kwargs)
        mock_ask.assert_called_once_with(question, choices=choices, show_default=True)
        self.assertEqual(result, "yes")


if __name__ == "__main__":
    unittest.main()
