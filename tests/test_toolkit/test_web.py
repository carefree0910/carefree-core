import json
import aiohttp
import asyncio
import unittest

import numpy as np

from io import BytesIO
from PIL import Image
from PIL import ImageOps
from typing import Dict
from fastapi import Response
from fastapi import HTTPException
from pydantic import BaseModel
from unittest.mock import call
from unittest.mock import patch
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from core.toolkit import web
from core.toolkit.constants import WEB_ERR_CODE


class DataModel(BaseModel):
    field: str


class SuccessModel(BaseModel):
    field: str


class TestWeb(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    @patch("core.toolkit.web.get_host_name")
    @patch("socket.gethostbyname")
    def test_get_ip(self, mock_gethostbyname, mock_get_host_name):
        mock_get_host_name.return_value = "localhost"
        mock_gethostbyname.return_value = "127.0.0.1"
        result = web.get_ip()
        self.assertEqual(result, "127.0.0.1")

    @patch("socket.gethostname")
    def test_get_host_name(self, mock_gethostname):
        mock_gethostname.return_value = "localhost"
        result = web.get_host_name()
        self.assertEqual(result, "localhost")

    def test_get_responses(self):
        json_example = {"field": "example"}
        result = web.get_responses(SuccessModel, json_example=json_example)
        self.assertIsInstance(result, dict)
        self.assertIn(200, result)
        self.assertIn(WEB_ERR_CODE, result)
        self.assertEqual(result[200]["model"], SuccessModel)
        self.assertEqual(
            result[200]["content"]["application/json"]["example"], json_example
        )
        self.assertEqual(result[WEB_ERR_CODE]["model"], web.RuntimeError)

    def test_get_responses_without_json_example(self):
        result = web.get_responses(SuccessModel)
        self.assertIsInstance(result, dict)
        self.assertIn(200, result)
        self.assertIn(WEB_ERR_CODE, result)
        self.assertEqual(result[200]["model"], SuccessModel)
        self.assertNotIn("content", result[200])
        self.assertEqual(result[WEB_ERR_CODE]["model"], web.RuntimeError)

    def test_get_image_response_kwargs(self):
        result = web.get_image_response_kwargs()
        self.assertIsInstance(result, dict)
        self.assertIn("responses", result)
        self.assertIn("response_class", result)
        self.assertIn("response_description", result)
        self.assertIsInstance(result["responses"], dict)
        self.assertIsInstance(result["response_class"], Response)
        self.assertIsInstance(result["response_description"], str)
        self.assertIn(200, result["responses"])
        self.assertIn(WEB_ERR_CODE, result["responses"])
        self.assertIn("content", result["responses"][200])
        self.assertIn("image/png", result["responses"][200]["content"])
        self.assertIn("example", result["responses"][200]["content"]["image/png"])
        self.assertIsInstance(
            result["responses"][200]["content"]["image/png"]["example"], str
        )
        self.assertEqual(result["responses"][WEB_ERR_CODE]["model"], web.RuntimeError)

    @patch("core.toolkit.web.logging.exception")
    @patch("core.toolkit.web.get_err_msg")
    def test_raise_err(self, mock_get_err_msg, mock_log_exception):
        mock_get_err_msg.return_value = "error message"
        err = Exception("test error")

        with self.assertRaises(HTTPException) as context:
            web.raise_err(err)

        self.assertEqual(context.exception.status_code, WEB_ERR_CODE)
        self.assertEqual(context.exception.detail, "error message")
        mock_log_exception.assert_called_once_with(err)
        mock_get_err_msg.assert_called_once_with(err)

    def test_get(self):
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.read.return_value = b"mocked response"

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.get.return_value.__aenter__.return_value = mock_response

        url = "http://example.com"
        result = self.loop.run_until_complete(web.get(url, mock_session))
        mock_session.get.assert_called_once_with(url)
        mock_response.read.assert_called_once()

        self.assertEqual(result, b"mocked response")

    def test_post_function(self):
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.json.return_value = {"key": "value"}

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.post.return_value.__aenter__.return_value = mock_response

        url = "http://example.com"
        json_data = {"param": "value"}

        result = self.loop.run_until_complete(web.post(url, json_data, mock_session))
        mock_session.post.assert_called_once_with(url, json=json_data)
        mock_response.json.assert_called_once()

        self.assertEqual(result, {"key": "value"})

    @patch("core.toolkit.web.logging.debug")
    def test_log_endpoint(self, mock_debug):
        endpoint = "/test"
        data = DataModel(field="value")
        expected_msg = f"{endpoint} endpoint entered with kwargs : {json.dumps(data.model_dump(), ensure_ascii=False)}"

        web.log_endpoint(endpoint, data)

        mock_debug.assert_called_once_with(expected_msg)

    @patch("core.toolkit.web.logging.debug")
    def test_log_times(self, mock_debug):
        endpoint = "/test"
        times: Dict[str, float] = {"time1": 1.0, "time2": 2.0}
        expected_total = sum(times.values())
        expected_msg = f"elapsed time of endpoint {endpoint} : {json.dumps({**times, '__total__': expected_total})}"

        web.log_times(endpoint, times)

        mock_debug.assert_called_once_with(expected_msg)

    @patch("core.toolkit.web.download_raw")
    @patch.object(Image, "open")
    @patch.object(ImageOps, "exif_transpose")
    def test_download_image_success(
        self, mock_exif_transpose, mock_open, mock_download_raw
    ):
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        url = "http://example.com/image.jpg"
        raw_data = b"raw image data"
        raw_data_io = BytesIO(raw_data)
        mock_download_raw.return_value = raw_data
        mock_image = MagicMock(spec=Image.Image)
        mock_open.return_value = mock_image
        mock_exif_transpose.return_value = mock_image

        with patch(
            "core.toolkit.web.BytesIO",
            new_callable=lambda: lambda _: raw_data_io,
        ):
            result = self.loop.run_until_complete(web.download_image(mock_session, url))

        mock_download_raw.assert_called_once_with(mock_session, url)
        mock_open.assert_called_once_with(raw_data_io)
        mock_exif_transpose.assert_called_once_with(mock_image)
        self.assertEqual(result, mock_image)

    @patch("core.toolkit.web.download_raw")
    def test_download_image_download_raw_exception(self, mock_download_raw):
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        url = "http://example.com/image.jpg"
        mock_download_raw.side_effect = Exception("Download raw error")

        with self.assertRaises(ValueError) as context:
            self.loop.run_until_complete(web.download_image(mock_session, url))

        self.assertIn("raw | None | err | Download raw error", str(context.exception))

    @patch("core.toolkit.web.download_raw")
    @patch.object(Image, "open")
    def test_download_image_image_open_exception(self, mock_open, mock_download_raw):
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        url = "http://example.com/image.jpg"

        raw_data = b"raw image data"
        mock_download_raw.return_value = raw_data
        mock_open.side_effect = Exception("Image open error")
        with self.assertRaises(ValueError) as context:
            self.loop.run_until_complete(web.download_image(mock_session, url))
        self.assertEqual("raw image data", str(context.exception))

        mock_download_raw.return_value = None
        with self.assertRaises(ValueError) as context:
            self.loop.run_until_complete(web.download_image(mock_session, url))
        self.assertEqual("raw | None | err | Image open error", str(context.exception))

        raw_data_arr = np.random.randint(0, 256, [10, 10], dtype=np.uint8)
        raw_data = Image.fromarray(raw_data_arr).tobytes()
        mock_download_raw.return_value = raw_data
        with self.assertRaises(ValueError) as context:
            self.loop.run_until_complete(web.download_image(mock_session, url))
        self.assertEqual(
            f"raw | {raw_data[:20]!r} | err | Image open error",
            str(context.exception),
        )

    @patch("core.toolkit.web.time.sleep")
    @patch("core.toolkit.web.logging.warning")
    def test_retry_with(self, mock_warning, mock_sleep):
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        url = "http://example.com"
        mock_download_fn = AsyncMock(side_effect=[Exception("error"), "response"])
        retry = 2
        interval = 1

        result = self.loop.run_until_complete(
            web.retry_with(
                mock_download_fn, mock_session, url, retry=retry, interval=interval
            )
        )

        self.assertEqual(result, "response")
        mock_download_fn.assert_has_calls(
            [call(mock_session, url), call(mock_session, url)]
        )
        mock_warning.assert_called_once_with(f"succeeded after 1 retries")
        mock_sleep.assert_called_once_with(interval)

    @patch("core.toolkit.web.time.sleep")
    def test_retry_with_failure(self, mock_sleep):
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        url = "http://example.com"
        mock_download_fn = AsyncMock(side_effect=Exception("error"))
        retry = 2
        interval = 1

        with self.assertRaises(ValueError) as context:
            self.loop.run_until_complete(
                web.retry_with(
                    mock_download_fn, mock_session, url, retry=retry, interval=interval
                )
            )

        self.assertEqual(str(context.exception), "error\n(After 2 retries)")
        mock_download_fn.assert_has_calls(
            [call(mock_session, url), call(mock_session, url)]
        )
        mock_sleep.assert_has_calls([call(interval), call(interval)])

    @patch("core.toolkit.web.retry_with")
    def test_download_raw_with_retry(self, mock_retry_with):
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        url = "http://example.com"
        mock_retry_with.return_value = b"response"
        retry = 2
        interval = 1

        result = self.loop.run_until_complete(
            web.download_raw_with_retry(
                mock_session, url, retry=retry, interval=interval
            )
        )

        self.assertEqual(result, b"response")
        mock_retry_with.assert_called_once_with(
            web.download_raw, mock_session, url, retry, interval
        )


if __name__ == "__main__":
    unittest.main()
