"""Offline-only tests for PhishGuard safe_web_fetcher.py.

Run from the project root: python -m unittest test_safe_web_fetcher.py -v
This file does not make outbound HTTP or DNS requests.
"""

import socket
import unittest
from unittest.mock import patch

from detection_logic.INFO_POUCH import safe_web_fetcher as fetcher


class SafeWebFetcherTests(unittest.TestCase):
    def test_blocks_local_and_metadata_addresses(self):
        bad_urls = (
            "http://127.0.0.1/account/verify",
            "http://10.2.3.4/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://localhost/admin",
            "http://sub.localhost/admin",
            "http://machine.internal/private",
            "ftp://example.com/file",
            "https://example.com:8443",
            "https://user:pass@example.com/",
        )
        with patch.object(fetcher, "_request_one_hop") as request:
            for url in bad_urls:
                with self.subTest(url=url):
                    result = fetcher.fetch_webpage_features(url)
                    self.assertFalse(result["inspection_performed"])
                    self.assertIn("Not inspected:", result["webpage_status"])
            request.assert_not_called()

    def test_mixed_public_private_dns_answers_are_blocked(self):
        answers = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ]
        with patch.object(fetcher.socket, "getaddrinfo", return_value=answers), \
             patch.object(fetcher, "_request_one_hop") as request:
            result = fetcher.fetch_webpage_features("https://example.com/")
            self.assertFalse(result["inspection_performed"])
            self.assertIn("non-public IP", result["webpage_status"])
            request.assert_not_called()

    def test_public_html_page_features(self):
        html = b"""<html><head><title>Demo login</title></head><body>
        <form action='https://other.example.org/collect'>
          <input type='password' name='pass'></form><iframe></iframe>
        </body></html>"""
        with patch.object(fetcher, "_resolve_public_target", return_value="8.8.8.8"), \
             patch.object(fetcher, "_request_one_hop", return_value=(
                 200, {"content-type": "text/html; charset=utf-8"}, html
             )) as request:
            result = fetcher.fetch_webpage_features("http://example.com/login")
        self.assertTrue(result["inspection_performed"])
        self.assertEqual(result["page_has_password_field"], 1)
        self.assertEqual(result["external_action_forms"], 1)
        self.assertEqual(result["page_has_iframe"], 1)
        self.assertEqual(result["page_form_count"], 1)
        self.assertEqual(result["redirect_count"], 0)
        request.assert_called_once()

    def test_redirect_to_private_destination_is_blocked(self):
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))]
        with patch.object(fetcher.socket, "getaddrinfo", return_value=answers), \
             patch.object(fetcher, "_request_one_hop", return_value=(
                 302, {"location": "http://127.0.0.1/internal"}, b""
             )) as request:
            result = fetcher.fetch_webpage_features("http://example.com/")
        self.assertFalse(result["inspection_performed"])
        self.assertIn("non-public IP", result["webpage_status"])
        request.assert_called_once()  # only the public first hop was requested

    def test_https_downgrade_redirect_is_blocked(self):
        with patch.object(fetcher, "_resolve_public_target", return_value="8.8.8.8"), \
             patch.object(fetcher, "_request_one_hop", return_value=(
                 301, {"location": "http://example.com/"}, b""
             )) as request:
            result = fetcher.fetch_webpage_features("https://example.com/")
        self.assertFalse(result["inspection_performed"])
        self.assertIn("HTTPS-to-HTTP", result["webpage_status"])
        request.assert_called_once()

    def test_tls_pool_uses_pinned_ip_and_original_hostname(self):
        class FakeResponse:
            status = 200
            headers = {"Content-Type": "text/html"}
            def read(self, *_args, **_kwargs):
                return b""  # no response body required for this unit test
            def close(self):
                pass

        class FakePool:
            def urlopen(self, *_args, **_kwargs):
                return FakeResponse()
            def close(self):
                pass

        with patch.object(fetcher.urllib3, "HTTPSConnectionPool", return_value=FakePool()) as cls:
            status, _headers, _body = fetcher._request_one_hop(
                "8.8.8.8", "https", "example.com", 443, "/login", float("inf")
            )
            self.assertEqual(status, 200)
            self.assertEqual(cls.call_args.kwargs["host"], "8.8.8.8")
            self.assertEqual(cls.call_args.kwargs["server_hostname"], "example.com")
            self.assertEqual(cls.call_args.kwargs["assert_hostname"], "example.com")
            self.assertTrue(cls.call_args.kwargs["ssl_context"].verify_mode)

    def test_public_ip_resolution(self):
        answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))]
        with patch.object(fetcher.socket, "getaddrinfo", return_value=answers):
            self.assertEqual(fetcher._resolve_public_target("example.com", 443), "8.8.8.8")


if __name__ == "__main__":
    unittest.main()
