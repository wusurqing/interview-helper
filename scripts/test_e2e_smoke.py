import io
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from e2e_test import check_endpoints


class SmokeContractTest(unittest.TestCase):
    def test_contract_errors_fail_the_smoke_run(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status, body = self.server.options_reply if self.path == "/api/options" else (200, b"{}")
                if self.path == "/":
                    body = b"<html></html>"
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            do_POST = do_GET

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                for status, body, expected in [
                    (200, b"{}", 0), (400, b"{}", 1), (401, b"{}", 1),
                    (500, b"{}", 1), (200, b"<html></html>", 1),
                ]:
                    with self.subTest(status=status, body=body), redirect_stdout(io.StringIO()):
                        server.options_reply = status, body
                        self.assertEqual(check_endpoints(base), expected)
            finally:
                server.shutdown()
                thread.join()
