"""Transport byte identity and bounds; no simulator/model inference."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import unittest

from unitree_g1_source_http_bridge import MAX_REPLY, MAX_REQUEST, exact, forward_request


class SourceTransportGuards(unittest.TestCase):
    def fixture(self, reply):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.path, self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200); self.send_header('Content-Length', str(len(reply))); self.end_headers()
                self.wfile.write(reply)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server.server_port, received

    def test_source_bytes_survive_without_stamp_or_frame_reserialization(self):
        reply = b'{ "sequence_id" : 41, "frames" : [1.23456789e-4], "text" : "\\u4e2d" }\n'
        port, received = self.fixture(reply)
        request = b'{ "observation": {"frame_id": 50, "sim_time_ns": 1000000000}, "camera_rgb_b64":"AB==" }\n'
        self.assertEqual(forward_request(request, port), reply)
        self.assertEqual(received, [('/infer', request)])

    def test_oversized_reply_and_request_are_rejected(self):
        port, _ = self.fixture(b'x' * (MAX_REPLY + 1))
        with self.assertRaises(ValueError):
            forward_request(b'{}', port)
        for request, port in [(b'', port), (b'x'*(MAX_REQUEST+1), port), (b'{}', True), (b'{}', 65536)]:
            with self.subTest(size=len(request), port=port), self.assertRaises(ValueError):
                forward_request(request, port)

    def test_short_socket_frame_is_not_padded(self):
        class ClosedConnection:
            def recv(self, length):
                return b''
        with self.assertRaises(ValueError):
            exact(ClosedConnection(), 4)


if __name__ == '__main__':
    unittest.main()
