#!/usr/bin/env python3
"""Finite source-container Unix transport to an existing IPv4 loopback policy owner.

Forwards original request/reply bytes without changing stamps, RGB, actions or
model ownership. This process loads no model and exposes no network listener.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import urllib.request

MAX_REQUEST = 2_097_152
MAX_REPLY = 1_048_576


def exact(connection, length):
    value = bytearray()
    while len(value) < length:
        part = connection.recv(length - len(value))
        if not part:
            raise ValueError('Source socket disconnected mid-frame')
        value.extend(part)
    return bytes(value)


def forward_request(raw, port):
    if not 0 < len(raw) <= MAX_REQUEST or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Invalid source transport bound or loopback port')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(f'http://127.0.0.1:{port}/infer', data=raw,
                                     headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=30) as response:
        reply = response.read(MAX_REPLY + 1)
    if not 0 < len(reply) <= MAX_REPLY:
        raise ValueError('Policy reply exceeds source transport bound')
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--max-calls', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.socket = args.socket.resolve()
    if args.socket.exists() or not 1 <= args.port <= 65535 or not 1 <= args.max_calls <= 30:
        parser.error('Use a fresh private socket, valid loopback port and1..30 calls')
    receipt = {'schema': 'g1_source_http_transport_v1', 'qualified': False,
        'model_loaded_here': False, 'model_owner_lifecycle_changed': False,
        'endpoint': f'http://127.0.0.1:{args.port}/infer', 'socket': str(args.socket),
        'request_reply_bytes_transformed': False, 'calls': [], 'closed': False,
        'transport_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    started = time.monotonic()
    with args.output.open('x') as output:
        def save():
            output.seek(0); output.truncate()
            json.dump(receipt, output, indent=2, allow_nan=False); output.write('\n'); output.flush()
        save()
        bound = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                os.chdir(args.socket.parent)
                server.bind(args.socket.name); bound = True
                # Original source image UID1234 and host UID1000 differ.
                args.socket.chmod(0o666)
                server.listen(1); server.settimeout(360)
                print('G1_SOURCE_TRANSPORT_READY', flush=True)
                for _ in range(args.max_calls):
                    connection, _ = server.accept()
                    with connection:
                        connection.settimeout(35)
                        length = int.from_bytes(exact(connection, 4), 'big')
                        if not 0 < length <= MAX_REQUEST:
                            raise ValueError('Source request exceeds transport bound')
                        raw = exact(connection, length)
                        if json.loads(raw) == {'stop': True}:
                            reply = b'{"stopped":true}'
                            connection.sendall(len(reply).to_bytes(4, 'big') + reply)
                            break
                        begin = time.monotonic()
                        reply = forward_request(raw, args.port)
                        connection.sendall(len(reply).to_bytes(4, 'big') + reply)
                        receipt['calls'].append({'request_sha256': hashlib.sha256(raw).hexdigest(),
                            'reply_sha256': hashlib.sha256(reply).hexdigest(), 'request_bytes': len(raw),
                            'reply_bytes': len(reply), 'wall_seconds': time.monotonic() - begin})
                        save()
                        print(f'G1_SOURCE_TRANSPORT calls={len(receipt["calls"])}', flush=True)
        except BaseException as error:
            receipt['error'] = repr(error)
            raise
        finally:
            if bound and args.socket.exists():
                args.socket.unlink()
            receipt.update(closed=True, wall_seconds=time.monotonic() - started)
            save()


if __name__ == '__main__':
    main()
