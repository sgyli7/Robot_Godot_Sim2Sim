#!/usr/bin/env python3
"""Bounded local Unix-socket T1 ONNX bridge for an isolated source container.

Uses the same checked decoder/five-graph model as the native localhost service.
Receives only RGB and measured self-state. No simulator or acceptance truth is
accessible to inference. This is a source reproduction tool, not a game service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

from unitree_g1_static_server import (
    action_chunk, decode_request, parse_json, verify_receipt, require_full_policy, write_capture,
)
from unitree_g1_static_onnx import StaticAppleOnnx


def exact(connection, length):
    value = bytearray()
    while len(value) < length:
        part = connection.recv(length - len(value))
        if not part:
            raise ValueError('Source container disconnected')
        value.extend(part)
    return bytes(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-calls', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.max_calls <= 8 or args.socket.exists():
        parser.error('Use 1..8 calls and a fresh socket identity')
    args.capture_dir.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'g1_t1_source_policy_bridge_v1', 'qualified': False,
        'successful_inferences': 0, 'failed_inferences': 0, 'owned_model_closed': False,
        'device': 'cuda', 'model_receipt': str(args.receipt), 'source_rollout_verified': False}
    with args.output.open('x') as output:
        try:
            started = time.monotonic()
            policy = StaticAppleOnnx(verify_receipt(args.receipt), 'cuda', small_only=False)
            require_full_policy(policy)
            result['load_seconds'] = time.monotonic() - started
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(args.socket))
                # This exact socket is in the container's private evidence mount;
                # host UID 1000 and original Isaac container UID 1234 differ.
                args.socket.chmod(0o666)
                server.listen(1)
                server.settimeout(360)
                print(json.dumps({'event': 'ready', 'socket': str(args.socket), 'provider': policy.provider}), flush=True)
                for _ in range(args.max_calls):
                    connection, _ = server.accept()
                    with connection:
                        connection.settimeout(30)
                        try:
                            length = int.from_bytes(exact(connection, 4), 'big')
                            if not 0 < length <= 2_097_152:
                                raise ValueError('Source request exceeds finite bound')
                            body = parse_json(exact(connection, length))
                            if body == {'stop': True}:
                                encoded = b'{"stopped":true}'
                                connection.sendall(len(encoded).to_bytes(4, 'big') + encoded)
                                break
                            observation = decode_request(body)
                            started = time.monotonic()
                            outputs, timings = policy.infer(observation, seed=body['sequence_id'])
                            elapsed = time.monotonic() - started
                            reply = action_chunk(body, outputs)
                            write_capture(args.capture_dir, body, observation, outputs, reply, timings, elapsed)
                            result['successful_inferences'] += 1
                            print(json.dumps({'event': 'inference', 'sequence_id': body['sequence_id'], 'seconds': elapsed}), flush=True)
                        except Exception as error:
                            result['failed_inferences'] += 1
                            reply = {'error': repr(error)}
                        encoded = json.dumps(reply, allow_nan=False).encode()
                        if len(encoded) > 1_048_576:
                            raise ValueError('Source reply exceeds finite bound')
                        connection.sendall(len(encoded).to_bytes(4, 'big') + encoded)
                    output.seek(0); output.truncate(); json.dump(result, output, indent=2); output.flush()
            del policy
            result['owned_model_closed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if args.socket.exists():
                args.socket.unlink()
            output.seek(0); output.truncate(); json.dump(result, output, indent=2); output.write('\n'); output.flush()


if __name__ == '__main__':
    main()
