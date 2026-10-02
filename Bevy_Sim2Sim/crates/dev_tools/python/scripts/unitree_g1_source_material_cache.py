#!/usr/bin/env python3
"""Cache original relative USD/MDL material files with bounded, hashed downloads.

This prepares storage only. It does not certify rendered materials, license
coverage or task capability. Built-in MDL module names remain runtime-resolved.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--background-url', required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    origin = urllib.parse.urlparse(args.background_url)
    if origin.scheme != 'https' or origin.hostname != 'omniverse-content-staging.s3-us-west-2.amazonaws.com':
        parser.error('Use the frozen original NVIDIA staging asset origin')
    cache = args.cache_root.resolve()
    inventory = json.loads(args.inventory.read_text())['t1_background']['assets']
    builtin = sorted({x['path'] for x in inventory if not x['path'].startswith('.')})
    pending = {urllib.parse.urljoin(args.background_url, x['path']) for x in inventory if x['path'].startswith('.')}
    visited = set()
    results = []
    total = 0

    def fetch(url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != origin.scheme or parsed.netloc != origin.netloc or parsed.query or parsed.fragment:
            raise ValueError(f'Unexpected dependency origin: {url}')
        path = (cache / parsed.path.lstrip('/')).resolve()
        if not path.is_relative_to(cache):
            raise ValueError('Dependency escaped cache')
        result = {'url': url, 'path': str(path)}
        try:
            if path.exists():
                data = path.read_bytes()
                result['cached'] = True
            else:
                with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as head:
                    etag = head.headers.get('ETag')
                    size = int(head.headers['Content-Length'])
                if not etag or not 0 < size <= 128 * 1024 * 1024:
                    raise ValueError('Missing identity or dependency exceeds 128 MiB')
                with urllib.request.urlopen(urllib.request.Request(url, headers={'If-Match': etag}), timeout=45) as reply:
                    if reply.headers.get('ETag') != etag:
                        raise ValueError('HEAD/GET identity mismatch')
                    data = reply.read(size + 1)
                if len(data) != size:
                    raise ValueError('HEAD/GET size mismatch')
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    stream.write(data)
                result.update(etag=etag, cached=False)
            result.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            dependencies = set()
            if path.suffix == '.mdl':
                source = data.decode('utf-8')
                result['module_imports'] = sorted(set(re.findall(r'\bimport\s+([^;]+);', source)))
                for literal in re.findall(r'"([^"\n]+)"', source):
                    if re.search(r'\.(png|jpg|jpeg|exr|hdr|mdl)$', literal, re.IGNORECASE):
                        dependencies.add(urllib.parse.urljoin(url, literal))
            return result, dependencies
        except Exception as error:
            result['error'] = repr(error)
            return result, set()

    with args.output.open('x') as output, ThreadPoolExecutor(max_workers=3) as pool:
        for depth in range(8):
            batch = sorted(pending - visited)
            if not batch:
                break
            if len(visited) + len(batch) > 256:
                raise ValueError('Dependency graph exceeds 256 files')
            pending = set()
            for result, children in pool.map(fetch, batch):
                visited.add(result['url'])
                results.append(result)
                total += result.get('bytes', 0)
                if total > 2 * 1024 * 1024 * 1024:
                    raise ValueError('Dependency cache exceeds 2 GiB')
                pending.update(children)
                print(json.dumps({'event': 'asset', 'url': result['url'], 'bytes': result.get('bytes'), 'error': result.get('error')}), flush=True)
            receipt = {'schema': 'g1_source_material_cache_v1', 'qualified': False,
                'background_url': args.background_url, 'builtin_assets': builtin,
                'files': results, 'total_bytes': total,
                'rendered_materials_verified': False,
                'file_references_complete': not any('error' in x for x in results) and not (pending - visited)}
            output.seek(0); output.truncate(); json.dump(receipt, output, indent=2); output.write('\n'); output.flush()
        if pending - visited or any('error' in x for x in results):
            raise ValueError('Original material dependency cache is incomplete; inspect receipt')


if __name__ == '__main__':
    main()
