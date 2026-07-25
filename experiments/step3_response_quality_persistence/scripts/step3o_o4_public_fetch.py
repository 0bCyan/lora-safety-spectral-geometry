#!/usr/bin/env python3
"""Bound one HTTPS JSON retrieval for the Step3O-O4 drand client."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("socket_timeout_seconds", type=int)
    parser.add_argument("maximum_response_bytes", type=int)
    args = parser.parse_args()
    if (
        not args.url.startswith("https://")
        or args.socket_timeout_seconds <= 0
        or args.maximum_response_bytes <= 0
    ):
        raise ValueError("invalid bounded public-fetch arguments")
    request = urllib.request.Request(
        args.url,
        headers={"Accept": "application/json", "User-Agent": "step3o-o4-protocol/1"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=args.socket_timeout_seconds
        ) as response:
            if response.geturl() != args.url:
                sys.stderr.write("bounded public fetch rejected a redirect\n")
                raise SystemExit(1)
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(min(65536, args.maximum_response_bytes + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > args.maximum_response_bytes:
                    raise OSError("public response exceeded the frozen byte limit")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        sys.stderr.write(f"bounded public fetch failed: {type(exc).__name__}\n")
        raise SystemExit(75) from exc
    sys.stdout.buffer.write(b"".join(chunks))


if __name__ == "__main__":
    main()
