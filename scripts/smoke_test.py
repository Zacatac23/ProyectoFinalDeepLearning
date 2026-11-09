"""End-to-end check of a running API, local or public. Uses only the standard library.

Usage:
    python scripts/smoke_test.py http://localhost:8000
    python scripts/smoke_test.py https://your-service.onrender.com --latency 100
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

STARTUP_TIMEOUT_S = 150  # Render's free plan takes about a minute to wake up
LOG_TIMEOUT_S = 30  # the log is written in the background, after the response
LATENCY_PAUSE_S = 0.25  # keeps the measurement from saturating a 0.1 CPU instance


def request(url: str, body: Optional[dict] = None, timeout: float = 60):
    data = json.dumps(body).encode('utf-8') if body is not None else None
    headers = {'Content-Type': 'application/json', 'User-Agent': 'fraude-smoke-test'}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.status, response.read().decode('utf-8')


def wait_until_up(base: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        try:
            status, body = request(f'{base}/health', timeout=30)
            if status == 200:
                print(f'/health: {body}')
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if time.monotonic() > deadline:
                sys.exit(f'The API did not start within {STARTUP_TIMEOUT_S} s: {exc}')
        time.sleep(3)


def logged_requests(base: str) -> int:
    return json.loads(request(f'{base}/stats')[1])['requests']


def score(base: str, example: dict) -> dict:
    return json.loads(request(f'{base}/score', body=example)[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('base_url')
    parser.add_argument('--latency', type=int, default=0, help='Extra requests used to measure client latency.')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')

    wait_until_up(base)
    examples = json.loads(request(f'{base}/ejemplos')[1])
    before = logged_requests(base)

    normal = score(base, examples['normal'])
    suspicious = score(base, examples['sondeo_y_vaciado'])
    print(f"normal: {normal['fraud_risk_score']} {normal['decision']}")
    print(f"suspicious: {suspicious['fraud_risk_score']} {suspicious['decision']}")
    assert normal['decision'] == 'APPROVE', normal
    assert suspicious['decision'] == 'BLOCK_TRANSACTION', suspicious

    deadline = time.monotonic() + LOG_TIMEOUT_S
    while logged_requests(base) < before + 2:
        assert time.monotonic() < deadline, 'The requests never reached the usage log.'
        time.sleep(1)

    csv_lines = request(f'{base}/logs.csv')[1].strip().splitlines()
    assert csv_lines[0].startswith('timestamp_utc,model,score'), csv_lines[0]
    assert len(csv_lines) >= before + 3, f'/logs.csv has only {len(csv_lines)} lines.'
    print(f'usage log: {len(csv_lines) - 1} rows')

    if args.latency:
        times, errors = [], 0
        for _ in range(args.latency):
            start = time.perf_counter()
            try:
                score(base, examples['sondeo_y_vaciado'])
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                errors += 1
                continue
            times.append((time.perf_counter() - start) * 1000.0)
            time.sleep(LATENCY_PAUSE_S)
        assert times, 'Every latency request failed.'
        times.sort()
        print(f'client latency over {len(times)} requests ({errors} failed): p50 {statistics.median(times):.1f} ms | '
              f'p95 {times[int(0.95 * (len(times) - 1))]:.1f} ms | max {times[-1]:.1f} ms')
    print('OK: the API responds, decides and records its usage.')


if __name__ == '__main__':
    main()
