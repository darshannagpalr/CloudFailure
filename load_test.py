import requests
import argparse
import time
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate HTTP load against the service under test.")
    parser.add_argument("url", nargs="?", default=os.environ.get("TARGET_URL"), help="Target service URL")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--duration", type=float, default=0.0, help="Total runtime in seconds; 0 means run forever")
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("Provide the target URL as TARGET_URL or as the positional argument.")

    start_time = time.time()

    while True:
        try:
            requests.get(args.url, timeout=5)
            print("Request sent")
        except requests.RequestException:
            print("Error")

        if args.duration and time.time() - start_time >= args.duration:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()