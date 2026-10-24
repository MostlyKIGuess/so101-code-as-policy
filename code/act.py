"""Send one step of Python to the live robot server (scripts/demo.py serve) and print what came back.

    python scripts/act.py "look()"
    python scripts/act.py "push_mouse()"
    python scripts/act.py -f step.py
    python scripts/act.py --shutdown

The step runs in the server's namespace, which keeps its variables between steps and
holds `demo`, `np`, `skills` and the skills bound to the arm: look, home, pick_pen,
place_pen, push_mouse, point_at_remote. Exits 1 if the step raised.
"""

import argparse
import json
import socket
import sys
from pathlib import Path

SOCKET = Path("/tmp/cap-so101-robot.sock")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code", nargs="?")
    parser.add_argument("-f", "--file", type=Path)
    parser.add_argument("--shutdown", action="store_true")
    args = parser.parse_args()
    if args.shutdown:
        request = {"shutdown": True}
    elif args.file is not None:
        request = {"code": args.file.read_text()}
    elif args.code is not None:
        request = {"code": args.code}
    else:
        parser.error("give code, -f FILE or --shutdown")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(SOCKET))
        with connection.makefile("rwb") as stream:
            stream.write((json.dumps(request) + "\n").encode())
            stream.flush()
            line = stream.readline()
    if not line:
        raise RuntimeError("the server closed the connection without a reply")
    result = json.loads(line)
    if result.get("output"):
        print(result["output"], end="" if result["output"].endswith("\n") else "\n")
    if result.get("value") is not None:
        print("=>", result["value"])
    if result.get("error"):
        print(result["error"], end="", file=sys.stderr)
    print(f"[step {result.get('step')}, {result.get('seconds')} s] frame {result.get('frame')} sim {result.get('sim')}")
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
