import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

from .core import Controller, ManagerError


def main():
    parser = argparse.ArgumentParser(prog="mgr", description="AetherSwitch: exclusive GPU mode orchestration")
    parser.add_argument("action", choices=("list", "status", "check", "switch", "start", "restart", "stop", "force-stop", "logs", "register", "_run"))
    parser.add_argument("mode", nargs="?", help="Mode ID or machine.toml alias (1, 2, 3, vllm, strata, wan)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        controller = Controller(Path(__file__).resolve().parent.parent)
        mode = controller.mode(args.mode) if args.mode else None
        if args.action not in {"list", "status", "stop", "force-stop"} and mode is None:
            parser.error("this action requires a mode")
        if args.action == "list":
            result = [{"mode": item.id, "name": item.cfg.get("name", item.id)} for item in controller.modes()]
        elif args.action == "status":
            result = controller.status(mode)
            if not args.json:
                for item in result["modes"]:
                    print(f"{item['mode']}: {item['state']}")
                    if item.get("error"):
                        print("  " + item["error"].replace("\n", " "))
                op = result.get("operation")
                for conflict in result.get("conflicts", []):
                    print("conflict: " + conflict)
                if op and op["in_progress"]:
                    print(f"operation: {op['action']} {op['target']} / {op['stage']}")
                return
        elif args.action in {"switch", "start", "restart"}:
            result = controller.switch(mode, restart=args.action == "restart")
        elif args.action in {"stop", "force-stop"}:
            result = controller.stop(mode, force=args.action == "force-stop")
        elif args.action == "check":
            result = controller.check(mode)
        elif args.action == "register":
            with controller.lock("register", mode.id):
                result = controller.register(mode)
        elif args.action == "_run":
            controller.internal_run(mode)
            return
        else:
            logfile = mode.runtime / "server.log"
            if not logfile.exists():
                raise ManagerError("No server log: " + str(logfile))
            subprocess.run(["tail", "-n", "80", *( ["-F"] if args.follow else []), str(logfile)], check=True)
            return
        if args.json or args.action in {"check", "list", "register"}:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.action in {"switch", "start", "restart"}:
            print(f"{result['mode']}: {result['state']}" + (" (unchanged)" if result.get("unchanged") else ""))
        else:
            print(f"{result.get('mode', 'machine')}: stopped")
    except KeyboardInterrupt:
        print("Interrupted; inspect mgr status. Logs and completed cleanup are preserved.", file=sys.stderr)
        sys.exit(130)
    except (ManagerError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
