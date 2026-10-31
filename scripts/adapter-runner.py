"""Shared adapter protocol runner, executed ONLY with the selected mode's Python."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys


def check_ports(ports):
    for host, port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, int(port)))


def main():
    directory, filename, action = sys.argv[1:]
    directory = Path(directory)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("mode_adapter", directory / filename)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if action == "run":
        launch = adapter.build_launch()
        argv, cwd, env = launch["argv"], launch["cwd"], launch["env"]
        if not argv or not all(isinstance(item, str) for item in argv) or not Path(argv[0]).is_absolute():
            raise ValueError("build_launch.argv must begin with an absolute executable")
        if not Path(cwd).is_absolute() or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError("build_launch requires an absolute cwd and string environment")
        check_ports(adapter.listen_ports())
        recover = getattr(adapter, "recover_before_launch", None)
        if recover:
            recover()
        os.umask(0o077)
        os.chdir(cwd)
        # Do not log environment or arguments: adapters can reference private credentials.
        print("server_mgr: exec prepared mode recipe", flush=True)
        os.execve(argv[0], argv, env)
    else:
        if action == "ports":
            result = {"listen_ports": adapter.listen_ports()}
        else:
            result = adapter.check() if action == "check" else adapter.probe()
            if not isinstance(result, dict):
                raise ValueError("adapter hooks must return objects")
            if action == "check":
                result["listen_ports"] = adapter.listen_ports()
            if action == "probe":
                for key in ("healthy", "accepting_requests", "busy"):
                    if type(result.get(key)) is not bool:
                        raise ValueError("probe must return boolean " + key)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
