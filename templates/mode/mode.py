"""Implement application hooks; lifecycle is provided by AetherSwitch."""


def check():
    raise NotImplementedError("Validate prepared local files, environment and resolved arguments")


def listen_ports():
    return []  # Declare actual listeners, including request-scoped workers.


def build_launch():
    raise NotImplementedError("Return absolute argv/cwd and a string environment")


def probe():
    raise NotImplementedError("Read-only API check: healthy/accepting_requests/busy booleans")
