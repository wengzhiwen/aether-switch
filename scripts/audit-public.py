"""Scan the Git index (or HEAD), never private deployment directories.

Run before committing/pushing and in CI. This is a publication guard, not a
complete secret scanner. Findings print paths/rule names, never matching values.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
RULES = {
    "personal home path": re.compile(rb"/(?:home|Users)/[A-Za-z0-9._-]+/"),
    "real GPU identity": re.compile(rb"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"),
    "hardware filesystem identity": re.compile(rb"\b[0-9A-F]{16}\b"),
    "private key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "GitHub credential": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "private network address": re.compile(rb"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b"),
    "credentials in URL": re.compile(rb"https?://[^\s/@]+:[^\s/@]+@"),
}
PRIVATE_PATHS = ("modes/", ".runtime/", ".local/", ".venv/")
PRIVATE_FILES = {"machine.toml", "controller-version.json", "docs/controller-validation.md",
                 "tests/validate_wan_interrupt.py"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def audit(head=False):
    paths = git("ls-tree", "-r", "--name-only", "-z", "HEAD") if head else git("ls-files", "--cached", "-z")
    findings = []
    count = 0
    for raw in paths.split(b"\0"):
        if not raw:
            continue
        name = raw.decode()
        count += 1
        if name.startswith(PRIVATE_PATHS) or name in PRIVATE_FILES:
            findings.append((name, "private deployment file"))
        data = git("show", ("HEAD:" if head else ":") + name)
        if len(data) > 1_000_000:
            findings.append((name, "unexpected large artifact"))
        if b"\x00" in data:
            findings.append((name, "unexpected binary artifact"))
        for rule, pattern in RULES.items():
            if pattern.search(data):
                findings.append((name, rule))
    if not count:
        raise RuntimeError("No indexed files; stage the intended public files before auditing")
    for name, rule in findings:
        print(f"{name}: {rule}", file=sys.stderr)
    print(f"Audited {count} public files; {len(findings)} findings")
    return bool(findings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", action="store_true", help="Audit committed HEAD instead of the index")
    args = parser.parse_args()
    sys.exit(audit(args.head))
