from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_spec, scale_spec
from .manifest import artifact_summary, generate
from .projections import aggregate, challenge, mobility
from .utils import StorageError, assert_storage_safe, write_json
from .verify import verify


def _repository_root() -> Path:
    return Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hdp-synthetic-population", description="Deterministic synthetic Israeli HDP population generator")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "preflight"):
        command = commands.add_parser(name)
        command.add_argument("--spec", default=str(_repository_root() / "data/specs/native.json"))
        command.add_argument("--out", required=True)
        if name == "generate":
            command.add_argument("--force", action="store_true")
        else:
            command.add_argument("--bytes-per-patient", type=int)
    for name in ("verify", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True)
    command = commands.add_parser("scale-spec")
    command.add_argument("--from", dest="source", default=str(_repository_root() / "data/specs/native.json"))
    command.add_argument("--population-size", type=int, required=True)
    command.add_argument("--dataset-version", required=True)
    command.add_argument("--out", required=True)
    command.add_argument("--zero-problems", action="store_true")
    for name in ("aggregate", "challenge", "mobility"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True)
        command.add_argument("--destination", required=True)
        if name == "mobility":
            command.add_argument("--allow-restricted", action="store_true")
            command.add_argument("--sensitivity-code", action="append", default=[])
    command = commands.add_parser("reproduce-release", help="Reconstruct the accepted 1k/10k source bundles and enforce published SHA256 identities")
    command.add_argument("--population-size", type=int, choices=(1000, 10000), required=True)
    command.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out).resolve()
    try:
        if args.command == "reproduce-release":
            from .release_repairs import reproduce_release
            target = reproduce_release(args.population_size, out, _repository_root())
            print(json.dumps({"status": "reproduced", **artifact_summary(target)}, indent=2))
            return 0
        if args.command == "generate":
            generate(load_spec(Path(args.spec)), out, _repository_root(), args.force)
            result = verify(out)
            print(json.dumps({"status": "generated" if result["ok"] else "failed", **artifact_summary(out), "verification": result}, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "verify":
            result = verify(out)
            print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "inspect":
            result = artifact_summary(out)
        elif args.command == "preflight":
            result = assert_storage_safe(load_spec(Path(args.spec)), out, args.bytes_per_patient)
        elif args.command == "scale-spec":
            write_json(out, scale_spec(load_spec(Path(args.source)), args.population_size, args.dataset_version, zero_problems=args.zero_problems))
            result = {"spec": str(out)}
        elif args.command == "mobility":
            result = mobility(out, Path(args.destination).resolve(), allow_restricted=args.allow_restricted, authorized_sensitivity=tuple(args.sensitivity_code))
        else:
            result = {"aggregate": aggregate, "challenge": challenge}[args.command](out, Path(args.destination).resolve())
        print(json.dumps({"artifactPath": str(result)} if isinstance(result, Path) else result, indent=2))
        return 0
    except (ValueError, KeyError, OSError, StorageError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
