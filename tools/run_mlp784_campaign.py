#!/usr/bin/env python3
"""Run MLP784 P0/P1 validation: export assets, offline IC check, optional HEMPS sim campaign."""

import argparse
import glob
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS = os.path.dirname(__file__)
TC_DIR = os.path.join(REPO, "testcases", "mlp784")


def run(cmd, cwd=None, env=None):
    print("+ %s" % " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd or REPO, env=env)


def hemps_env(root):
    env = os.environ.copy()
    env["HEMPS_PATH"] = root
    env["PATH"] = os.path.join(root, "build_env", "bin") + os.pathsep + env.get("PATH", "")
    return env


def find_hemps_run(root):
    local = os.path.join(root, "build_env", "bin", "hemps-run")
    if os.path.isfile(local):
        return local
    from shutil import which
    return which("hemps-run")


def main():
    parser = argparse.ArgumentParser(description="MLP784 P0/P1 campaign runner")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--sim-time-ms", type=int, default=50)
    parser.add_argument("--hemps-root", default=os.environ.get("HEMPS_PATH", REPO))
    parser.add_argument("--run-sim", action="store_true", help="Run hemps-run if available")
    parser.add_argument("--bounds-json", default=None)
    args = parser.parse_args()

    if not args.skip_export:
        export_cmd = [sys.executable, os.path.join(TOOLS, "export_mlp784_assets.py")]
        if args.bounds_json:
            export_cmd += ["--bounds-json", args.bounds_json]
        run(export_cmd)

    if not args.skip_offline:
        run([sys.executable, os.path.join(TOOLS, "verify_mlp784_offline.py")])
        run([sys.executable, os.path.join(TOOLS, "verify_mlp784_offline.py"), "--corrupt-fc1"])

    if args.run_sim:
        hemps_run = find_hemps_run(args.hemps_root)
        if not hemps_run:
            print("hemps-run not found; install HEMPS toolchain and set HEMPS_PATH", file=sys.stderr)
            return 1

        env = hemps_env(args.hemps_root)
        os.makedirs(TC_DIR, exist_ok=True)
        for yaml in sorted(glob.glob(os.path.join(TC_DIR, "*.yaml"))):
            name = os.path.basename(yaml)
            log = os.path.join(TC_DIR, name.replace(".yaml", "_sim.txt"))
            print("=== sim %s ===" % name)
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.call(
                    [hemps_run, yaml, str(args.sim_time_ms)],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=TC_DIR,
                    env=env,
                )
            if rc != 0:
                print("WARN: hemps-run %s exit %d (see %s)" % (name, rc, log))

        run(
            [sys.executable, os.path.join(TOOLS, "parse_campaign_results.py"), TC_DIR, "--grid"],
            env=env,
        )

    print("MLP784 campaign complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
