"""
Build a versioned Sphinx site into a single output tree.

Layout produced under ``site/``::

    site/
        index.html                # version selector landing page
        versions.json             # metadata for the version switcher
        stable/                    # alias -> most recent released tag
        latest/                    # alias -> current main / dev build
        versions/<slug>/...        # immutable per-version build (e.g. 0.1.0)

The script is intentionally light: it shells out to ``sphinx-build`` once per
version and copies the result into place. It tracks which versions have already
been built by writing ``versions.json`` and skipping any slug whose entry
matches the requested source commit / tag. ``--prune`` removes entries that
are no longer referenced (e.g. a tag was deleted).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SITE_DIR = DOCS_DIR / "_build" / "site"
VERSIONS_FILE = SITE_DIR / "versions.json"
MAX_KEPT = 20

SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def slugify(version: str) -> str:
    slug = SLUG_RE.sub("-", version.strip()).strip("-")
    return slug or "unknown"


def load_index() -> dict:
    if VERSIONS_FILE.exists():
        try:
            return json.loads(VERSIONS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"stable": None, "latest": None, "versions": {}}


def save_index(index: dict) -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_FILE.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def sh(*args: str, cwd: Path | None = None) -> None:
    print(f"+ {' '.join(args)}", flush=True)
    subprocess.run(list(args), cwd=cwd or REPO_ROOT, check=True)


def build_version(
    *,
    version_string: str,
    out_dir: Path,
    is_latest: bool = False,
) -> None:
    """Build the docs once with ``version_string`` exported as FLASHDRR_DOCS_VERSION.

    The docs conf.py picks that env var up to override ``release`` so the
    banner / version selector can show the actual version instead of the
    default (which is whatever flashdrr.__version__ is at build time).
    ``is_latest`` is also exported as FLASHDRR_DOCS_IS_LATEST so the
    in-page switcher can label this build as the moving development build.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["FLASHDRR_DOCS_VERSION"] = version_string
    env["FLASHDRR_DOCS_IS_LATEST"] = "1" if is_latest else ""
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-q",
        str(DOCS_DIR),
        str(out_dir),
    ]
    print(
        f"+ {' '.join(cmd)}  (FLASHDRR_DOCS_VERSION={version_string},"
        f" latest={is_latest})",
        flush=True,
    )
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def write_alias(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        # Windows without dev mode / non-symlink filesystem: fall back to copy.
        shutil.copytree(src, dst)


def write_landing() -> None:
    """Drop a minimal landing page that links into ``latest``/``stable``."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FlashDRR documentation</title>
<meta http-equiv="refresh" content="0; url=latest/">
<link rel="canonical" href="latest/">
</head>
<body>
<p><a href="latest/">Latest (development)</a> &middot;
   <a href="stable/">Stable</a> &middot;
   <a href="versions.json">versions.json</a></p>
</body>
</html>
"""
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")


def discover_tags() -> list[str]:
    """Return release tags sorted by semver-ish order, newest first.

    Tags may be either ``v1.2.3`` or bare ``1.2.3``; both are accepted and
    the leading ``v`` is stripped so they all normalize to the same slug.
    """
    out = subprocess.run(
        ["git", "tag", "--list", "--sort=-v:refname"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tags = []
    for line in out.stdout.splitlines():
        t = line.strip()
        if not t:
            continue
        # Skip non-release tags (e.g. build metadata, pre-releases that look
        # like ``x.y.z-<suffix>`` are still kept; only exclude obvious noise).
        tags.append(t.removeprefix("v"))
    return tags


def _prune_only() -> int:
    index = load_index()
    tags = discover_tags()
    keep = {slugify(t) for t in tags}
    for v in (index.get("latest"), index.get("stable")):
        if isinstance(v, str):
            keep.add(v)
    versions_dir = SITE_DIR / "versions"
    for old in list(index["versions"].keys()):
        if old not in keep:
            old_path = versions_dir / old
            if old_path.exists():
                shutil.rmtree(old_path)
            index["versions"].pop(old, None)
    for alias in ("latest", "stable"):
        if index.get(alias) not in index["versions"]:
            index[alias] = None
    if index.get("stable") and (versions_dir / index["stable"]).exists():
        write_alias(versions_dir / index["stable"], SITE_DIR / "stable")
    else:
        (SITE_DIR / "stable").unlink(missing_ok=True)
    if index.get("latest") and (versions_dir / index["latest"]).exists():
        write_alias(versions_dir / index["latest"], SITE_DIR / "latest")
    else:
        (SITE_DIR / "latest").unlink(missing_ok=True)
    save_index(index)
    write_landing()
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--alias",
        choices=["latest", "stable"],
        help="Which alias slot to publish this build into.",
    )
    p.add_argument(
        "--version",
        help="Human-readable version label (e.g. '0.1.0' or 'dev').",
    )
    p.add_argument(
        "--ref",
        default="",
        help="Git ref / commit the build came from (used for skip detection).",
    )
    p.add_argument(
        "--prune",
        action="store_true",
        help="Remove versioned dirs not present in the tag list.",
    )
    p.add_argument(
        "--prune-only",
        action="store_true",
        help="Skip the rebuild; just reconcile versions.json + dirs against tags.",
    )
    args = p.parse_args()

    if args.prune_only:
        return _prune_only()

    if not args.alias or not args.version:
        p.error("--alias and --version are required unless --prune-only is set")

    index = load_index()
    slug = slugify(args.version)
    versions_dir = SITE_DIR / "versions"
    version_path = versions_dir / slug

    # --- Mutate the in-memory index for the new build, then persist BEFORE
    # running sphinx so the per-page static asset (flashdrr-versions.js)
    # carries the new state, not the previous one.
    entry = {"version": args.version, "ref": args.ref}
    index["versions"][slug] = entry
    index[args.alias] = slug

    # Cap the number of retained versions. We never drop the alias targets.
    keep = set(index["versions"].keys())
    keep.update(v for v in (index.get("latest"), index.get("stable")) if v)
    if len(index["versions"]) > MAX_KEPT:
        ordered = sorted(
            index["versions"].keys(),
            key=lambda slug: index["versions"][slug].get("version", ""),
            reverse=True,
        )
        keep = set(ordered[:MAX_KEPT])
        for s in (index.get("latest"), index.get("stable")):
            if s:
                keep.add(s)
    for old in list(index["versions"].keys()):
        if old not in keep:
            del index["versions"][old]

    if args.prune:
        tags = discover_tags()
        keep = {slugify(tag) for tag in tags}
        for alias in ("latest", "stable"):
            v = index.get(alias)
            if isinstance(v, str):
                keep.add(v)
        for old in list(index["versions"].keys()):
            if old not in keep:
                del index["versions"][old]
        for alias in ("latest", "stable"):
            if index.get(alias) not in index["versions"]:
                index[alias] = None

    # Write versions.json and the page-local data file now. Sphinx will
    # copy the latter into every page's _static/ when it runs.
    save_index(index)

    # --- Run sphinx into a temp dir so a failed build never leaves a
    # half-baked version behind on the Pages artifact.
    tmp_path = DOCS_DIR / "_build" / f"_tmp_{slug}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    try:
        build_version(
            version_string=args.version,
            out_dir=tmp_path,
            is_latest=(args.alias == "latest"),
        )
    except subprocess.CalledProcessError:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise

    if version_path.exists():
        shutil.rmtree(version_path)
    shutil.move(str(tmp_path), str(version_path))

    # Now that the new build is in place, drop any on-disk versioned dirs
    # that the in-memory prune removed above.
    survivors = set(index["versions"].keys())
    if versions_dir.exists():
        for entry in versions_dir.iterdir():
            if entry.is_dir() and entry.name not in survivors:
                shutil.rmtree(entry)

    # Refresh aliases.
    if index.get("stable") and (versions_dir / index["stable"]).exists():
        write_alias(versions_dir / index["stable"], SITE_DIR / "stable")
    else:
        (SITE_DIR / "stable").unlink(missing_ok=True)
    if index.get("latest") and (versions_dir / index["latest"]).exists():
        write_alias(versions_dir / index["latest"], SITE_DIR / "latest")
    else:
        (SITE_DIR / "latest").unlink(missing_ok=True)

    # save_index was already called before sphinx ran so the data file would
    # be picked up by the static asset copy step; do not call it again.
    write_landing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
