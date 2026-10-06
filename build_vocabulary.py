#!/usr/bin/env python3
"""Build per-language vocabulary JSON files from Divvun lexc stem files.

Only ``*.lexc`` files are scanned.  Each usable lexical entry contributes its
lemma and its first morphological tag (the part of speech), for example
``antseptili+A:...`` becomes ``("antseptili", "A")``.  The pairs are
deduplicated before any calls to UralicNLP are made.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

try:
    from uralicNLP import uralicApi
except ImportError:  # pragma: no cover - depends on the user's installation
    sys.exit(
        "uralicNLP is required. Run this script with the system python3 that has "
        "UralicNLP installed."
    )


# POS tags in the dictionaries are ASCII identifiers such as N, V, A, Adv, or
# Pcle. Restricting this avoids treating continuation lexicon syntax as an entry.
POS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_/-]*$")


def split_unescaped_plus(text: str) -> tuple[str, str] | None:
    """Split the lexical side at its first unescaped ``+``.

    In lexc, a literal plus is written as ``%+``.  That belongs to the lemma,
    while the first unescaped plus starts the lexical tags.
    """
    escaped = False
    for index, character in enumerate(text):
        if escaped:
            escaped = False
        elif character == "%":
            escaped = True
        elif character == "+":
            return text[:index], text[index + 1 :]
    return None


def parse_entry(line: str) -> tuple[str, str] | None:
    """Extract ``(lemma, POS)`` from a single lexc entry, or return ``None``."""
    source = line.split("!", 1)[0].strip()
    if not source or ":" not in source:
        return None
    lexical_side = source.split(":", 1)[0].strip()
    split = split_unescaped_plus(lexical_side)
    if split is None:
        return None
    lemma, tags = split
    # The first tag is the POS. Further tags (e.g. +Prop) qualify the entry but
    # are not accepted by uralicApi.get_all_forms' POS argument.
    pos = tags.split("+", 1)[0]
    if not lemma or not POS_RE.fullmatch(pos):
        return None
    return lemma.replace("%+", "+"), pos


def extract_lemma_pos_pairs(files: Iterable[Path]) -> set[tuple[str, str]]:
    """Read lexc files and return distinct ``(lemma, POS)`` pairs."""
    pairs: set[tuple[str, str]] = set()
    for path in files:
        try:
            with path.open(encoding="utf-8") as source:
                for line in source:
                    entry = parse_entry(line)
                    if entry is not None:
                        pairs.add(entry)
        except UnicodeDecodeError as error:
            raise RuntimeError(f"{path}: not valid UTF-8") from error
    return pairs


def parse_form_reading(reading: str) -> tuple[str, list[str]] | None:
    """Convert ``surface+TAG+TAG`` returned by UralicNLP to JSON data."""
    parts = reading.split("+")
    if len(parts) < 2 or not parts[0]:
        return None
    return parts[0], parts[1:]


def build_vocabulary(language: str, lexc_files: list[Path]) -> tuple[dict[str, list[list[str]]], list[str]]:
    """Inflect all unique entries and group their analyses by surface form."""
    pairs = extract_lemma_pos_pairs(lexc_files)
    vocabulary: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    failures: list[str] = []

    for lemma, pos in tqdm(
        sorted(pairs), desc=f"Inflecting {language}", unit="lemma", dynamic_ncols=True
    ):
        try:
            forms = uralicApi.get_all_forms(lemma, pos, language)
        except Exception as error:  # Keep a long run useful if a single entry fails.
            failures.append(f"{lemma}+{pos}: {error}")
            continue
        for reading, _weight in forms:
            parsed = parse_form_reading(reading)
            if parsed is None:
                failures.append(f"{lemma}+{pos}: unrecognised UralicNLP reading {reading!r}")
                continue
            surface, tags = parsed
            vocabulary[surface].add(tuple(tags))

    # Stable ordering makes output reproducible and makes differences reviewable.
    return {
        surface: [list(tags) for tags in sorted(readings)]
        for surface, readings in sorted(vocabulary.items())
    }, failures


def language_directories(root: Path, selected_language: str | None) -> list[Path]:
    if selected_language:
        directory = root / selected_language
        if not directory.is_dir():
            raise ValueError(f"language directory not found: {directory}")
        return [directory]
    return sorted(path for path in root.iterdir() if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, nargs="?", default=Path("lexc_stems"),
        help="directory containing one subdirectory per language (default: lexc_stems)",
    )
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("vocabularies"),
        help="directory for <language>.json files (default: vocabularies)",
    )
    parser.add_argument("--language", help="build only this language directory")
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"source directory not found: {args.source}")
    try:
        directories = language_directories(args.source, args.language)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)

    had_failures = False
    for directory in directories:
        # ``Path.rglob`` also returns dangling editor lock symlinks. They have a
        # .lexc suffix but are not source files and cannot be opened.
        files = sorted(path for path in directory.rglob("*.lexc") if path.is_file())
        if not files:
            print(f"{directory.name}: no .lexc files; skipped", file=sys.stderr)
            continue
        print(f"{directory.name}: scanning {len(files)} lexc files", file=sys.stderr)
        vocabulary, failures = build_vocabulary(directory.name, files)
        output_path = args.output / f"{directory.name}.json"
        output_path.write_text(
            json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{directory.name}: wrote {len(vocabulary)} surface forms to {output_path}",
            file=sys.stderr,
        )
        if failures:
            had_failures = True
            error_path = args.output / f"{directory.name}.errors.txt"
            error_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
            print(
                f"{directory.name}: {len(failures)} failures recorded in {error_path}",
                file=sys.stderr,
            )
    return 1 if had_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
