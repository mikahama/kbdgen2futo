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
from typing import Any, Iterable

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

DEFAULT_MORPH_LOOKUP = Path(__file__).resolve().parent / "morph_lookup"
DEFAULT_BLACKLIST_DIR = Path(__file__).resolve().parent / "black_list"


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
    surface = parts[0].split(":", 1)[0] if ":" in parts[0] else parts[0]
    return surface, parts[1:]


def load_blacklist(
    source: Path | set[str] | list[str] | dict[str, Any] | None,
    language: str,
) -> set[str]:
    """Load the blacklisted POS tags for a given language if available.

    Returns a set of blacklisted POS tag strings.
    """
    if source is None:
        return set()
    if isinstance(source, (set, list, tuple)):
        return set(source)
    if isinstance(source, dict):
        if language in source:
            val = source[language]
            return set(val) if isinstance(val, (list, set, tuple)) else set()
        return set(source.keys())

    path = Path(source)
    if not path.exists() and DEFAULT_BLACKLIST_DIR.is_dir():
        path = DEFAULT_BLACKLIST_DIR

    if path.is_dir():
        file_path = path / f"{language}.json"
        if file_path.is_file():
            with file_path.open(encoding="utf-8") as f:
                data = json.load(f)
                return set(data) if isinstance(data, list) else set(data.keys())
        return set()
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set(data.keys())
    return set()


def load_morph_lookup(
    source: Path | dict[str, Any] | None,
    language: str,
) -> dict[str, list[str]] | None:
    """Load the POS-to-tags lookup table for a given language if available.

    Returns a mapping of POS to a list of morphological tag strings, or
    ``None`` if no lookup data is available for the language.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        if language in source and isinstance(source[language], dict):
            return source[language]
        if all(isinstance(v, dict) for v in source.values()):
            return None
        return source

    path = Path(source)
    if not path.exists() and DEFAULT_MORPH_LOOKUP.is_dir():
        path = DEFAULT_MORPH_LOOKUP

    if path.is_dir():
        file_path = path / f"{language}_lookup.json"
        if file_path.is_file():
            with file_path.open(encoding="utf-8") as f:
                return json.load(f)
        file_path = path / f"{language}.json"
        if file_path.is_file():
            with file_path.open(encoding="utf-8") as f:
                return json.load(f)
        return None
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return None


def build_vocabulary(
    language: str,
    lexc_files: list[Path],
    morph_lookup: Path | dict[str, Any] | None = DEFAULT_MORPH_LOOKUP,
    black_list: Path | set[str] | list[str] | dict[str, Any] | None = DEFAULT_BLACKLIST_DIR,
) -> tuple[dict[str, list[list[str]]], list[str]]:
    """Inflect all unique entries and group their analyses by surface form.

    If a POS is found in black_list, the word is not inflected and is stored as
    is with the POS as its only morphological tag.
    If the (language, POS) combination is found in morph_lookup, inflections
    are generated via uralicApi.generate using the tags from the lookup table.
    Otherwise, uralicApi.get_all_forms is used as a fallback.
    """
    pairs = extract_lemma_pos_pairs(lexc_files)
    vocabulary: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    failures: list[str] = []

    blacklist = load_blacklist(black_list, language)
    raw_lookup = load_morph_lookup(morph_lookup, language)
    lookup: dict[str, list[tuple[str, tuple[str, ...]]]] | None = None
    if raw_lookup is not None:
        lookup = {
            pos: [
                (tag.lstrip("+"), tuple(t for t in tag.lstrip("+").split("+") if t))
                for tag in tags
            ]
            for pos, tags in raw_lookup.items()
        }

    for lemma, pos in tqdm(
        sorted(pairs), desc=f"Inflecting {language}", unit="lemma", dynamic_ncols=True
    ):
        if pos in blacklist:
            vocabulary[lemma].add((pos,))
        elif lookup is not None and pos in lookup:
            for tag_str, tag_parts in lookup[pos]:
                query = f"{lemma}+{tag_str}"
                try:
                    forms = uralicApi.generate(query, language)
                except Exception as error:
                    failures.append(f"{query}: {error}")
                    continue
                for surface, _weight in forms:
                    if surface:
                        vocabulary[surface].add(tag_parts)
        else:
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
    parser.add_argument(
        "--morph-lookup",
        type=Path,
        default=DEFAULT_MORPH_LOOKUP,
        help="directory containing morphological lookup JSON files (default: morph_lookup)",
    )
    parser.add_argument(
        "--black-list",
        "--blacklist",
        dest="black_list",
        type=Path,
        default=DEFAULT_BLACKLIST_DIR,
        help="directory containing <language>.json blacklist files (default: black_list)",
    )
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
        vocabulary, failures = build_vocabulary(
            directory.name,
            files,
            morph_lookup=args.morph_lookup,
            black_list=args.black_list,
        )
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
