#!/usr/bin/env python3
"""Count morphological analyses in the corpora stored by language.

Both plain-text (``.txt``) and XML (``.xml``) files below each directory in
``lang_corpora`` are tokenised.  Every distinct token is analysed once with
UralicNLP, then each returned analysis is added once for every occurrence of
that token.  Consequently ambiguous tokens contribute to every one of their
analyses, as intended.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

try:
    from uralicNLP import uralicApi
except ImportError:  # pragma: no cover - depends on the user's installation
    sys.exit(
        "uralicNLP is required. Run this script with the system python3 that has "
        "UralicNLP installed."
    )


# Retain hyphenated and apostrophe-containing forms, which occur in the target
# languages, but require at least one alphabetic character below.
TOKEN_RE = re.compile(r"[^\W_]+(?:[-'\u2019\u02bc][^\W_]+)*", re.UNICODE)
SUPPORTED_SUFFIXES = {".txt", ".xml"}


def tokens(text: str) -> Iterator[str]:
    """Yield NFC-normalised word tokens from text."""
    for match in TOKEN_RE.finditer(unicodedata.normalize("NFC", text)):
        token = match.group(0)
        if any(character.isalpha() for character in token):
            yield token


def corpus_files(language_directory: Path) -> list[Path]:
    """Return corpus source files, excluding Finder archive metadata."""
    return sorted(
        path
        for path in language_directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and "__MACOSX" not in path.parts
        and not path.name.startswith("._")
    )


def xml_text_chunks(path: Path) -> Iterator[str]:
    """Stream text and tails from XML without retaining the complete tree."""
    for _event, element in ET.iterparse(path, events=("end",)):
        if element.text:
            yield element.text
        if element.tail:
            yield element.tail
        element.clear()


def file_tokens(path: Path) -> Iterator[str]:
    """Yield tokens from a UTF-8 TXT or XML corpus file."""
    if path.suffix.lower() == ".txt":
        # Text corpora are modest compared with the XML corpora.  Reading them
        # by line also avoids breaking a word at a buffered chunk boundary.
        with path.open(encoding="utf-8-sig", errors="replace") as source:
            for line in source:
                yield from tokens(line)
        return

    for chunk in xml_text_chunks(path):
        yield from tokens(chunk)


def morphology(analysis: object) -> str | None:
    """Return the tag portion of an UralicNLP analysis.

    ``uralicApi.analyze`` normally returns ``(reading, weight)`` tuples, with
    readings such as ``lemma+N+Sg+Nom``.  Accept a string as well, which keeps
    this small adapter compatible with API versions returning bare readings.
    """
    reading = analysis if isinstance(analysis, str) else analysis[0]  # type: ignore[index]
    if not isinstance(reading, str) or "+" not in reading:
        return None
    _lemma, tags = reading.split("+", 1)
    return tags or None


def count_morphologies(
    language: str, files: Iterable[Path]
) -> tuple[Counter[str], dict[str, int], list[str]]:
    """Analyse corpus files and return frequencies, statistics, and errors."""
    token_counts: Counter[str] = Counter()
    errors: list[str] = []
    file_count = 0

    for path in files:
        file_count += 1
        try:
            token_counts.update(file_tokens(path))
        except (OSError, ET.ParseError, UnicodeError) as error:
            errors.append(f"{path}: {error}")

    frequencies: Counter[str] = Counter()
    analysed_tokens = 0
    for token, occurrences in tqdm(
        token_counts.items(), desc=f"Analysing {language}", unit="type", dynamic_ncols=True
    ):
        try:
            readings = uralicApi.analyze(token, language)
            analyses = [tag for reading in readings if (tag := morphology(reading))]
        except Exception as error:  # Do not discard an entire corpus for one token.
            errors.append(f"{token}: {error}")
            analyses = []
        if analyses:
            analysed_tokens += occurrences
            for tags in analyses:
                frequencies[tags] += occurrences

    statistics = {
        "files": file_count,
        "tokens": sum(token_counts.values()),
        "unique_tokens": len(token_counts),
        "tokens_with_analysis": analysed_tokens,
        "unique_morphologies": len(frequencies),
    }
    return frequencies, statistics, errors


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
        "source", type=Path, nargs="?", default=Path("lang_corpora"),
        help="directory containing one subdirectory per language (default: lang_corpora)",
    )
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("morphology_frequencies"),
        help="directory for <language>.json frequency files (default: morphology_frequencies)",
    )
    parser.add_argument("--language", help="process only this language directory")
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"source directory not found: {args.source}")
    try:
        directories = language_directories(args.source, args.language)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)

    had_errors = False
    for directory in directories:
        files = corpus_files(directory)
        if not files:
            print(f"{directory.name}: no TXT or XML files; skipped", file=sys.stderr)
            continue
        print(f"{directory.name}: scanning {len(files)} corpus files", file=sys.stderr)
        frequencies, statistics, errors = count_morphologies(directory.name, files)
        output_path = args.output / f"{directory.name}.json"
        result = {
            "language": directory.name,
            "statistics": statistics,
            "frequencies": dict(frequencies.most_common()),
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{directory.name}: wrote {len(frequencies)} morphologies to {output_path}",
            file=sys.stderr,
        )
        if errors:
            had_errors = True
            error_path = args.output / f"{directory.name}.errors.txt"
            error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
            print(
                f"{directory.name}: {len(errors)} errors recorded in {error_path}",
                file=sys.stderr,
            )
    return 1 if had_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
