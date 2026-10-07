#!/usr/bin/env python3
"""Convert predictive-model JSON into an AOSP/FUTO Keyboard wordlist.

The primary output is an AOSP ``.combined`` wordlist. It can be compiled to
the binary ``.dict`` format accepted by FUTO Keyboard and HeliBoard with the
standard AOSP dicttool. Invoke it with a language code, such as
``--language sms``. The input is streamed twice, so a large model does not
have to fit in memory.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO

AOSP_LOG_BASE = 1.15
DEFAULT_MIN_FREQUENCY = 15
DEFAULT_MAX_FREQUENCY = 255
DEFAULT_DICTTOOL = Path("dictool/dicttool_aosp.jar")


class ConversionError(ValueError):
    """A predictive-model file cannot be converted safely."""


class JsonArrayReader:
    """Read the top-level ``forms`` array without loading the whole JSON file."""

    def __init__(self, source: TextIO, chunk_size: int = 1024 * 1024):
        self.source, self.chunk_size = source, chunk_size
        self.buffer, self.position, self.eof = "", 0, False
        self.decoder = json.JSONDecoder()

    def _fill(self) -> bool:
        if self.eof:
            return False
        if self.position:
            self.buffer, self.position = self.buffer[self.position:], 0
        chunk = self.source.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _skip_space(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._fill():
                return

    def _character(self) -> str:
        self._skip_space()
        if self.position >= len(self.buffer):
            raise ConversionError("unexpected end of JSON")
        return self.buffer[self.position]

    def _take(self, expected: str) -> None:
        if self._character() != expected:
            raise ConversionError(f"expected {expected!r} in JSON")
        self.position += 1

    def _value(self) -> Any:
        self._skip_space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if not self._fill():
                    raise ConversionError(f"invalid JSON: {error.msg}") from error

    def forms(self) -> Iterator[dict[str, Any]]:
        self._take("{")
        while True:
            if self._character() == "}":
                raise ConversionError("top-level JSON object has no forms array")
            key = self._value()
            if not isinstance(key, str):
                raise ConversionError("top-level JSON object has a non-string key")
            self._take(":")
            if key == "forms":
                self._take("[")
                break
            self._value()
            separator = self._character()
            if separator == ",":
                self.position += 1
            elif separator != "}":
                raise ConversionError("expected ',' or '}' in top-level JSON object")
        while True:
            if self._character() == "]":
                self.position += 1
                return
            record = self._value()
            if not isinstance(record, dict):
                raise ConversionError("forms array contains a non-object record")
            yield record
            separator = self._character()
            if separator == ",":
                self.position += 1
            elif separator != "]":
                raise ConversionError("expected ',' or ']' in forms array")


def iter_forms(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as source:
            yield from JsonArrayReader(source).forms()
    except OSError as error:
        raise ConversionError(str(error)) from error


def model_language(path: Path) -> str:
    """Read only the model header to determine its language."""
    try:
        with path.open(encoding="utf-8") as source:
            reader = JsonArrayReader(source)
            reader._take("{")
            while True:
                key = reader._value()
                reader._take(":")
                value = reader._value()
                if key == "language":
                    if isinstance(value, str) and value:
                        return value
                    raise ConversionError("model language must be a non-empty string")
                if reader._character() != ",":
                    break
                reader.position += 1
    except OSError as error:
        raise ConversionError(str(error)) from error
    raise ConversionError("top-level JSON object has no language field")


def usable_form(record: dict[str, Any]) -> tuple[str, float] | None:
    word, probability = record.get("surface"), record.get("probability")
    if not isinstance(word, str) or not word or not isinstance(probability, (int, float)):
        return None
    if not math.isfinite(probability) or probability <= 0 or any(c in word for c in "\r\n,="):
        return None
    return word, float(probability)


def aosp_frequency(probability: float, largest: float, minimum: int, maximum: int) -> int:
    """Map a model probability to AOSP's logarithmic 0--255 score."""
    if probability <= 0 or largest <= 0:
        raise ConversionError("probabilities must be positive")
    raw = maximum + math.log(probability / largest, AOSP_LOG_BASE)
    return max(minimum, min(maximum, int(round(raw))))


@dataclass(frozen=True)
class ConversionResult:
    output: Path
    language: str
    forms_written: int
    forms_skipped: int
    maximum_probability: float


def convert_model(source: Path, destination: Path, *, language: str | None = None,
                  dictionary: str = "main", description: str | None = None,
                  date: int | None = None, version: str = "1", minimum: int = DEFAULT_MIN_FREQUENCY,
                  maximum: int = DEFAULT_MAX_FREQUENCY) -> ConversionResult:
    if not (0 <= minimum <= maximum <= 255):
        raise ConversionError("frequency bounds must satisfy 0 <= min <= max <= 255")
    language = language or model_language(source)
    largest = max((form[1] for record in iter_forms(source) if (form := usable_form(record))), default=0.0)
    if largest <= 0:
        raise ConversionError("model contains no usable forms")
    if destination.exists():
        raise ConversionError(f"{destination}: exists (use --overwrite)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    # AOSP uses the dictionary header as a dictionary identity. ``main:myv``
    # is the conventional main-dictionary identity; accept a fully qualified
    # value too for specialised dictionary types.
    dictionary_id = dictionary if ":" in dictionary else f"{dictionary}:{language}"
    header = {"dictionary": dictionary_id, "locale": language,
              "description": description or f"{language} predictive model",
              "date": str(int(time.time()) if date is None else date), "version": version}
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            output.write(",".join(f"{name}={value}" for name, value in header.items()) + "\n")
            for record in iter_forms(source):
                form = usable_form(record)
                if not form:
                    skipped += 1
                    continue
                word, probability = form
                output.write(f" word={word},f={aosp_frequency(probability, largest, minimum, maximum)}\n")
                written += 1
    except OSError as error:
        raise ConversionError(str(error)) from error
    return ConversionResult(destination, language, written, skipped, largest)


def compile_dictionary(combined: Path, output: Path, dicttool: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise ConversionError(f"{output}: exists (use --overwrite)")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["java", "-jar", str(dicttool), "makedict", "-s", str(combined), "-d", str(output)], check=True)
    except FileNotFoundError as error:
        raise ConversionError("Java is required to compile a .dict file") from error
    except subprocess.CalledProcessError as error:
        raise ConversionError(f"dicttool failed with exit code {error.returncode}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, help="language code, e.g. sms")
    parser.add_argument("--models", type=Path, default=Path("predictive_models"),
                        help="predictive-model directory (default: predictive_models)")
    parser.add_argument("--output", type=Path, default=Path("futo_dictionaries"),
                        help="dictionary-output directory (default: futo_dictionaries)")
    parser.add_argument("--dictionary", default="main", help="AOSP dictionary type (default: main)")
    parser.add_argument("--description", help="dictionary description")
    parser.add_argument("--date", type=int, help="Unix timestamp stored in the dictionary header")
    parser.add_argument("--version", default="1", help="dictionary version stored in the header")
    parser.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
    parser.add_argument("--max-frequency", type=int, default=DEFAULT_MAX_FREQUENCY)
    parser.add_argument("--dicttool", type=Path, default=DEFAULT_DICTTOOL,
                        help="AOSP dicttool JAR (default: dictool/dicttool_aosp.jar)")
    parser.add_argument("--dict-output", type=Path, help="binary .dict output; defaults under --output")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = parser.parse_args()
    source = args.models / f"{args.language}.json"
    destination = args.output / f"{args.language}.combined"
    if not source.is_file():
        parser.error(f"predictive model does not exist: {source}")
    if not args.dicttool.is_file():
        parser.error(f"dicttool JAR does not exist: {args.dicttool}")
    if destination.exists() and args.overwrite:
        destination.unlink()
    try:
        result = convert_model(source, destination, language=args.language, dictionary=args.dictionary,
                               description=args.description, date=args.date, version=args.version,
                               minimum=args.min_frequency, maximum=args.max_frequency)
        print(f"{source} -> {result.output} ({result.forms_written} forms; {result.forms_skipped} skipped)")
        binary = args.dict_output or args.output / f"{args.dictionary}_{result.language}.dict"
        compile_dictionary(destination, binary, args.dicttool, args.overwrite)
        print(f"{destination} -> {binary}")
    except ConversionError as error:
        print(f"{source}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
