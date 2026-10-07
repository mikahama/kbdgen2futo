#!/usr/bin/env python3
"""Convert a Divvun kbdgen Android primary layout to a FUTO Keyboard layout.

The converter intentionally uses the phone-sized ``android.primary`` layout.
Desktop, tablet, and iOS kbdgen targets have different geometry and do not map
cleanly to FUTO Keyboard's mobile layout format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the user's installation
    sys.exit("PyYAML is required: install it with `python3 -m pip install PyYAML`.")


SPECIAL_KEYS = {
    "backspace": "$delete",
    "return": "$enter",
    "shift": "$shift",
    "shiftSymbols": "$symbols",
    "space": "$space",
    "spacer": "$gap",
}


class ConversionError(ValueError):
    """The source layout cannot be represented safely by this converter."""


def tokenise(layer: Any, layer_name: str) -> list[list[str]]:
    """Turn kbdgen's whitespace-separated multi-line layer into rows."""
    if not isinstance(layer, str):
        raise ConversionError(f"android.primary.layers.{layer_name} must be a string block")
    rows = [line.split() for line in layer.splitlines() if line.split()]
    if not rows:
        raise ConversionError(f"android.primary.layers.{layer_name} has no rows")
    return rows


def convert_special(token: str) -> str:
    """Convert a kbdgen special-key token, leaving ordinary key specs intact."""
    if not (token.startswith(r"\s{") and token.endswith("}")):
        return token
    body = token[3:-1]
    kind = body.split(":", 1)[0]
    if kind not in SPECIAL_KEYS:
        raise ConversionError(f"unsupported kbdgen special key {token!r}")
    return SPECIAL_KEYS[kind]


def futo_base_key(token: str, longpress: dict[str, list[str]]) -> Any:
    """Convert one kbdgen key and attach its explicit long-press choices."""
    converted = convert_special(token)
    if converted.startswith("$"):
        return converted
    more_keys = longpress.get(token, [])
    return [converted, *(convert_special(item) for item in more_keys)] if more_keys else converted


def futo_key(normal: str, shifted: str | None, longpress: dict[str, list[str]]) -> Any:
    """Return the shortest FUTO representation of one key position."""
    normal_key = futo_base_key(normal, longpress)
    if shifted is None:
        return normal_key
    shifted_key = futo_base_key(shifted, longpress)
    if normal_key == shifted_key:
        return normal_key
    if (isinstance(normal_key, str) and normal_key.startswith("$")) or (
        isinstance(shifted_key, str) and shifted_key.startswith("$")
    ):
        raise ConversionError(
            f"a special key changes between default and shift: {normal_key!r} / {shifted_key!r}"
        )
    return {"type": "case", "normal": normal_key, "shifted": shifted_key}


def longpress_keys(source: dict[str, Any]) -> dict[str, list[str]]:
    """Parse kbdgen's shared longpress mapping into FUTO moreKeys lists."""
    raw = source.get("longpress", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConversionError("longpress must be a mapping")
    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ConversionError("longpress keys and values must be strings")
        options = value.split()
        if options:
            result[key] = options
    return result


def source_metadata(source: dict[str, Any], source_path: Path) -> tuple[str, str]:
    language = source_path.stem
    names = source.get("displayNames", {})
    if not isinstance(names, dict):
        names = {}
    name = names.get(language) or names.get("en") or language
    if not isinstance(name, str):
        name = language
    return language, name


def convert_layout(source: dict[str, Any], source_path: Path, name: str | None,
                   language: str | None) -> dict[str, Any]:
    """Convert parsed kbdgen YAML to the FUTO YAML document data structure."""
    try:
        layers = source["android"]["primary"]["layers"]
    except (KeyError, TypeError) as error:
        raise ConversionError("missing android.primary.layers") from error
    if not isinstance(layers, dict) or "default" not in layers:
        raise ConversionError("missing android.primary.layers.default")

    default_rows = tokenise(layers["default"], "default")
    shifted_rows = tokenise(layers["shift"], "shift") if "shift" in layers else None
    if shifted_rows is not None and len(default_rows) != len(shifted_rows):
        raise ConversionError("default and shift layers have different numbers of rows")

    longpress = longpress_keys(source)
    result_rows: list[dict[str, list[Any]]] = []
    for row_index, default_row in enumerate(default_rows, start=1):
        shifted_row = shifted_rows[row_index - 1] if shifted_rows is not None else None
        if shifted_row is not None and len(default_row) != len(shifted_row):
            raise ConversionError(
                f"default and shift row {row_index} have different key counts"
            )
        if len(default_row) > 12:
            raise ConversionError(
                f"row {row_index} has {len(default_row)} keys; FUTO supports at most 12"
            )
        result_rows.append({
            "letters": [
                futo_key(key, shifted_row[position] if shifted_row else None, longpress)
                for position, key in enumerate(default_row)
            ]
        })

    detected_language, detected_name = source_metadata(source, source_path)
    result = {
        "name": name or detected_name,
        "languages": [language or detected_language],
        "rows": result_rows,
    }
    # kbdgen's map contains all long-press behavior. Prevent FUTO from adding
    # unrelated automatic alternatives on top of those explicit choices.
    if longpress:
        result["attributes"] = {"moreKeyMode": "OnlyExplicit"}
    return result


def load_and_convert(path: Path, name: str | None, language: str | None) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConversionError(str(error)) from error
    except yaml.YAMLError as error:
        raise ConversionError(f"invalid YAML: {error}") from error
    if not isinstance(document, dict):
        raise ConversionError("the YAML document must contain a mapping")
    return convert_layout(document, path, name, language)


def dump(document: dict[str, Any]) -> str:
    return yaml.safe_dump(
        document, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1000
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", help="kbdgen YAML file, or a directory of YAML files")
    parser.add_argument("destination", type=Path, nargs="?", help="output file, or output directory")
    parser.add_argument("--name", help="override the FUTO layout name (single-file conversion only)")
    parser.add_argument("--language", help="override the language code (single-file conversion only)")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True,
                        help="replace existing output files (default)")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="fail rather than replace an existing output file")
    args = parser.parse_args()

    # The normal project workflow converts every checked-in kbdgen layout. A
    # supplied source retains the original single-file and custom-directory
    # behaviours below.
    if args.source is None:
        args.source = Path("kbdgen_layouts")
        args.destination = Path("futo_layouts")
    if not args.source.exists():
        parser.error(f"source does not exist: {args.source}")
    if args.source.is_dir():
        if args.destination is None:
            parser.error("a destination directory is required when source is a directory")
        if args.name or args.language:
            parser.error("--name and --language are only supported for a single source file")
        sources = sorted((*args.source.rglob("*.yaml"), *args.source.rglob("*.yml")))
        if not sources:
            parser.error(f"no YAML files found in {args.source}")
        args.destination.mkdir(parents=True, exist_ok=True)
        failures = 0
        for source in sources:
            output = args.destination / f"{source.stem}.yaml"
            if output.exists() and not args.overwrite:
                print(f"{output}: exists (use --overwrite)", file=sys.stderr)
                failures += 1
                continue
            try:
                output.write_text(dump(load_and_convert(source, None, None)), encoding="utf-8")
                print(f"{source} -> {output}")
            except ConversionError as error:
                print(f"{source}: {error}", file=sys.stderr)
                failures += 1
        return 1 if failures else 0

    try:
        converted = dump(load_and_convert(args.source, args.name, args.language))
    except ConversionError as error:
        print(f"{args.source}: {error}", file=sys.stderr)
        return 1
    if args.destination is None:
        sys.stdout.write(converted)
        return 0
    if args.destination.exists() and not args.overwrite:
        print(f"{args.destination}: exists (use --overwrite)", file=sys.stderr)
        return 1
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(converted, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
