# kbdgen to FUTO Keyboard converter

`kbdgen2futo.py` converts the Android phone layout in a Divvun/kbdgen YAML
file into a [FUTO Keyboard layout](https://github.com/futo-org/futo-keyboard-layouts/blob/main/LayoutSpec.md).
It reads `android.primary.layers.default` and (when present) `shift`, producing
FUTO `case` keys wherever the two layers differ.

## Requirements

Python 3 and [PyYAML](https://pyyaml.org/):

```sh
python3 -m pip install PyYAML
```

## Use

Convert one layout to standard output:

```sh
python3 kbdgen2futo.py kbdgen_layouts/sms.yaml
```

Write one layout to a file:

```sh
python3 kbdgen2futo.py kbdgen_layouts/sms.yaml out/sms.yaml
```

Convert every YAML layout below a directory:

```sh
python3 kbdgen2futo.py kbdgen_layouts out/
```

Use `--overwrite` to replace outputs. For a single layout, `--name` and
`--language` override the name selected from `displayNames` and the language
code inferred from the source filename.

## Conversion scope

The converter intentionally targets kbdgen's Android `primary` layout. It
maps `\\s{shift}`, `\\s{backspace}`, `\\s{return}`, `\\s{space}`, and
`\\s{shiftSymbols}` to FUTO template keys. It does not attempt to translate
iOS/tablet/desktop layouts, spell-checker configuration, dead keys, or kbdgen
features that have no direct FUTO equivalent. Unsupported special keys and
incompatible row sizes stop conversion with a clear error instead of silently
producing a broken layout.

## Building a vocabulary from lexc stems

`build_vocabulary.py` scans only `.lexc` files below each language directory
and inflects unique `(lemma, POS)` pairs using UralicNLP. If a POS tag is listed
in `black_list/<language>.json`, the word is not inflected and is stored as is
with the POS tag as its only morphological analysis. If a `(language, POS)`
combination is present in `morph_lookup/`, tags from that lookup are used to
inflect words with `uralicApi.generate`; otherwise, `uralicApi.get_all_forms` is
used as a fallback. Run it with the system `python3` that provides UralicNLP:

```sh
python3 build_vocabulary.py lexc_stems vocabularies
```

To build a single language:

```sh
python3 build_vocabulary.py lexc_stems vocabularies --language sms
```

The result is one file per language, such as `vocabularies/sms.json`. Its keys
are surface forms. Every value is a list of morphological readings; each
reading is a list of tags returned after the surface form by UralicNLP. This
retains separate readings where different analyses have the same spelling.
Any individual entries UralicNLP cannot inflect are logged in a neighbouring
`.errors.txt` file, while the remaining vocabulary is still saved.

## Counting corpus morphology frequencies

`count_morphology_frequencies.py` reads all `.txt` and `.xml` corpus files
below each language directory in `lang_corpora`. It tokenizes their text and
uses `uralicApi.analyze` to count complete morphological tag sequences (for
example, `N+Sg+Nom`). Ambiguous words add to every returned analysis. Analyses
are cached per distinct token, so a token occurring many times is analysed only
once while all its occurrences are counted.

```sh
python3 count_morphology_frequencies.py lang_corpora morphology_frequencies
```

To process a single language:

```sh
python3 count_morphology_frequencies.py --language sms
```

The output `morphology_frequencies/<language>.json` contains metadata and a
`frequencies` mapping, sorted from the most frequent analysis to the least.
Language directories without corpus files (such as an as-yet-empty `mdf`
directory) are skipped.
