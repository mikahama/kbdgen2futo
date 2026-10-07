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
example, `N+Sg+Nom`). Analyses are cached per distinct token, so a token
occurring many times is analysed only once while all its occurrences are
counted. Each ambiguous token is split equally among its analyses, preventing
ambiguity from inflating counts.

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

## Building a predictive-text unigram model

`predictive_model.py` combines the existing target-language FST, corpus data,
vocabulary forms, and `fin_frequencies.json` into a per-language word-form
model:

```sh
python3 predictive_model.py --language sms
```

It writes `predictive_models/sms.json` (forms with probabilities, costs, and
their analyses), a compact `sms.report.json`, and a persistent lexical lookup
cache in `predictive_models/cache/`. Delete that cache only when dictionary
data has changed. Target surface forms are lemmatized by their target-language
FST analyses and translated with `get_translation(lemma, language, "fin")`.
Those Finnish translations are looked up directly, because
`fin_frequencies.json` is already lemma-keyed. A Finnish lemma's count is split
uniformly among all target lemmas reached by it.
Thus Finnish Common Crawl counts are first tempered and normalised within POS,
then used only as `mu` pseudo-counts; they are never added as target tokens.

For an analysis `POS + lemma + morphology`, the implemented probability is:

```
P(POS) * P(lemma | POS) * P(morphology | lemma, POS)
```

Ambiguous FST readings are summed for a surface form. During corpus collection
an ambiguous token gives each reading `1/k` evidence, rather than one full
count. The model enumerates bounded legal forms with `get_all_forms` and
renormalizes the POS morphology backoff distribution over each lemma's legal
morphologies. If enumeration fails, analysed vocabulary/corpus forms remain
the legal-form fallback.

The main tuning options are `--tau` (smaller flattens Finnish ranks), `--mu`
(Finnish-prior pseudo-count strength), `--kappa` (POS morphology smoothing),
`--rho` (lemma morphology smoothing), `--eta` (POS smoothing), `--epsilon`,
and `--probability-floor`. Defaults are `0.7`, `100`, `10`, `5`, `1`, `1e-8`,
and `1e-300` respectively. Legal generation is bounded by
`--legal-form-limit` (per lemma) and `--legal-enumeration-max-lemmas` (2,000
by default), so exceptionally large lexica remain buildable; the report notes
when the fallback was used. Individual translation/FST failures are retained
in the report rather than stopping a build. `tqdm` shows progress while reading
corpora, analysing forms, resolving Finnish mappings, and generating legal
forms; use `--no-progress` for quiet batch runs.

## Converting a predictive model to a FUTO/HeliBoard dictionary

`predictive_model2dict.py` converts a model in `predictive_models/` into the
AOSP combined wordlist format used by FUTO Keyboard and HeliBoard. It streams
the model twice, so even a multi-gigabyte model does not need to fit in memory.
Model probabilities are converted to AOSP's logarithmic `f=0..255` score;
relative ranking is retained and scores are clamped to `15..255` by default.

```sh
python3 predictive_model2dict.py --language myv
```

This reads `predictive_models/myv.json`, writes
`futo_dictionaries/myv.combined`, then uses the bundled
`dictool/dicttool_aosp.jar` to produce `futo_dictionaries/main_myv.dict`. Use
`--models` or `--output` only when those standard project directories have
been intentionally changed.

The resulting `.combined` file is the inspectable source dictionary. Compile
it with the AOSP dictionary tool to make the binary file that FUTO Keyboard and
HeliBoard import:

```sh
java -jar dicttool_aosp.jar makedict -s futo_dictionaries/myv.combined -d main_myv.dict
```

To use a different copy of the tool:

```sh
python3 predictive_model2dict.py --language myv \
  --dicttool path/to/dicttool_aosp.jar
```
