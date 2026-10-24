# Build a FUTO Keyboard / HeliBoard layout and dictionary

This project turns two kinds of language resources into files that can be used
with **FUTO Keyboard** and **HeliBoard**:

1. a Divvun/kbdgen keyboard definition becomes a FUTO layout YAML file; and
2. a lexicon plus optional text corpus becomes a predictive-text dictionary
   (`.dict`).

The compiled predictive dictionaries created from the models are compatible
with both keyboards.

This guide is for adding a **new language**. The language must already be
supported by [UralicNLP](https://github.com/mikahama/uralicNLP): its analyser,
generator, and translation data are used to build the model. A layout alone can
be converted without UralicNLP, but predictive text cannot be built for a
language that UralicNLP does not support.

Use the language code recognised by UralicNLP consistently below. In the
examples, `xxx` means that code.

## Before you start

You need:

- Python 3 with `PyYAML`, `tqdm`, and UralicNLP installed;
- Java, to compile the final Android dictionary; and
- for a layout: a kbdgen YAML layout for the language;
- for predictive text: the language's `.lexc` source files and, preferably,
  UTF-8 plain-text (`.txt`) or XML (`.xml`) corpus files.

Install the Python dependencies in the same Python environment you will use to
run the scripts:

```sh
python3 -m pip install PyYAML tqdm uralicNLP
```

Then verify that UralicNLP can analyse the language before preparing a large
build. Replace `xxx` with the actual language code:

```sh
python3 -c 'from uralicNLP import uralicApi; print(uralicApi.analyze("test", "xxx"))'
```

If this fails because the language is unknown or its resources cannot be
loaded, stop here and install/configure that language in UralicNLP first. This
repository does not create language analysers or dictionaries for UralicNLP.

## 1. Convert the keyboard layout

Put the kbdgen YAML file somewhere in the repository, for example
`kbdgen_layouts/xxx.yaml`. It must contain the phone layout at
`android.primary.layers.default`; a `shift` layer is used when present.

Convert one layout:

```sh
python3 kbdgen2futo.py kbdgen_layouts/xxx.yaml futo_layouts/xxx.yaml \
  --language xxx --name "Language name"
```

The result, `futo_layouts/xxx.yaml`, is a FUTO Keyboard layout and is also
compatible with HeliBoard. Review it before importing it, particularly the
language code, key rows, shifted keys, and long-press characters.

The converter handles the Android phone `primary` layout only. It preserves
the default and shift layers, converts common kbdgen special keys, and carries
over explicit `longpress` entries. It deliberately does not convert desktop,
tablet, iOS, spell-checker, or dead-key settings. A row with more than 12 keys
or an unsupported special key produces an error instead of a layout that may
not work.

To convert every YAML layout in a directory instead, provide an output
directory:

```sh
python3 kbdgen2futo.py kbdgen_layouts futo_layouts
```

Existing output is replaced by default. Add `--no-overwrite` to prevent that.

## 2. Prepare resources for predictive text

Predictive text needs a morphological lexicon. Create these directories, using
the same UralicNLP language code:

```text
lexc_stems/xxx/       # one or more .lexc files, including subdirectories
lang_corpora/xxx/     # optional but strongly recommended .txt and/or .xml files
```

Copy the language's `.lexc` files under `lexc_stems/xxx/`. Only files ending
in `.lexc` are read. The builder extracts lexical entries that contain a lemma
and a part of speech, then asks UralicNLP to generate usable surface forms.

Put representative, legally usable corpus text under `lang_corpora/xxx/`. The
model uses it to rank forms according to observed language use. It can build
with no corpus, but then its rankings depend mostly on the generated lexicon
and Finnish-derived prior, so quality will be substantially lower.

Two optional per-language controls can improve generation for unusual lexica:

- `morph_lookup/xxx_lookup.json` lists the tags to generate for each part of
  speech. Copy and adapt an existing file if needed.
- `black_list/xxx.json` lists parts of speech that should remain uninflected.

Neither file is required to begin: without a lookup, the builder uses
UralicNLP's `get_all_forms` fallback.

## 3. Build the vocabulary

Generate a JSON vocabulary from the `.lexc` stems:

```sh
python3 build_vocabulary.py lexc_stems vocabularies --language xxx
```

This creates `vocabularies/xxx.json`. If individual lemmas cannot be
generated, they are recorded in `vocabularies/xxx.errors.txt`; inspect that
file, fix the source or UralicNLP configuration if necessary, and rerun the
command. Other successfully generated words are still written.

## 4. Build the predictive-text model

Run the model builder after the vocabulary exists:

```sh
python3 predictive_model.py --language xxx
```

It reads `lang_corpora/xxx/`, `vocabularies/xxx.json`, and the included
`fin_frequencies.json`, then writes:

- `predictive_models/xxx.json` — the word-form model;
- `predictive_models/xxx.report.json` — counts, coverage, and failures; and
- `predictive_models/cache/xxx.json` plus resumable checkpoints.

Read the report before publishing a dictionary. In particular, check how many
target lemmas were recognised, how many mapped to Finnish frequencies, and any
reported API failures. The builder resumes expensive stages automatically. Use
`--restart` only when you intentionally want to discard its checkpoints, such
as after changing the source lexicon or corpus.

For a quiet non-interactive run, add `--no-progress`. Advanced smoothing and
form-generation controls are available through `--help`; the defaults are a
reasonable first build.

### Optional: inspect corpus morphology

This independent diagnostic counts morphological analyses in the corpus. It
does not feed the model builder directly, but can help assess coverage:

```sh
python3 count_morphology_frequencies.py lang_corpora morphology_frequencies \
  --language xxx
```

The result is `morphology_frequencies/xxx.json`.

## 5. Compile an installable dictionary

Convert the model to the AOSP dictionary format used by both FUTO Keyboard and
HeliBoard:

```sh
python3 predictive_model2dict.py --language xxx --overwrite
```

This produces two files in `futo_dictionaries/`:

- `xxx.combined` — readable source word list for inspection; and
- `main_xxx.dict` — the compiled binary dictionary to import.

The command uses the bundled `dictool/dicttool_aosp.jar` and requires Java.
If you keep the tool elsewhere, pass `--dicttool path/to/dicttool_aosp.jar`.

## What to import

Import `futo_layouts/xxx.yaml` as the keyboard layout and
`futo_dictionaries/main_xxx.dict` as the predictive-text dictionary in FUTO
Keyboard or HeliBoard. Keep the layout and dictionary language codes aligned.
The precise import screens vary by keyboard version, but both artifacts use
their shared FUTO/AOSP-compatible formats.

## Typical new-language checklist

- [ ] Confirm `xxx` is supported and working in UralicNLP.
- [ ] Convert and review `futo_layouts/xxx.yaml`.
- [ ] Add `.lexc` files under `lexc_stems/xxx/`.
- [ ] Add representative corpus text under `lang_corpora/xxx/`.
- [ ] Build `vocabularies/xxx.json` and review any errors file.
- [ ] Build and inspect `predictive_models/xxx.report.json`.
- [ ] Compile and import `futo_dictionaries/main_xxx.dict`.
