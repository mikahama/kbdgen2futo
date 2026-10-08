#!/usr/bin/env python3
"""Build a smoothed unigram predictive-text model for Uralic languages.

Finnish Common Crawl counts are deliberately used only as a *normalised*,
tempered prior over target-language lemmas.  They are never added to target
corpus token counts.  Target corpus observations and FST-licensed forms then
provide the POS and inflectional distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from count_morphology_frequencies import corpus_files, file_tokens
from tqdm import tqdm

try:
    from uralicNLP import uralicApi
except ImportError:  # pragma: no cover - depends on the user's installation
    sys.exit("uralicNLP is required; use the Python installation that provides it.")


def write_json_atomic(path: Path, data: Any) -> None:
    """Write a checkpoint without leaving a truncated file after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class CheckpointStore:
    """Small JSON checkpoints for independently expensive build stages."""

    VERSION = 1

    def __init__(self, root: Path | None, language: str, resume: bool = True):
        self.root, self.language, self.resume = root, language, resume

    def load(self, stage: str, fingerprint: Mapping[str, Any]) -> Any | None:
        if self.root is None or not self.resume:
            return None
        path = self.root / self.language / f"{stage}.json"
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (saved.get("version") == self.VERSION and saved.get("language") == self.language
                and saved.get("fingerprint") == fingerprint):
            return saved.get("data")
        return None

    def save(self, stage: str, fingerprint: Mapping[str, Any], data: Any) -> None:
        if self.root is not None:
            write_json_atomic(
                self.root / self.language / f"{stage}.json",
                {"version": self.VERSION, "language": self.language, "fingerprint": fingerprint, "data": data},
            )


@dataclass(frozen=True, order=True)
class Analysis:
    """One FST reading split into the factors used by the probability model."""

    pos: str
    lemma: str
    morphology: tuple[str, ...]


def analysis_to_json(item: Analysis) -> dict[str, Any]:
    return {"pos": item.pos, "lemma": item.lemma, "morphology": list(item.morphology)}


def analysis_from_json(data: Mapping[str, Any]) -> Analysis:
    return Analysis(str(data["pos"]), str(data["lemma"]), tuple(str(tag) for tag in data.get("morphology", [])))


def analyses_to_json(analyses_by_surface: Mapping[str, set[Analysis]]) -> dict[str, list[dict[str, Any]]]:
    return {surface: [analysis_to_json(item) for item in sorted(readings)] for surface, readings in sorted(analyses_by_surface.items())}


def analyses_from_json(data: Mapping[str, Any]) -> dict[str, set[Analysis]]:
    return {surface: {analysis_from_json(item) for item in readings} for surface, readings in data.items() if isinstance(surface, str) and isinstance(readings, list)}


def initial_lowercase_variant(surface: str, candidates: Iterable[str]) -> str:
    """Return lowercase only when both initial-case spellings occur."""
    if not surface or not surface[0].isupper():
        return surface
    lower = surface[0].lower() + surface[1:]
    return lower if lower != surface and lower in candidates else surface


def merge_initial_case_counts(counts: Counter[str]) -> Counter[str]:
    """Merge ``Word`` into ``word`` only when both are present."""
    merged: Counter[str] = Counter()
    known = set(counts)
    for surface, count in counts.items():
        merged[initial_lowercase_variant(surface, known)] += count
    return merged


def merge_initial_case_analyses(analyses_by_surface: Mapping[str, set[Analysis]]) -> dict[str, set[Analysis]]:
    """Merge title-case/lowercase duplicate output forms, retaining lowercase."""
    merged: dict[str, set[Analysis]] = defaultdict(set)
    known = set(analyses_by_surface)
    for surface, readings in analyses_by_surface.items():
        merged[initial_lowercase_variant(surface, known)].update(readings)
    return dict(merged)


@dataclass
class ModelConfig:
    """Tunable pseudo-count and prior parameters.

    ``tau`` flattens Finnish differences (smaller means flatter); ``mu`` is
    the finite Finnish-prior strength; ``kappa`` and ``rho`` control POS and
    lemma morphology backoff respectively; ``eta`` smooths POS frequencies.
    """

    tau: float = 0.7
    mu: float = 100.0
    kappa: float = 10.0
    rho: float = 5.0
    eta: float = 1.0
    epsilon: float = 1e-8
    probability_floor: float = 1e-300
    translation_ambiguity: str = "uniform"
    fst_ambiguity: str = "fractional"
    enumerate_legal_forms: bool = True
    legal_form_limit: int = 2000
    legal_enumeration_max_lemmas: int = 2000
    show_progress: bool = True

    def validate(self) -> None:
        if min(self.mu, self.kappa, self.rho, self.eta, self.epsilon, self.probability_floor) <= 0:
            raise ValueError("smoothing strengths and floors must be positive")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.legal_form_limit <= 0 or self.legal_enumeration_max_lemmas <= 0:
            raise ValueError("legal-form limits must be positive")
        if self.translation_ambiguity != "uniform" or self.fst_ambiguity != "fractional":
            raise ValueError("only uniform translation and fractional FST ambiguity are supported")


def parse_reading(reading: object) -> Analysis | None:
    """Parse the ``lemma+POS+...`` reading returned by UralicNLP."""
    if isinstance(reading, (tuple, list)):
        reading = reading[0] if reading else None
    if not isinstance(reading, str):
        return None
    parts = reading.split("+")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    # get_all_forms can return ``surface:lemma+POS+...``.
    lemma = parts[0].rsplit(":", 1)[-1]
    return Analysis(parts[1], lemma, tuple(parts[2:]))


def _normalise(weights: Mapping[str, float], floor: float) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        size = len(weights)
        return {key: 1.0 / size for key in weights} if size else {}
    # Do not clamp individual normalised values: doing so would make their sum
    # exceed one.  Callers add epsilon to priors and apply the final numerical
    # floor only when taking a log or emitting a form probability.
    return {key: max(0.0, value) / total for key, value in weights.items()}


class TranslationCache:
    """Persistent cache for the expensive target-lemma -> Finnish-lemma lookup."""

    CACHE_VERSION = 2

    def __init__(self, path: Path | None, language: str, api: Any = uralicApi):
        self.path, self.language, self.api = path, language, api
        self.entries: dict[str, dict[str, Any]] = {}
        if path and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if (data.get("version") == self.CACHE_VERSION and data.get("language") == language
                        and isinstance(data.get("entries"), dict)):
                    self.entries = data["entries"]
            except (OSError, ValueError):
                pass

    def finnish_terms(self, lemma: str) -> tuple[list[str], str]:
        cached = self.entries.get(lemma)
        if cached is not None:
            return list(cached.get("finnish_terms", [])), str(cached.get("status", "no_translation"))
        translations: list[str] = []
        finnish_terms: set[str] = set()
        status = "no_translation"
        try:
            value = self.api.get_translation(lemma, self.language, "fin")
            translations = [item for item in value if isinstance(item, str)] if isinstance(value, (list, tuple)) else []
            if translations:
                status = "translation_without_frequency"
            for translation in translations:
                # fin_frequencies.json is already keyed by Finnish lemmas.
                # Do not run Finnish lemmatization here: it can alter a correct
                # dictionary lemma and would needlessly require a Finnish FST.
                finnish_terms.add(translation)
        except Exception as error:
            status = "api_failure"
            self.entries[lemma] = {"translations": translations, "finnish_terms": [], "status": status, "error": str(error)}
            self.save()
            return [], status
        self.entries[lemma] = {"translations": translations, "finnish_terms": sorted(finnish_terms), "status": status}
        # This is intentionally immediate: dictionary lookups are the longest
        # stage, and completed lemmas must survive an interrupted build.
        self.save()
        return sorted(finnish_terms), status

    def save(self) -> None:
        if self.path is None:
            return
        write_json_atomic(self.path, {"version": self.CACHE_VERSION, "language": self.language, "entries": self.entries})


@dataclass
class TargetStatistics:
    pos: Counter[str]
    lemma: Counter[tuple[str, str]]
    morphology: Counter[tuple[str, tuple[str, ...]]]
    lemma_morphology: Counter[tuple[str, str, tuple[str, ...]]]
    analyses_by_surface: dict[str, set[Analysis]]
    corpus_tokens: int
    ambiguous_tokens: int
    failures: list[str]


def statistics_to_json(stats: TargetStatistics) -> dict[str, Any]:
    return {
        "pos": [[pos, count] for pos, count in sorted(stats.pos.items())],
        "lemma": [[pos, lemma, count] for (pos, lemma), count in sorted(stats.lemma.items())],
        "morphology": [[pos, list(morph), count] for (pos, morph), count in sorted(stats.morphology.items())],
        "lemma_morphology": [[pos, lemma, list(morph), count] for (pos, lemma, morph), count in sorted(stats.lemma_morphology.items())],
        "analyses_by_surface": analyses_to_json(stats.analyses_by_surface),
        "corpus_tokens": stats.corpus_tokens,
        "ambiguous_tokens": stats.ambiguous_tokens,
        "failures": stats.failures,
    }


def statistics_from_json(data: Mapping[str, Any]) -> TargetStatistics:
    return TargetStatistics(
        Counter({str(pos): float(count) for pos, count in data["pos"]}),
        Counter({(str(pos), str(lemma)): float(count) for pos, lemma, count in data["lemma"]}),
        Counter({(str(pos), tuple(str(tag) for tag in morph)): float(count) for pos, morph, count in data["morphology"]}),
        Counter({(str(pos), str(lemma), tuple(str(tag) for tag in morph)): float(count) for pos, lemma, morph, count in data["lemma_morphology"]}),
        analyses_from_json(data["analyses_by_surface"]), int(data["corpus_tokens"]), int(data["ambiguous_tokens"]), list(data.get("failures", [])),
    )


def collect_target_statistics(
    language: str,
    files: Iterable[Path],
    analyze: Callable[[str, str], Iterable[object]] = uralicApi.analyze,
    show_progress: bool = True,
) -> TargetStatistics:
    """Collect fractional FST corpus counts: each token contributes mass one."""
    surface_counts: Counter[str] = Counter()
    failures: list[str] = []
    file_list = list(files)
    for path in tqdm(file_list, desc=f"Reading {language} corpus", unit="file", dynamic_ncols=True, disable=not show_progress):
        try:
            surface_counts.update(file_tokens(path))
        except Exception as error:
            failures.append(f"{path}: {error}")
    surface_counts = merge_initial_case_counts(surface_counts)
    pos: Counter[str] = Counter()
    lemma: Counter[tuple[str, str]] = Counter()
    morphology: Counter[tuple[str, tuple[str, ...]]] = Counter()
    lemma_morphology: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    analyses_by_surface: dict[str, set[Analysis]] = {}
    ambiguous_tokens = 0
    for surface, count in tqdm(surface_counts.items(), total=len(surface_counts), desc=f"Analysing {language} corpus", unit="type", dynamic_ncols=True, disable=not show_progress):
        try:
            analyses = {parsed for reading in analyze(surface, language) if (parsed := parse_reading(reading))}
        except Exception as error:
            failures.append(f"{surface}: {error}")
            continue
        if not analyses:
            continue
        analyses_by_surface[surface] = analyses
        if len(analyses) > 1:
            ambiguous_tokens += count
        share = count / len(analyses)
        for item in analyses:
            pos[item.pos] += share
            lemma[item.pos, item.lemma] += share
            morphology[item.pos, item.morphology] += share
            lemma_morphology[item.pos, item.lemma, item.morphology] += share
    return TargetStatistics(pos, lemma, morphology, lemma_morphology, analyses_by_surface, sum(surface_counts.values()), ambiguous_tokens, failures)


def load_vocabulary_surfaces(path: Path) -> list[str]:
    """Load the existing vocabulary's surface-form keys, tolerating empty files."""
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, ValueError, UnicodeError):
        return []
    return sorted(key for key in data if isinstance(key, str)) if isinstance(data, dict) else []


def analyse_surfaces(
    language: str, surfaces: Iterable[str], analyze: Callable[[str, str], Iterable[object]] = uralicApi.analyze,
    show_progress: bool = True,
) -> tuple[dict[str, set[Analysis]], list[str]]:
    results: dict[str, set[Analysis]] = {}
    errors: list[str] = []
    raw_surfaces = set(surfaces)
    unique_surfaces = sorted({initial_lowercase_variant(surface, raw_surfaces) for surface in raw_surfaces})
    for surface in tqdm(unique_surfaces, desc=f"Analysing {language} vocabulary", unit="form", dynamic_ncols=True, disable=not show_progress):
        try:
            analyses = {parsed for reading in analyze(surface, language) if (parsed := parse_reading(reading))}
        except Exception as error:
            errors.append(f"{surface}: {error}")
            continue
        if analyses:
            results[surface] = analyses
    return results, errors


def add_legal_generated_forms(
    language: str,
    analyses_by_surface: dict[str, set[Analysis]],
    limit: int,
    max_lemmas: int,
    get_all_forms: Callable[..., Iterable[object]] = uralicApi.get_all_forms,
    show_progress: bool = True,
) -> list[str]:
    """Add FST-generated legal forms for already known ``(lemma, POS)`` pairs.

    This is intentionally bounded.  Some transducers have very large or
    cyclic paradigms, so failing to enumerate one lemma leaves the analysed
    vocabulary/corpus forms in place rather than failing a whole language.
    """
    errors: list[str] = []
    pairs = sorted({(item.lemma, item.pos) for values in analyses_by_surface.values() for item in values})
    if len(pairs) > max_lemmas:
        errors.append(f"legal-form enumeration limited to {max_lemmas} of {len(pairs)} lemma/POS pairs")
        pairs = pairs[:max_lemmas]
    for lemma, pos in tqdm(pairs, desc=f"Generating {language} legal forms", unit="lemma", dynamic_ncols=True, disable=not show_progress):
        try:
            forms = get_all_forms(lemma, pos, language, limit_forms=limit)
        except Exception as error:
            errors.append(f"{lemma}+{pos}: {error}")
            continue
        for form in forms:
            reading = form[0] if isinstance(form, (tuple, list)) and form else form
            parsed = parse_reading(reading)
            if not parsed or not isinstance(reading, str):
                continue
            # UralicNLP's get_all_forms returns surface:lemma+POS+tags.
            prefix = reading.split("+", 1)[0]
            if ":" not in prefix:
                continue
            surface = prefix.split(":", 1)[0]
            if surface:
                analyses_by_surface.setdefault(surface, set()).add(parsed)
    merged = merge_initial_case_analyses(analyses_by_surface)
    analyses_by_surface.clear()
    analyses_by_surface.update(merged)
    return errors


class UnigramModel:
    """Normalised POS × lemma × morphology model with FST ambiguity summation."""

    def __init__(self, config: ModelConfig, stats: TargetStatistics, analyses_by_surface: Mapping[str, set[Analysis]], finnish_counts: Mapping[str, int], translations: TranslationCache):
        config.validate()
        self.config, self.stats, self.analyses_by_surface = config, stats, dict(analyses_by_surface)
        self.analyses = sorted({item for readings in analyses_by_surface.values() for item in readings})
        self.lemmas_by_pos: dict[str, set[str]] = defaultdict(set)
        self.legal_morphology: dict[tuple[str, str], set[tuple[str, ...]]] = defaultdict(set)
        for item in self.analyses:
            self.lemmas_by_pos[item.pos].add(item.lemma)
            self.legal_morphology[item.pos, item.lemma].add(item.morphology)
        for pos, lemma in stats.lemma:
            self.lemmas_by_pos[pos].add(lemma)
        for pos, lemma, morph in stats.lemma_morphology:
            self.legal_morphology[pos, lemma].add(morph)
        self._build(finnish_counts, translations)

    def _build(self, finnish_counts: Mapping[str, int], translations: TranslationCache) -> None:
        cfg = self.config
        # Build reverse links so one Finnish lemma's mass is divided over all
        # target lemmas it reaches, rather than copied to every translation.
        finnish_targets: dict[str, set[tuple[str, str]]] = defaultdict(set)
        mapping_status: dict[tuple[str, str], str] = {}
        target_pairs = sorted((pos, lemma) for pos, lemmas in self.lemmas_by_pos.items() for lemma in lemmas)
        for pos, lemma in tqdm(target_pairs, desc="Mapping target lemmas to Finnish", unit="lemma", dynamic_ncols=True, disable=not cfg.show_progress):
            finnish, status = translations.finnish_terms(lemma)
            mapping_status[pos, lemma] = status
            for fin in finnish:
                finnish_targets[fin].add((pos, lemma))
        raw: Counter[tuple[str, str]] = Counter()
        for fin, targets in finnish_targets.items():
            count = finnish_counts.get(fin, 0)
            if count:
                for target in targets:
                    raw[target] += count / len(targets)
                    mapping_status[target] = "finnish_frequency"
        self.mapping_status = mapping_status
        self.finnish_prior: dict[tuple[str, str], float] = {}
        for pos, lemmas in self.lemmas_by_pos.items():
            values = {lemma: (raw[pos, lemma] + cfg.epsilon) ** cfg.tau for lemma in lemmas}
            self.finnish_prior.update({(pos, lemma): value for lemma, value in _normalise(values, cfg.probability_floor).items()})

        pos_keys = sorted(self.lemmas_by_pos) or sorted(self.stats.pos)
        total_pos = sum(self.stats.pos.values())
        pos_prior = 1.0 / len(pos_keys) if pos_keys else 0.0
        self.pos_probability = {
            pos: (self.stats.pos[pos] + cfg.eta * pos_prior) / (total_pos + cfg.eta)
            for pos in pos_keys
        }
        self.lemma_probability: dict[tuple[str, str], float] = {}
        for pos, lemmas in self.lemmas_by_pos.items():
            denominator = self.stats.pos[pos] + cfg.mu
            for lemma in lemmas:
                self.lemma_probability[pos, lemma] = (self.stats.lemma[pos, lemma] + cfg.mu * self.finnish_prior[pos, lemma]) / denominator

        self.pooled_morphology: dict[tuple[str, tuple[str, ...]], float] = {}
        for pos, lemmas in self.lemmas_by_pos.items():
            support = set(morph for known_pos, morph in self.stats.morphology if known_pos == pos)
            support.update(morph for lemma in lemmas for morph in self.legal_morphology[pos, lemma])
            if not support:
                support.add(())
            uniform = 1.0 / len(support)
            denom = self.stats.pos[pos] + cfg.kappa
            for morph in support:
                self.pooled_morphology[pos, morph] = (self.stats.morphology[pos, morph] + cfg.kappa * uniform) / denom

    def morphology_probability(self, item: Analysis) -> tuple[float, bool]:
        legal = self.legal_morphology[item.pos, item.lemma] or {item.morphology}
        base = {morph: self.pooled_morphology.get((item.pos, morph), 0.0) for morph in legal}
        normalised_base = _normalise(base, self.config.probability_floor)
        count = self.stats.lemma[item.pos, item.lemma]
        observed = self.stats.lemma_morphology[item.pos, item.lemma, item.morphology]
        denominator = count + self.config.rho
        probability = (observed + self.config.rho * normalised_base.get(item.morphology, self.config.probability_floor)) / denominator
        return max(self.config.probability_floor, probability), observed > 0

    def analysis_probability(self, item: Analysis) -> float:
        morph, _ = self.morphology_probability(item)
        return max(self.config.probability_floor, self.pos_probability.get(item.pos, self.config.probability_floor) * self.lemma_probability.get((item.pos, item.lemma), self.config.probability_floor) * morph)

    def surface_probability(self, surface: str) -> float:
        return sum(self.analysis_probability(item) for item in self.analyses_by_surface.get(surface, ()))

    def surface_records(self) -> list[dict[str, Any]]:
        records = []
        for surface in sorted(self.analyses_by_surface):
            readings = sorted(self.analyses_by_surface[surface])
            probability = max(self.config.probability_floor, sum(self.analysis_probability(item) for item in readings))
            records.append({
                "surface": surface, "probability": probability, "cost": -math.log(probability),
                "ambiguity": len(readings),
                "analyses": [{"lemma": item.lemma, "pos": item.pos, "morphology": list(item.morphology), "probability": self.analysis_probability(item), "finnish_mapping": self.mapping_status.get((item.pos, item.lemma), "no_translation"), "lemma_attested": self.stats.lemma[item.pos, item.lemma] > 0, "morphology_attested": self.morphology_probability(item)[1]} for item in readings],
            })
        return records

    def report(self) -> dict[str, Any]:
        lemmas = sum(len(value) for value in self.lemmas_by_pos.values())
        statuses = Counter(self.mapping_status.values())
        records = self.surface_records()
        attested = sum(self.stats.lemma[pos, lemma] > 0 for pos, values in self.lemmas_by_pos.items() for lemma in values)
        return {"corpus_tokens": self.stats.corpus_tokens, "distinct_target_lemmas": lemmas, "target_lemmas_attested": attested, "target_lemmas_unattested": lemmas - attested, "mapped_to_finnish": statuses["finnish_frequency"], "mapped_percent": 100 * statuses["finnish_frequency"] / lemmas if lemmas else 0.0, "translations_without_finnish_frequency": statuses["translation_without_frequency"], "no_translation": statuses["no_translation"], "api_failures": statuses["api_failure"], "generated_surface_forms": len(records), "forms_using_morphology_backoff": sum(not a["morphology_attested"] for r in records for a in r["analyses"]), "ambiguous_corpus_tokens": self.stats.ambiguous_tokens, "pos_probability_mass": sum(self.pos_probability.values()), "surface_probability_mass": sum(record["probability"] for record in records), "failures": self.stats.failures}


def load_finnish_frequencies(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a lemma-to-count mapping")
    return {lemma: count for lemma, count in data.items() if isinstance(lemma, str) and isinstance(count, (int, float)) and count > 0}


def file_fingerprint(files: Iterable[Path]) -> list[list[Any]]:
    """Cheap invalidation key for source-dependent checkpoints."""
    result = []
    for path in files:
        try:
            stat = path.stat()
            result.append([str(path), stat.st_size, stat.st_mtime_ns])
        except OSError:
            result.append([str(path), None, None])
    return result


def build_language_model(
    language: str,
    corpus_root: Path,
    vocabulary_root: Path,
    finnish_path: Path,
    cache_path: Path,
    config: ModelConfig,
    checkpoint_root: Path | None = None,
    resume: bool = True,
) -> tuple[UnigramModel, list[str]]:
    checkpoints = CheckpointStore(checkpoint_root, language, resume)
    files = corpus_files(corpus_root / language)
    corpus_fingerprint = {"files": file_fingerprint(files)}
    saved_stats = checkpoints.load("corpus_statistics", corpus_fingerprint)
    if saved_stats is not None:
        print(f"{language}: resumed corpus statistics checkpoint", file=sys.stderr)
        stats = statistics_from_json(saved_stats)
    else:
        stats = collect_target_statistics(language, files, show_progress=config.show_progress)
        checkpoints.save("corpus_statistics", corpus_fingerprint, statistics_to_json(stats))

    vocabulary_path = vocabulary_root / f"{language}.json"
    vocabulary_fingerprint = {"files": file_fingerprint([vocabulary_path])}
    combined_fingerprint = {
        "corpus": corpus_fingerprint,
        "vocabulary": vocabulary_fingerprint,
        "enumerate_legal_forms": config.enumerate_legal_forms,
        "legal_form_limit": config.legal_form_limit,
        "legal_enumeration_max_lemmas": config.legal_enumeration_max_lemmas,
    }
    saved_combined = checkpoints.load("candidate_analyses", combined_fingerprint)
    if saved_combined is not None:
        print(f"{language}: resumed candidate analyses checkpoint", file=sys.stderr)
        combined = analyses_from_json(saved_combined["analyses"])
        errors = list(saved_combined.get("errors", []))
    else:
        saved_vocabulary = checkpoints.load("vocabulary_analyses", vocabulary_fingerprint)
        if saved_vocabulary is not None:
            print(f"{language}: resumed vocabulary analyses checkpoint", file=sys.stderr)
            vocabulary_analyses = analyses_from_json(saved_vocabulary["analyses"])
            errors = list(saved_vocabulary.get("errors", []))
        else:
            vocabulary_surfaces = load_vocabulary_surfaces(vocabulary_path)
            vocabulary_analyses, errors = analyse_surfaces(language, vocabulary_surfaces, show_progress=config.show_progress)
            checkpoints.save("vocabulary_analyses", vocabulary_fingerprint, {"analyses": analyses_to_json(vocabulary_analyses), "errors": errors})
        combined_source: dict[str, set[Analysis]] = defaultdict(set)
        for source in (stats.analyses_by_surface, vocabulary_analyses):
            for surface, readings in source.items():
                combined_source[surface].update(readings)
        combined = merge_initial_case_analyses(combined_source)
        if config.enumerate_legal_forms:
            errors.extend(add_legal_generated_forms(language, combined, config.legal_form_limit, config.legal_enumeration_max_lemmas, show_progress=config.show_progress))
        checkpoints.save("candidate_analyses", combined_fingerprint, {"analyses": analyses_to_json(combined), "errors": errors})
    cache = TranslationCache(cache_path, language)
    model = UnigramModel(config, stats, combined, load_finnish_frequencies(finnish_path), cache)
    return model, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--corpora", type=Path, default=Path("lang_corpora"))
    parser.add_argument("--vocabularies", type=Path, default=Path("vocabularies"))
    parser.add_argument("--finnish-frequencies", type=Path, default=Path("fin_frequencies.json"))
    parser.add_argument("--output", type=Path, default=Path("predictive_models"))
    parser.add_argument("--cache", type=Path, default=Path("predictive_models/cache"))
    parser.add_argument("--checkpoints", type=Path, help="directory for resumable build checkpoints (default: <output>/checkpoints)")
    parser.add_argument("--restart", action="store_true", help="ignore existing checkpoints and rebuild all stages")
    parser.add_argument("--no-progress", action="store_true", help="disable tqdm progress bars")
    for name in ("tau", "mu", "kappa", "rho", "eta", "epsilon", "probability_floor", "legal_form_limit", "legal_enumeration_max_lemmas"):
        parser.add_argument("--" + name.replace("_", "-"), type=int if name.startswith("legal_") else float)
    args = parser.parse_args()
    values = {key: value for key, value in vars(args).items() if key in ModelConfig.__dataclass_fields__ and value is not None}
    values["show_progress"] = not args.no_progress
    config = ModelConfig(**values)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.checkpoints or args.output / "checkpoints"
    model, errors = build_language_model(args.language, args.corpora, args.vocabularies, args.finnish_frequencies, args.cache / f"{args.language}.json", config, checkpoint_root, resume=not args.restart)
    write_json_atomic(args.output / f"{args.language}.json", {"language": args.language, "config": asdict(config), "forms": model.surface_records()})
    write_json_atomic(args.output / f"{args.language}.report.json", model.report() | {"vocabulary_analysis_failures": errors})
    print(json.dumps(model.report(), ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
