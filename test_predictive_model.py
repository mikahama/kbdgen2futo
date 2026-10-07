"""Synthetic tests for the predictive unigram model (no FST download needed)."""

import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import predictive_model as pm


class FakeApi:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_translation(self, lemma, language, trans_lang):
        return self.mapping.get(lemma, [])


def analysis(lemma, morph=()):
    return pm.Analysis("N", lemma, tuple(morph))


def statistics(surface_readings, lemma_counts=None, morph_counts=None):
    lemma_counts = Counter(lemma_counts or {})
    morph_counts = Counter(morph_counts or {})
    pos = Counter()
    for (pos_name, _lemma), count in lemma_counts.items():
        pos[pos_name] += count
    lemma_morph = Counter()
    for (pos_name, morph), count in morph_counts.items():
        # Fixtures use only one lemma unless stated otherwise.
        candidates = [lemma for known_pos, lemma in lemma_counts if known_pos == pos_name]
        if len(candidates) == 1:
            lemma_morph[pos_name, candidates[0], morph] = count
    return pm.TargetStatistics(pos, lemma_counts, morph_counts, lemma_morph, surface_readings, int(sum(pos.values())), 0, [])


def model(readings, finnish, mapping, lemma_counts=None, morph_counts=None, config=None):
    stats = statistics(readings, lemma_counts, morph_counts)
    return pm.UnigramModel(config or pm.ModelConfig(mu=10, kappa=2, rho=2, show_progress=False), stats, readings, finnish, pm.TranslationCache(None, "x", FakeApi(mapping)))


class PredictiveModelTests(unittest.TestCase):
    def test_one_to_one_and_frequency_order(self):
        readings = {"a": {analysis("a")}, "b": {analysis("b")}}
        result = model(readings, {"fi_a": 100, "fi_b": 10}, {"a": ["fi_a"], "b": ["fi_b"]})
        self.assertGreater(result.finnish_prior["N", "a"], result.finnish_prior["N", "b"])
        self.assertEqual(result.mapping_status["N", "a"], "finnish_frequency")

    def test_translation_mass_is_not_duplicated_and_multiple_finnish_sum(self):
        readings = {"a": {analysis("a")}, "b": {analysis("b")}, "c": {analysis("c")}}
        result = model(readings, {"shared": 100, "one": 60, "two": 60}, {"a": ["shared"], "b": ["shared"], "c": ["one", "two"]})
        self.assertAlmostEqual(result.finnish_prior["N", "a"], result.finnish_prior["N", "b"])
        self.assertGreater(result.finnish_prior["N", "c"], result.finnish_prior["N", "a"])

    def test_missing_mapping_and_missing_finnish_frequency_are_nonzero(self):
        readings = {"a": {analysis("a")}, "b": {analysis("b")}}
        result = model(readings, {}, {"a": [], "b": ["not_in_frequency_file"]})
        self.assertGreater(result.finnish_prior["N", "a"], 0)
        self.assertGreater(result.finnish_prior["N", "b"], 0)
        self.assertEqual(result.mapping_status["N", "a"], "no_translation")
        self.assertEqual(result.mapping_status["N", "b"], "translation_without_frequency")

    def test_target_counts_can_overcome_finnish_prior_and_normalise(self):
        readings = {"a": {analysis("a")}, "b": {analysis("b")}}
        result = model(readings, {"fi_a": 1_000_000, "fi_b": 1}, {"a": ["fi_a"], "b": ["fi_b"]}, {("N", "a"): 1, ("N", "b"): 1_000}, config=pm.ModelConfig(mu=1, kappa=2, rho=2, show_progress=False))
        self.assertGreater(result.lemma_probability["N", "b"], result.lemma_probability["N", "a"])
        self.assertAlmostEqual(sum(result.lemma_probability["N", lemma] for lemma in ("a", "b")), 1.0)
        self.assertAlmostEqual(sum(result.pos_probability.values()), 1.0)

    def test_morphology_observation_and_unobserved_legal_backoff(self):
        nom, gen = ("Sg", "Nom"), ("Sg", "Gen")
        readings = {"a_nom": {analysis("a", nom)}, "a_gen": {analysis("a", gen)}}
        result = model(readings, {}, {"a": []}, {("N", "a"): 4}, {("N", nom): 4})
        observed, observed_flag = result.morphology_probability(analysis("a", nom))
        unseen, unseen_flag = result.morphology_probability(analysis("a", gen))
        self.assertTrue(observed_flag)
        self.assertFalse(unseen_flag)
        self.assertGreater(observed, unseen)
        self.assertGreater(unseen, 0)

    def test_restricted_paradigm_renormalises_and_ambiguity_sums(self):
        nom, gen = ("Sg", "Nom"), ("Sg", "Gen")
        a_nom, b_gen = analysis("a", nom), analysis("b", gen)
        readings = {"restricted": {a_nom}, "other": {b_gen}, "ambiguous": {a_nom, b_gen}}
        stats = pm.TargetStatistics(
            Counter({"N": 2}), Counter({("N", "a"): 1, ("N", "b"): 1}),
            Counter({("N", nom): 1, ("N", gen): 1}),
            Counter({("N", "a", nom): 1, ("N", "b", gen): 1}), readings, 2, 0, [],
        )
        result = pm.UnigramModel(pm.ModelConfig(mu=10, kappa=2, rho=2, show_progress=False), stats, readings, {}, pm.TranslationCache(None, "x", FakeApi({})))
        self.assertAlmostEqual(result.morphology_probability(a_nom)[0], 1.0)
        self.assertAlmostEqual(result.surface_probability("ambiguous"), result.analysis_probability(a_nom) + result.analysis_probability(b_gen))

    def test_fractional_corpus_ambiguity_contributes_one(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.txt"
            corpus.write_text("word", encoding="utf-8")
            stats = pm.collect_target_statistics("x", [corpus], lambda _word, _lang: [("a+N+Sg+Nom", 0), ("b+N+Sg+Gen", 0)], show_progress=False)
        self.assertAlmostEqual(sum(stats.pos.values()), 1.0)
        self.assertAlmostEqual(stats.lemma["N", "a"], 0.5)
        self.assertEqual(stats.ambiguous_tokens, 1)

    def test_fst_generated_legal_forms_are_added(self):
        readings = {"a_nom": {analysis("a", ("Sg", "Nom"))}}
        failures = pm.add_legal_generated_forms(
            "x", readings, 10, 10,
            lambda lemma, pos, language, limit_forms: [("a_gen:a+N+Sg+Gen", 0.0)], show_progress=False,
        )
        self.assertFalse(failures)
        self.assertIn("a_gen", readings)
        self.assertIn(analysis("a", ("Sg", "Gen")), readings["a_gen"])

    def test_all_probabilities_are_finite_positive(self):
        readings = {"a": {analysis("a")}, "b": {analysis("b", ("Sg", "Gen"))}}
        result = model(readings, {}, {"a": [], "b": []})
        for record in result.surface_records():
            self.assertTrue(math.isfinite(record["probability"]))
            self.assertTrue(math.isfinite(record["cost"]))
            self.assertGreater(record["probability"], 0)


if __name__ == "__main__":
    unittest.main()
