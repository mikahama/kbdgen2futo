#!/usr/bin/env python3
"""Tests for predictive-model to AOSP combined dictionary conversion."""
import json
import tempfile
import unittest
from pathlib import Path

import predictive_model2dict as converter


class ConverterTests(unittest.TestCase):
    def write_model(self, directory: Path, forms, language="sms") -> Path:
        path = directory / "model.json"
        path.write_text(json.dumps({"language": language, "config": {}, "forms": forms}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_writes_aosp_header_and_logarithmic_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_model(root, [{"surface": "most", "probability": 1.0}, {"surface": "less", "probability": 1 / converter.AOSP_LOG_BASE}, {"surface": "rare", "probability": 1e-100}])
            output = root / "sms.combined"
            result = converter.convert_model(source, output, date=123, description="Skolt Sami")
            self.assertEqual(result.forms_written, 3)
            self.assertEqual(output.read_text(encoding="utf-8").splitlines(), ["dictionary=main:sms,locale=sms,description=Skolt Sami,date=123,version=1", " word=most,f=255", " word=less,f=254", " word=rare,f=15"])

    def test_skips_invalid_forms_and_reader_handles_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_model(root, [{"surface": "valid", "probability": 0.5}, {"surface": "bad,word", "probability": 0.4}, {"surface": "zero", "probability": 0}], language="myv")
            with source.open(encoding="utf-8") as handle:
                self.assertEqual(len(list(converter.JsonArrayReader(handle, chunk_size=7).forms())), 3)
            result = converter.convert_model(source, root / "myv.combined", date=1)
            self.assertEqual((result.forms_written, result.forms_skipped), (1, 2))

    def test_model_language_and_frequency_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.write_model(root, [{"surface": "a", "probability": 0.5}])
            self.assertEqual(converter.model_language(source), "sms")
            with self.assertRaises(converter.ConversionError):
                converter.convert_model(source, root / "x.combined", minimum=256)


if __name__ == "__main__":
    unittest.main()
