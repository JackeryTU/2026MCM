"""Exercise stat-only binding and drift rejection without content digests."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import latex_paper_nohash as adapter

HERE = Path(__file__).resolve().parent


class MetadataBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="binding-test-", dir=HERE)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        shutil.copytree(HERE / "export_smoke", self.project)
        shutil.copytree(HERE / "smoke_inputs", self.root / "smoke_inputs")
        self.main = self.project / "main.tex"
        self.figure = self.project / "figs/result_q1_optimal_schedule.png"

    def tearDown(self):
        self.temp.cleanup()

    def test_unchanged_bound_copy(self):
        self.assertEqual(adapter._resource_binding_issues(self.project), [])

    def test_changed_copy_mtime_is_rejected(self):
        stat = self.figure.stat()
        os.utime(self.figure, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        self.assertTrue(any("漂移" in s for s in adapter._resource_binding_issues(self.project)))

    def test_missing_source_is_rejected(self):
        authority = self.root / "smoke_inputs/result_q1_optimal_schedule.png"
        authority.rename(authority.with_suffix(".held"))
        self.assertTrue(any("缺失" in s for s in adapter._resource_binding_issues(self.project)))

    def test_unbound_resource_is_rejected(self):
        shutil.copy2(self.figure, self.project / "unbound.png")
        self.assertTrue(any("未绑定" in s for s in adapter._resource_binding_issues(self.project)))

    def test_source_snapshot_tracks_mtime(self):
        before = adapter.source_bundle_metadata(self.main)
        stat = self.main.stat()
        os.utime(self.main, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
        self.assertNotEqual(before, adapter.source_bundle_metadata(self.main))

    def test_original_published_pdf_binding(self):
        main = HERE / "export_smoke/main.tex"
        pdf = HERE / "export_smoke.pdf"
        _, issues = adapter._verify_pdf_binding(main, pdf, adapter.source_bundle_metadata(main))
        self.assertEqual(issues, [])

    def test_changed_source_snapshot_rejects_pdf(self):
        main = HERE / "export_smoke/main.tex"
        before = adapter.source_bundle_metadata(main)
        before["main.tex"]["bytes"] += 1
        _, issues = adapter._verify_pdf_binding(main, HERE / "export_smoke.pdf", before)
        self.assertTrue(any("源码元数据" in s for s in issues))

    def test_latexmk_version_ignores_codepage_banner(self):
        with patch.object(adapter.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = ("Initial Win CP for console\nLatexmk, John Collins, 15 June 2025. Version 4.87\n", None)
            popen.return_value.returncode = 0
            self.assertEqual(adapter._tool_version("latexmk.exe"), "Latexmk, John Collins, 15 June 2025. Version 4.87")


if __name__ == "__main__":
    unittest.main(verbosity=2)
