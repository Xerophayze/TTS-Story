import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from src.gemini_processor import GeminiProcessorError


class GeminiPrepResiliencyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.original_jobs_data_dir = app_module.JOBS_DATA_DIR
        self.original_jobs_archive_dir = app_module.JOBS_ARCHIVE_DIR
        self.original_jobs_db_path = app_module.JOBS_DB_PATH
        self.original_prep_jobs_data_dir = app_module.PREP_JOBS_DATA_DIR
        self.original_output_dir = app_module.OUTPUT_DIR

        app_module.JOBS_DATA_DIR = self.temp_path / "data" / "jobs"
        app_module.JOBS_ARCHIVE_DIR = app_module.JOBS_DATA_DIR / "archive"
        app_module.JOBS_DB_PATH = app_module.JOBS_DATA_DIR / "jobs.db"
        app_module.PREP_JOBS_DATA_DIR = app_module.JOBS_DATA_DIR / "prep"
        app_module.OUTPUT_DIR = self.temp_path / "static" / "audio"

        app_module.JOBS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        app_module.JOBS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        app_module.PREP_JOBS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        app_module.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        app_module.prep_job_threads.clear()
        app_module.prep_job_cancel_events.clear()
        app_module._init_jobs_db()
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        app_module.prep_job_threads.clear()
        app_module.prep_job_cancel_events.clear()
        app_module.JOBS_DATA_DIR = self.original_jobs_data_dir
        app_module.JOBS_ARCHIVE_DIR = self.original_jobs_archive_dir
        app_module.JOBS_DB_PATH = self.original_jobs_db_path
        app_module.PREP_JOBS_DATA_DIR = self.original_prep_jobs_data_dir
        app_module.OUTPUT_DIR = self.original_output_dir
        self.temp_dir.cleanup()

    def _config(self):
        config = copy.deepcopy(app_module.DEFAULT_CONFIG)
        config.update(
            {
                "llm_provider": "gemini",
                "gemini_api_key": "test-key",
                "gemini_model": "gemini-test",
                "gemini_prompt": "Preserve tags",
            }
        )
        return config

    def test_create_prep_job_persists_initial_state(self):
        sections = [
            {"title": "Chapter 1", "content": "First section", "source": "section"},
            {"title": "Chapter 2", "content": "Second section", "source": "section"},
        ]

        with patch.object(app_module, "load_config", return_value=self._config()), patch.object(
            app_module,
            "build_gemini_sections",
            return_value=sections,
        ), patch.object(app_module, "_launch_prep_job_worker") as launch_worker:
            response = self.client.post(
                "/api/gemini/prep-jobs",
                json={
                    "text": "Source manuscript",
                    "prefer_chapters": True,
                },
            )

        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        job = payload["job"]
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["processed_sections"], 0)
        self.assertEqual(job["total_sections"], 2)
        launch_worker.assert_called_once_with(job["job_id"])

        stored = app_module._load_prep_job(job["job_id"])
        self.assertIsNotNone(stored)
        self.assertEqual(stored["status"], "queued")
        self.assertEqual(stored["total_sections"], 2)
        self.assertEqual(Path(stored["text_path"]).read_text(encoding="utf-8"), "Source manuscript")

    def test_prep_worker_persists_partial_output_and_resumes(self):
        config = self._config()
        sections = [
            {"title": "Chapter 1", "content": "First section", "source": "section"},
            {"title": "Chapter 2", "content": "Second section", "source": "section"},
        ]
        job_entry = app_module._create_prep_job_entry(
            "Original text",
            prefer_chapters=True,
            custom_heading="",
            prompt_override="",
            config=config,
        )
        job_entry["sections"] = sections
        job_entry["total_sections"] = len(sections)
        app_module._persist_prep_job_state(job_entry)

        first_failure = GeminiProcessorError(
            "Gemini API error: 503 UNAVAILABLE",
            retryable=True,
            status_code=503,
            attempts=3,
            max_retries=3,
        )
        call_count = {"value": 0}

        def flaky_run(_prompt, _config, retry_callback=None):
            call_count["value"] += 1
            if call_count["value"] == 1:
                return "[narrator]First done[/narrator]"
            raise first_failure

        with patch.object(app_module, "load_config", return_value=config), patch.object(
            app_module,
            "_run_llm_prompt",
            side_effect=flaky_run,
        ):
            app_module._process_prep_job_worker(job_entry["job_id"])

        partial = app_module._load_prep_job(job_entry["job_id"])
        self.assertEqual(partial["status"], "failed")
        self.assertTrue(partial["retryable"])
        self.assertEqual(partial["processed_sections"], 1)
        self.assertIn("0", partial["completed_outputs"])
        self.assertEqual(partial["known_speakers"], ["narrator"])
        self.assertEqual(partial["failure_section_index"], 1)

        with patch.object(app_module, "load_config", return_value=config), patch.object(
            app_module,
            "_run_llm_prompt",
            return_value="[narrator]Second done[/narrator]",
        ):
            app_module._process_prep_job_worker(job_entry["job_id"])

        resumed = app_module._load_prep_job(job_entry["job_id"])
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["processed_sections"], 2)
        self.assertIn("First done", resumed["final_text"])
        self.assertIn("Second done", resumed["final_text"])
        self.assertEqual(resumed["known_speakers"], ["narrator"])

    def test_retryable_process_section_error_returns_503(self):
        retryable_error = GeminiProcessorError(
            "Gemini API error: 503 UNAVAILABLE",
            retryable=True,
            status_code=503,
            attempts=3,
            max_retries=3,
        )

        with patch.object(app_module, "load_config", return_value=self._config()), patch.object(
            app_module,
            "_run_llm_prompt",
            side_effect=retryable_error,
        ):
            response = self.client.post(
                "/api/gemini/process-section",
                json={"content": "Section text"},
            )

        self.assertEqual(response.status_code, 503)
        payload = response.get_json()
        self.assertFalse(payload["success"])
        self.assertTrue(payload["retryable"])
        self.assertEqual(payload["status_code"], 503)

    def test_restore_prep_jobs_marks_interrupted_work_as_resumable(self):
        config = self._config()
        job_entry = app_module._create_prep_job_entry(
            "Original text",
            prefer_chapters=True,
            custom_heading="",
            prompt_override="",
            config=config,
        )
        job_entry["status"] = "processing"
        app_module._persist_prep_job_state(job_entry)

        app_module._restore_prep_jobs_from_db()
        restored = app_module._load_prep_job(job_entry["job_id"])
        self.assertEqual(restored["status"], "failed")
        self.assertTrue(restored["retryable"])
        self.assertIn("Resume to continue", restored["last_error"])


if __name__ == "__main__":
    unittest.main()
