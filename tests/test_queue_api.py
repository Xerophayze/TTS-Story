"""Job Queue screen API: additive title / position / output fields on /api/queue and job details."""
import json
import queue

import pytest

import app as app_module


def _job(status="queued", created_at="2026-07-15T19:00:00", **extra):
    entry = {
        "status": status,
        "created_at": created_at,
        "text_preview": "[narrator]Once upon a time.[/narrator]",
        "merge_options": {"output_format": "mp3", "output_bitrate_kbps": 128, "acx_compliance": False},
        "processed_chunks": 0,
        "total_chunks": 7,
    }
    entry.update(extra)
    return entry


@pytest.fixture
def queue_state(monkeypatch):
    state = {}
    pending = queue.Queue()
    monkeypatch.setattr(app_module, "jobs", state)
    monkeypatch.setattr(app_module, "job_queue", pending)
    monkeypatch.setattr(app_module, "current_job_ids", set())
    monkeypatch.setattr(app_module, "current_job_id", None)
    return state, pending


def _get_queue():
    response = app_module.app.test_client().get("/api/queue")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    return {job["job_id"]: job for job in payload["jobs"]}


@pytest.mark.parametrize("status,post_total,post_done,should_queue", [
    ("paused", 0, 0, True),
    ("paused", 2, 1, True),
    ("paused", 2, 2, False),
    ("completed", 0, 0, False),
])
def test_resume_after_final_chunk_preserves_pending_exports_and_never_persists_under_lock(
        queue_state, monkeypatch, status, post_total, post_done, should_queue):
    state, pending = queue_state
    state["last-chunk"] = _job(status=status, total_chunks=1, processed_chunks=1,
                              post_process_total=post_total, post_process_done=post_done)
    monkeypatch.setattr(app_module, "start_worker_thread", lambda: None)
    monkeypatch.setattr(app_module, "pause_flags", {"last-chunk": True})
    monkeypatch.setattr(app_module, "_build_job_data_from_entry", lambda job_id, entry: {"job_id": job_id})
    persisted = []
    def persist(job_id, **kwargs):
        assert not app_module.queue_lock.locked(), "persistence must be outside queue_lock"
        persisted.append(job_id)
    monkeypatch.setattr(app_module, "_persist_job_state", persist)
    response = app_module.app.test_client().post('/api/jobs/last-chunk/resume')
    assert response.status_code == 200
    assert pending.qsize() == int(should_queue)
    assert state["last-chunk"]["status"] == ("queued" if should_queue else "completed")
    assert persisted == ["last-chunk"]
    if should_queue:
        assert state["last-chunk"]["resume_from_chunk_index"] == 1


def test_title_prefers_submitted_collection_title_and_keeps_text_title(queue_state):
    state, _ = queue_state
    state["a"] = _job(
        collection_title="  The   Hidden Path ",
        text_preview="[narrator]Chapter 1 - The hidden passage[/narrator]\n[narrator]Once upon a time.",
    )
    job = _get_queue()["a"]
    assert job["title"] == "The Hidden Path"
    assert job["text_title"] == "Chapter 1 - The hidden passage"


def test_title_falls_back_to_first_sentence_without_tags_or_directions(queue_state):
    state, _ = queue_state
    state["plain"] = _job(
        text_preview="[direction]Read slowly.[/direction]\n***\n[narrator]Once upon a time, in a village. "
                     "Alice lived there.[/narrator]",
    )
    state["cut"] = _job(text_preview="[narrator]Hello there friend[/narr")
    state["open-direction"] = _job(text_preview="Short opening line\n[direction]A direction that was cut off")
    state["empty"] = _job(text_preview="")
    jobs = _get_queue()
    assert jobs["plain"]["title"] == "Once upon a time, in a village."
    assert jobs["cut"]["title"] == "Hello there friend"
    assert jobs["open-direction"]["title"] == "Short opening line"
    assert jobs["empty"]["title"] == ""


@pytest.mark.parametrize("text, expected", [
    ("[narrator]Mr. Holmes looked up from the fire.[/narrator]", "Mr. Holmes looked up from the fire."),
    ("Dr. Watson arrived at dawn... The fog was thick.", "Dr. Watson arrived at dawn..."),
    ("It was 3 a.m. when the bell rang. Nobody moved.", "It was 3 a.m. when the bell rang."),
    ("Mrs. St. Clair went to St. Ives on Tuesday. It rained.", "Mrs. St. Clair went to St. Ives on Tuesday."),
    ("He lived in the U.S. for years. Then moved.", "He lived in the U.S. for years."),
    ('"Run away from here now!" she said. Then left.', '"Run away from here now!"'),
    ("Yes. I will go to the store tomorrow. And then home.", "Yes. I will go to the store tomorrow."),
    ("Chapter I. The Beginning", "Chapter I. The Beginning"),
    ("The price was 3.5 dollars. Cheap.", "The price was 3.5 dollars."),
    # Pause controls are not part of the title and "silent.***" still ends a sentence.
    ("[narrator]CHAPTER ONE******[/narrator]\n[narrator]It began.[/narrator]", "CHAPTER ONE"),
    ("[narrator]Chapter Three ***[/narrator]", "Chapter Three"),
    ("[narrator]Chapter 4.******[/narrator]", "Chapter 4."),
    ("[narrator]The room fell silent.*** Then the door opened slowly. More.", "The room fell silent."),
])
def test_fallback_title_does_not_stop_at_abbreviations_or_tiny_sentences(text, expected):
    assert app_module._queue_title_from_text(text) == expected


def test_long_titles_are_truncated_to_80_chars(queue_state):
    state, _ = queue_state
    state["long"] = _job(text_preview="word " * 60)
    title = _get_queue()["long"]["title"]
    assert len(title) <= 80
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")


def test_breeze_production_title_is_used_only_when_it_is_not_the_text_opening():
    text = "[narrator]Once upon a time there was a long story that keeps going[/narrator]"
    hints = app_module._submitted_queue_title_hints({"production_title": text[:40]}, text)
    assert hints["production_title"] == ""
    hints = app_module._submitted_queue_title_hints({"production_title": "Moon Book", "project_name": ""}, text)
    assert hints["production_title"] == "Moon Book"
    hints = app_module._submitted_queue_title_hints({"book_title": "Sherlock", "project_name": "Draft"}, text)
    assert hints["collection_title"] == "Sherlock"


def test_production_title_is_used_when_no_collection_title(queue_state):
    state, _ = queue_state
    state["breeze"] = _job(collection_title="", production_title="Moonlight Library")
    assert _get_queue()["breeze"]["title"] == "Moonlight Library"


def test_restored_jobs_read_collection_title_from_metadata_once(queue_state, tmp_path, monkeypatch):
    state, _ = queue_state
    job_dir = tmp_path / "restored"
    job_dir.mkdir()
    (job_dir / app_module.JOB_METADATA_FILENAME).write_text(
        json.dumps({"collection_title": "Restored Story"}), encoding="utf-8")
    state["restored"] = _job(status="completed", job_dir=str(job_dir))
    state["no-meta"] = _job(status="completed", job_dir=str(tmp_path / "missing"))

    calls = []
    original = app_module.load_job_metadata
    monkeypatch.setattr(app_module, "load_job_metadata", lambda path: calls.append(path) or original(path))

    first = _get_queue()
    assert first["restored"]["title"] == "Restored Story"
    assert first["no-meta"]["title"] == "Once upon a time."
    assert len(calls) == 2

    second = _get_queue()
    assert second["restored"]["title"] == "Restored Story"
    assert second["no-meta"]["title"] == "Once upon a time."
    assert len(calls) == 2, "metadata.json must only be read once per job"


def test_queue_position_follows_real_dispatch_order(queue_state):
    state, pending = queue_state
    state["first-created"] = _job(created_at="2026-07-15T18:00:00")
    state["resumed"] = _job(created_at="2026-07-15T17:00:00")
    state["paused"] = _job(status="paused")
    state["running"] = _job(status="processing")
    # Taken off the queue by the dispatcher, waiting for a cloud slot.
    state["dispatched"] = _job(created_at="2026-07-15T20:00:00", _dispatched=True)
    # Just submitted: the entry is visible but job_queue.put() has not happened yet.
    state["submitting"] = _job(created_at="2026-07-15T16:00:00")
    state["design"] = _job(job_type="qwen3_voice_design_preview")
    for job_id in ("design", "first-created", "paused", "resumed", "first-created"):
        pending.put({"job_id": job_id})

    jobs = _get_queue()
    assert "design" not in jobs
    assert jobs["dispatched"]["queue_position"] == 1
    assert jobs["first-created"]["queue_position"] == 2
    assert jobs["resumed"]["queue_position"] == 3
    assert jobs["submitting"]["queue_position"] == 4, "a job not yet in job_queue must not jump ahead"
    assert jobs["paused"]["queue_position"] is None
    assert jobs["running"]["queue_position"] is None


def test_dispatcher_flags_the_job_it_takes_and_processing_clears_it(queue_state, monkeypatch):
    state, pending = queue_state
    state["cloud"] = _job()
    real_run = app_module._run_queued_job
    seen = {}

    def stop_after_dispatch(job_data, cloud_slot=False):
        seen["flag_at_run"] = state["cloud"].get("_dispatched")
        seen["position"] = _get_queue()["cloud"]["queue_position"]
        raise SystemExit  # leave the dispatcher loop after one job

    monkeypatch.setattr(app_module, "_is_concurrent_cloud_job", lambda job_data: False)
    monkeypatch.setattr(app_module, "_run_queued_job", stop_after_dispatch)
    pending.put({"job_id": "cloud"})
    with pytest.raises(SystemExit):
        app_module.process_job_worker()
    assert seen == {"flag_at_run": True, "position": 1}

    # Starting the job clears the flag (a later resume must not inherit it).
    monkeypatch.setattr(app_module, "_persist_job_state", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "process_audio_job", lambda job_data: None)
    pending.put({"job_id": "cloud"})
    pending.get()  # _run_queued_job calls job_queue.task_done() for the dispatcher's get()
    real_run({"job_id": "cloud"})
    assert "_dispatched" not in state["cloud"]
    assert state["cloud"]["status"] == "processing"


def test_queued_jobs_expose_the_submit_time_chunk_estimate(queue_state):
    state, pending = queue_state
    state["waiting"] = _job(total_chunks=None)
    state["unknown"] = _job(total_chunks=None)
    pending.put({"job_id": "waiting", "total_chunks": 7})
    pending.put({"job_id": "unknown"})
    jobs = _get_queue()
    assert jobs["waiting"]["estimated_total_chunks"] == 7
    assert jobs["waiting"]["total_chunks"] is None, "the real total stays unset until processing starts"
    assert jobs["unknown"]["estimated_total_chunks"] is None


def test_dispatched_job_keeps_its_chunk_estimate_after_leaving_job_queue(queue_state):
    # A cloud job the dispatcher took off job_queue (waiting for a slot) has no payload
    # left to read the estimate from: the job entry's submit-time estimate is used.
    state, pending = queue_state
    state["next-up"] = _job(total_chunks=None, estimated_total_chunks=42, _dispatched=True)
    state["behind"] = _job(total_chunks=None, estimated_total_chunks=5)
    state["done"] = _job(status="completed", total_chunks=None, estimated_total_chunks=9)
    pending.put({"job_id": "behind", "total_chunks": 6})
    jobs = _get_queue()
    assert jobs["next-up"]["queue_position"] == 1
    assert jobs["next-up"]["estimated_total_chunks"] == 42
    assert jobs["behind"]["estimated_total_chunks"] == 6, "the queued payload stays the first source"
    assert jobs["done"]["estimated_total_chunks"] is None


def test_output_summary_comes_from_merge_options_with_snapshot_fallback(queue_state):
    state, _ = queue_state
    state["mp3"] = _job(merge_options={"output_format": "MP3", "output_bitrate_kbps": 192, "acx_compliance": True})
    state["snapshot"] = _job(merge_options={}, config_snapshot={"output_format": "wav", "output_bitrate_kbps": 0})
    state["broken"] = _job(merge_options="not-a-dict", config_snapshot=None)
    jobs = _get_queue()
    assert (jobs["mp3"]["output_format"], jobs["mp3"]["output_bitrate_kbps"], jobs["mp3"]["acx_compliance"]) == ("mp3", 192, True)
    assert (jobs["snapshot"]["output_format"], jobs["snapshot"]["output_bitrate_kbps"], jobs["snapshot"]["acx_compliance"]) == ("wav", None, False)
    assert (jobs["broken"]["output_format"], jobs["broken"]["output_bitrate_kbps"], jobs["broken"]["acx_compliance"]) == (None, None, False)


def test_existing_queue_fields_are_unchanged(queue_state):
    state, pending = queue_state
    state["a"] = _job()
    pending.put({"job_id": "a"})
    response = app_module.app.test_client().get("/api/queue").get_json()
    assert response["queue_size"] == 1
    job = response["jobs"][0]
    for key in ("job_id", "status", "text_preview", "total_chunks", "processed_chunks", "eta_seconds",
                "chapter_mode", "full_story_requested", "post_process_total", "error"):
        assert key in job


def test_job_details_include_output_summary_error_and_optional_text(queue_state, tmp_path, monkeypatch):
    state, _ = queue_state
    text_path = tmp_path / "job.txt"
    text_path.write_text("[narrator]Hello world.[/narrator]", encoding="utf-8")
    monkeypatch.setattr(app_module, "_load_job_text", lambda path: text_path.read_text(encoding="utf-8") if path else "")
    state["d"] = _job(status="failed", error="Engine crashed", speakers=["narrator"], text_path=str(text_path),
                      eta_seconds=12, chapter_count=3, collection_title="My Book",
                      merge_options={"output_format": "ogg", "output_bitrate_kbps": 160, "acx_compliance": False})
    client = app_module.app.test_client()

    full = client.get("/api/jobs/d/details").get_json()["job"]
    assert full["text"] == "[narrator]Hello world.[/narrator]"
    assert full["error"] == "Engine crashed"
    assert (full["output_format"], full["output_bitrate_kbps"], full["acx_compliance"]) == ("ogg", 160, False)
    assert full["title"] == "My Book"
    assert full["eta_seconds"] == 12
    assert full["chapter_count"] == 3
    assert full["speakers"] == ["narrator"]

    light = client.get("/api/jobs/d/details?include_text=0").get_json()["job"]
    assert light["text"] is None
    assert light["speakers"] == ["narrator"]
    assert light["output_format"] == "ogg"

    assert client.get("/api/jobs/missing/details").status_code == 404


def test_light_details_of_a_restored_job_read_the_manuscript_once_outside_the_lock(queue_state, tmp_path, monkeypatch):
    # Jobs restored from the DB carry no "speakers": the first light refresh extracts them
    # (without holding queue_lock) and stores them so later polls skip the manuscript.
    state, _ = queue_state
    text_path = tmp_path / "restored.txt"
    text_path.write_text("[narrator]Hi.[/narrator] [alice]Hello.[/alice]", encoding="utf-8")
    reads = []

    def load_text(path):
        reads.append(app_module.queue_lock.locked())
        return text_path.read_text(encoding="utf-8") if path else ""

    monkeypatch.setattr(app_module, "_load_job_text", load_text)
    state["restored"] = _job(status="processing", text_path=str(text_path))
    client = app_module.app.test_client()

    first = client.get("/api/jobs/restored/details?include_text=0").get_json()["job"]
    second = client.get("/api/jobs/restored/details?include_text=0").get_json()["job"]
    assert first["speakers"] == second["speakers"] == ["narrator", "alice"]
    assert first["text"] is None and second["text"] is None
    assert reads == [False], "one manuscript read, never while holding queue_lock"
    assert state["restored"]["speakers"] == ["narrator", "alice"]

    full = client.get("/api/jobs/restored/details").get_json()["job"]
    assert full["text"].startswith("[narrator]Hi.")
    assert reads == [False, False]


@pytest.fixture
def submit_without_worker(queue_state, monkeypatch, tmp_path):
    """/api/generate with the worker, persistence and engine side effects stubbed out."""
    monkeypatch.setattr(app_module, "start_worker_thread", lambda: None)
    monkeypatch.setattr(app_module, "_persist_job_state", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "_archive_old_jobs", lambda: None)
    monkeypatch.setattr(app_module, "_consume_engine_first_run_notice", lambda *a: False)
    monkeypatch.setattr(app_module, "_write_job_text", lambda *a: str(tmp_path / "input.txt"))
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path / "audio")
    monkeypatch.setattr(app_module, "load_config", lambda: {**app_module.DEFAULT_CONFIG, "tts_engine": "kokoro"})
    client = app_module.app.test_client()

    def submit(**body):
        assignments = {"narrator": {"voice": "af_heart", "lang_code": "a"}}
        response = client.post("/api/generate", json={"tts_engine": "kokoro", "voice_assignments": assignments, **body})
        assert response.status_code == 200, response.get_json()
        return response.get_json()["job_id"]

    return submit


def test_generate_keeps_the_submitted_project_title_for_the_queue(submit_without_worker):
    text = "[narrator]Chapter 1 - The hidden passage[/narrator]\n[narrator]Once upon a time.[/narrator]"
    titled = submit_without_worker(text=text, project_name="  The   Hidden Path ")
    produced = submit_without_worker(text=text, production_title="Moonlight Library")
    untitled = submit_without_worker(text=text)

    # The hints live on the job entry itself (not only in metadata.json).
    assert app_module.jobs[titled]["collection_title"] == "The Hidden Path"
    assert app_module.jobs[produced]["production_title"] == "Moonlight Library"
    assert app_module.jobs[titled]["estimated_total_chunks"] >= 1

    jobs = _get_queue()
    assert jobs[titled]["title"] == "The Hidden Path"
    assert jobs[titled]["text_title"] == "Chapter 1 - The hidden passage"
    assert jobs[produced]["title"] == "Moonlight Library"
    assert jobs[untitled]["title"] == "Chapter 1 - The hidden passage"
    assert [jobs[j]["queue_position"] for j in (titled, produced, untitled)] == [1, 2, 3]
    assert jobs[titled]["estimated_total_chunks"] >= 1


def test_renaming_a_production_in_the_library_refreshes_its_queue_title(queue_state, tmp_path, monkeypatch):
    state, _ = queue_state
    monkeypatch.setattr(app_module, "OUTPUT_DIR", tmp_path)
    for job_id in ("submitted", "restored"):
        (tmp_path / job_id).mkdir()
        (tmp_path / job_id / app_module.JOB_METADATA_FILENAME).write_text(
            json.dumps({"collection_title": "Old Name"}), encoding="utf-8")
    state["submitted"] = _job(status="completed", collection_title="Old Name", job_dir=str(tmp_path / "submitted"))
    state["restored"] = _job(status="completed", job_dir=str(tmp_path / "restored"))
    client = app_module.app.test_client()

    before = _get_queue()
    assert before["submitted"]["title"] == before["restored"]["title"] == "Old Name"

    for job_id in ("submitted", "restored"):
        response = client.put(f"/api/library/{job_id}/title", json={"title": "New Name"})
        assert response.get_json()["success"] is True

    after = _get_queue()
    assert after["submitted"]["title"] == "New Name"
    assert after["restored"]["title"] == "New Name"
    details = client.get("/api/jobs/restored/details?include_text=0").get_json()["job"]
    assert details["title"] == "New Name"
