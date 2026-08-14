from __future__ import annotations

from pathlib import Path

import app as app_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_project_library_is_shared_and_imports_legacy_browser_projects(tmp_path, monkeypatch):
    project_file = tmp_path / "projects.json"
    monkeypatch.setattr(app_module, "PROJECTS_FILE", project_file)
    client = app_module.app.test_client()

    saved = client.post(
        "/api/projects",
        json={
            "project": {
                "id": 100,
                "name": "Shared Story",
                "saved_at": "2026-08-10T10:00:00",
                "text": "Original text",
            }
        },
    )
    assert saved.status_code == 200
    assert saved.get_json()["project"]["id"] == "100"

    imported = client.post(
        "/api/projects/import",
        json={
            "projects": [
                {
                    "id": 999,
                    "name": "Shared Story",
                    "saved_at": "2026-08-10T11:00:00",
                    "text": "Newer browser copy",
                },
                {
                    "id": 200,
                    "name": "Another Story",
                    "saved_at": "2026-08-10T09:00:00",
                    "text": "Another project",
                },
            ]
        },
    )
    assert imported.status_code == 200
    projects = imported.get_json()["projects"]
    assert len(projects) == 2
    by_name = {project["name"]: project for project in projects}
    assert by_name["Shared Story"]["id"] == "100"
    assert by_name["Shared Story"]["text"] == "Newer browser copy"

    listed = client.get("/api/projects")
    assert listed.status_code == 200
    assert [project["name"] for project in listed.get_json()["projects"]] == [
        "Shared Story",
        "Another Story",
    ]

    deleted = client.delete("/api/projects/100")
    assert deleted.status_code == 200
    remaining = client.get("/api/projects").get_json()["projects"]
    assert [project["name"] for project in remaining] == ["Another Story"]


def test_frontend_migrates_local_storage_then_uses_backend_projects():
    javascript = (PROJECT_ROOT / "static" / "js" / "main.js").read_text(encoding="utf-8")

    assert "'/api/projects/import'" in javascript
    assert "fetch('/api/projects'" in javascript
    assert "localStorage.removeItem(PROJECT_STORAGE_KEY)" in javascript
    assert "localStorage.setItem(PROJECT_STORAGE_KEY" not in javascript
    assert "let savedProjects = []" in javascript

