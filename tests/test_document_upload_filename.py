import io

import app as application


def test_extract_document_accepts_non_ascii_filename():
    client = application.app.test_client()
    data = {
        "file": (
            io.BytesIO("第一段。\n第二段。".encode("utf-8")),
            "第一章 测试.txt",
        )
    }
    response = client.post(
        "/api/extract-document",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["format"] == "Plain Text"
    assert "第一段" in payload["text"]


def test_extract_document_keeps_ascii_filename_behaviour():
    client = application.app.test_client()
    data = {"file": (io.BytesIO(b"hello"), "notes.txt")}
    response = client.post(
        "/api/extract-document",
        data=data,
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["success"] is True
