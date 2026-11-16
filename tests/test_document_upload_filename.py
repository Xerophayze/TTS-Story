import io

import pytest
import app as application


@pytest.mark.parametrize('filename', ['第一章.txt', 'Résumé.TXT', 'notes.txt', 'C:\\books\\第一章.txt'])
def test_upload_preserves_supported_suffix(filename):
    response = application.app.test_client().post('/api/extract-document', data={
        'file': (io.BytesIO('你好, world'.encode('utf-8')), filename),
    }, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.get_json()['text'] == '你好, world'
    assert response.get_json()['filename'].endswith('.txt')


def test_unsupported_extension_is_rejected():
    response = application.app.test_client().post('/api/extract-document', data={
        'file': (io.BytesIO(b'content'), '第一章.exe'),
    }, content_type='multipart/form-data')
    assert response.status_code == 400
