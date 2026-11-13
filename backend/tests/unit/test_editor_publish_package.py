from types import SimpleNamespace

import pytest

from src.services.editor_publish_package import _export_source


def test_storage_export_uses_durable_object_even_when_local_copy_exists(tmp_path):
    (tmp_path / "export-job.mp4").write_bytes(b"temporary")
    storage = SimpleNamespace(
        enabled=True,
        editor_reference=lambda directory, name: "s3://bucket/" + name,
        object_size=lambda reference: 123,
    )
    assert _export_source(SimpleNamespace(directory=tmp_path), storage, "job") == (
        "s3://bucket/export-job.mp4", 123
    )


def test_local_export_uses_the_machine_copy_without_touching_storage(tmp_path):
    local = tmp_path / "export-job.mp4"
    local.write_bytes(b"local-bytes")
    storage = SimpleNamespace(
        enabled=False,
        bucket=None,
        editor_reference=lambda directory, name: "s3://bucket/" + name,
        object_size=lambda reference: pytest.fail("object must not be probed in local mode"),
    )
    assert _export_source(SimpleNamespace(directory=tmp_path), storage, "job") == (
        str(local), len(b"local-bytes")
    )


def test_local_export_reports_missing_media_as_a_clear_error(tmp_path):
    storage = SimpleNamespace(
        enabled=False,
        bucket=None,
        editor_reference=lambda directory, name: "s3://bucket/" + name,
        object_size=lambda reference: None,
    )
    with pytest.raises(FileNotFoundError):
        _export_source(SimpleNamespace(directory=tmp_path), storage, "job")
