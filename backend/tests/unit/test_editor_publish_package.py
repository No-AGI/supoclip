from types import SimpleNamespace

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
