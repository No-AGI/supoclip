from src.clip_naming import clip_download_filename, clip_position


def test_filename_uses_readable_task_number_hook_and_source():
    clip = {"id": "real-id", "clip_order": 2, "hook_title": "Почему это сработало?"}

    assert clip_download_filename(clip, "Игра: Ночной лес") == (
        "clip_02 - Почему это сработало - Игра Ночной лес.mp4"
    )


def test_filename_sanitizes_platform_characters_and_preset_suffix():
    clip = {"clip_order": 1, "hook_title": 'CON: "bad"/name?\n'}

    name = clip_download_filename(clip, "<source>|name", preset="instagram/reels")

    assert name == "clip_01 - CON bad name - source name_instagram reels.mp4"
    assert len(name) <= 120
    assert not any(character in name for character in '<>:"/\\|?*\n')


def test_filename_handles_empty_values_and_long_labels():
    assert clip_download_filename({"clip_order": 1}, None) == "clip_01 - Clip.mp4"
    name = clip_download_filename(
        {"clip_order": 987654321, "hook_title": "Длинный " * 80}, "Источник " * 80
    )
    assert len(name) <= 120
    assert name.endswith(".mp4")
    assert not name.removesuffix(".mp4").endswith("Длинны")
    assert not name.removesuffix(".mp4").endswith("Источни")


def test_preset_suffix_does_not_change_long_base_name():
    clip = {"clip_order": 7, "hook_title": "Важный игровой момент " * 10}
    source = "Длинное название источника " * 10

    base_name = clip_download_filename(clip, source)
    preset_name = clip_download_filename(clip, source, preset="tiktok")

    assert base_name.removesuffix(".mp4") == preset_name.removesuffix("_tiktok.mp4")
    assert len(preset_name) <= 120


def test_position_uses_task_list_identity_and_clip_order_fallback():
    clip = {"id": "clip-real-id", "clip_order": 5}
    clips = [{"id": "first"}, clip]

    assert clip_position(clip, clips) == 2
    assert clip_position({"id": "missing", "clip_order": 5}, clips) == 5
