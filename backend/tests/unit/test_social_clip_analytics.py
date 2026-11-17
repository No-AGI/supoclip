import sqlite3
from types import SimpleNamespace

import pytest

from src.repositories.social_repository import SocialRepository


class SQLiteResult:
    def __init__(self, cursor):
        self.cursor = cursor

    def fetchall(self):
        return [SimpleNamespace(**dict(row)) for row in self.cursor.fetchall()]


class SQLiteDB:
    """Execute the repository query against an isolated, in-memory fixture."""

    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE tasks (id TEXT PRIMARY KEY, user_id TEXT);
            CREATE TABLE generated_clips (
                id TEXT PRIMARY KEY, task_id TEXT, hook_title TEXT, clip_order INTEGER
            );
            CREATE TABLE social_posts (
                id TEXT PRIMARY KEY, user_id TEXT, task_id TEXT, clip_id TEXT,
                provider TEXT, status TEXT, hook_title TEXT, title TEXT
            );
            CREATE TABLE social_post_metrics (
                id TEXT PRIMARY KEY, post_id TEXT, captured_at TEXT,
                views INTEGER, likes INTEGER, comments INTEGER, shares INTEGER, saves INTEGER
            );
            INSERT INTO tasks VALUES ('task-a', 'user-a'), ('task-b', 'user-b');
            INSERT INTO generated_clips VALUES ('clip-a', 'task-a', 'Clip hook', 1);
            INSERT INTO generated_clips VALUES ('clip-b', 'task-b', 'Other tenant', 1);
            INSERT INTO social_posts VALUES
                ('post-1', 'user-a', 'task-a', 'clip-a', 'youtube', 'published', NULL, 'Post title'),
                ('post-2', 'user-a', 'task-a', 'clip-a', 'youtube', 'published', NULL, 'Post title'),
                ('post-3', 'user-a', 'task-a', 'clip-a', 'youtube', 'published', NULL, 'Post title'),
                ('post-draft', 'user-a', 'task-a', 'clip-a', 'youtube', 'scheduled', NULL, 'Draft'),
                ('post-other', 'user-b', 'task-b', 'clip-b', 'youtube', 'published', NULL, 'Other');
            INSERT INTO social_post_metrics VALUES
                ('m1-old', 'post-1', '2026-11-12T10:00:00Z', 10, 1, 1, 1, 1),
                ('m1-new', 'post-1', '2026-11-12T11:00:00Z', 15, 2, 2, 2, 2),
                ('m1-null-time', 'post-1', NULL, 999, 999, 999, 999, 999),
                ('m2-new', 'post-2', '2026-11-12T10:30:00Z', 4, NULL, 3, NULL, NULL),
                ('draft-m', 'post-draft', '2026-11-12T12:00:00Z', 999, 999, 999, 999, 999),
                ('other-m', 'post-other', '2026-11-12T12:00:00Z', 888, 888, 888, 888, 888);
            """
        )

    async def execute(self, statement, params):
        cursor = self.connection.execute(str(statement), params)
        return SQLiteResult(cursor)


@pytest.mark.asyncio
async def test_clip_analytics_executes_latest_snapshot_aggregation_with_tenant_and_null_guards():
    db = SQLiteDB()

    rows = await SocialRepository.get_clip_level_performance(db, "user-a")

    assert rows == [
        {
            "clip_id": "clip-a",
            "clip_order": 1,
            "title": "Clip hook",
            "provider": "youtube",
            "posts": 3,
            "measured_posts": 2,
            "views": None,  # one of three published posts has no snapshot
            "likes": None,  # missing in one of the contributing snapshots
            "comments": None,
            "shares": None,
            "saves": None,
            "snapshot_from": "2026-11-12T10:30:00Z",
            "snapshot_through": "2026-11-12T11:00:00Z",
        }
    ]


@pytest.mark.asyncio
async def test_clip_analytics_sums_metrics_when_every_published_post_has_a_value():
    db = SQLiteDB()
    db.connection.execute("DELETE FROM social_posts WHERE id = 'post-3'")
    db.connection.execute("UPDATE social_post_metrics SET likes=3, comments=4, shares=1, saves=2 WHERE id='m2-new'")

    [row] = await SocialRepository.get_clip_level_performance(db, "user-a")

    assert row["posts"] == 2
    assert row["measured_posts"] == 2
    assert row["views"] == 19
    assert row["likes"] == 5
    assert row["comments"] == 6
    assert row["shares"] == 3
    assert row["saves"] == 4
