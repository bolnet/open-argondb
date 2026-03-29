"""Tests for BackupManager."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from open_arangodb.backup.manager import BackupManager
from open_arangodb.models import BackupConfig, RestoreConfig


@pytest.fixture()
def manager() -> BackupManager:
    return BackupManager(
        host="http://localhost:8529",
        username="root",
        password="test",
    )


class TestDump:
    def test_dump_success(self, manager: BackupManager) -> None:
        """Successful dump returns BackupResult with success=True."""
        config = BackupConfig(output_dir="/tmp/backup-001", compress=True)

        with patch.object(manager, "_run_command", return_value=(0, "ok", "")) as mock_cmd:
            result = manager.dump(config)

        assert result.success is True
        assert result.path == "/tmp/backup-001"
        assert result.error is None

        cmd_args = mock_cmd.call_args[0][0]
        assert "arangodump" in cmd_args
        assert "--compress-output" in cmd_args
        assert "--output-directory" in cmd_args

    def test_dump_with_collections(self, manager: BackupManager) -> None:
        """Collections are passed as --collection flags."""
        config = BackupConfig(
            output_dir="/tmp/backup-002",
            collections=["users", "orders"],
        )

        with patch.object(manager, "_run_command", return_value=(0, "", "")) as mock_cmd:
            result = manager.dump(config)

        assert result.success is True
        cmd_args = mock_cmd.call_args[0][0]
        collection_flags = [
            cmd_args[i + 1]
            for i, v in enumerate(cmd_args)
            if v == "--collection"
        ]
        assert "users" in collection_flags
        assert "orders" in collection_flags

    def test_dump_failure(self, manager: BackupManager) -> None:
        """Non-zero exit code returns error result."""
        config = BackupConfig(output_dir="/tmp/backup-fail")

        with patch.object(
            manager, "_run_command", return_value=(1, "", "connection refused")
        ):
            result = manager.dump(config)

        assert result.success is False
        assert "connection refused" in (result.error or "")


class TestRestore:
    def test_restore_success(self, manager: BackupManager) -> None:
        """Successful restore returns BackupResult with success=True."""
        config = RestoreConfig(input_dir="/tmp/backup-001")

        with patch.object(manager, "_run_command", return_value=(0, "ok", "")) as mock_cmd:
            result = manager.restore(config)

        assert result.success is True
        assert result.path == "/tmp/backup-001"

        cmd_args = mock_cmd.call_args[0][0]
        assert "arangorestore" in cmd_args
        assert "--create-database" in cmd_args

    def test_restore_failure(self, manager: BackupManager) -> None:
        """Non-zero exit code returns error result."""
        config = RestoreConfig(input_dir="/tmp/backup-bad")

        with patch.object(
            manager, "_run_command", return_value=(1, "", "not found")
        ):
            result = manager.restore(config)

        assert result.success is False
        assert result.error is not None


class TestListBackups:
    def test_list_backups(self, manager: BackupManager) -> None:
        """Lists subdirectories as backup entries."""
        with (
            patch("os.path.isdir", side_effect=lambda p: True),
            patch("os.listdir", return_value=["2024-01-01", "2024-01-02"]),
            patch.object(manager, "_dir_size", return_value=1024),
        ):
            results = manager.list_backups("/backups")

        assert len(results) == 2
        assert results[0].timestamp == "2024-01-01"
        assert results[1].timestamp == "2024-01-02"
        assert results[0].size_bytes == 1024

    def test_list_backups_empty_dir(self, manager: BackupManager) -> None:
        """Returns empty list when base_dir does not exist."""
        with patch("os.path.isdir", return_value=False):
            results = manager.list_backups("/nonexistent")

        assert results == []


class TestVerifyBackup:
    def test_verify_backup_valid(self, manager: BackupManager) -> None:
        """Returns True when dump files are present."""
        with (
            patch("os.path.isdir", return_value=True),
            patch(
                "os.listdir",
                return_value=["users.data.json", "users.structure.json"],
            ),
        ):
            assert manager.verify_backup("/backups/2024-01-01") is True

    def test_verify_backup_compressed(self, manager: BackupManager) -> None:
        """Returns True when compressed dump files are present."""
        with (
            patch("os.path.isdir", return_value=True),
            patch(
                "os.listdir",
                return_value=["users.data.json.gz"],
            ),
        ):
            assert manager.verify_backup("/backups/2024-01-01") is True

    def test_verify_backup_empty(self, manager: BackupManager) -> None:
        """Returns False for an empty directory."""
        with (
            patch("os.path.isdir", return_value=True),
            patch("os.listdir", return_value=[]),
        ):
            assert manager.verify_backup("/backups/empty") is False

    def test_verify_backup_not_a_dir(self, manager: BackupManager) -> None:
        """Returns False when path is not a directory."""
        with patch("os.path.isdir", return_value=False):
            assert manager.verify_backup("/not/a/dir") is False
