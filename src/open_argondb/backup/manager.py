"""Backup manager wrapping arangodump/arangorestore."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone

from open_argondb.models import BackupConfig, BackupResult, RestoreConfig


class BackupManager:
    """Hot Backup equivalent using arangodump/arangorestore and filesystem snapshots."""

    def __init__(
        self,
        host: str = "http://localhost:8529",
        username: str = "root",
        password: str = "",
    ) -> None:
        self._host = host
        self._username = username
        self._password = password

    def dump(self, config: BackupConfig) -> BackupResult:
        """Run arangodump. Returns BackupResult."""
        cmd = [
            "arangodump",
            "--server.endpoint", self._host,
            "--server.username", self._username,
            "--server.password", self._password,
            "--output-directory", config.output_dir,
        ]

        if config.collections:
            for coll in config.collections:
                cmd.extend(["--collection", coll])

        if config.include_system:
            cmd.append("--include-system-collections")

        if config.compress:
            cmd.extend(["--compress-output", "true"])

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=config.output_dir,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"arangodump exited with code {returncode}",
            )

        return BackupResult(
            path=config.output_dir,
            timestamp=datetime.now(timezone.utc).isoformat(),
            collections=config.collections or [],
            success=True,
        )

    def restore(self, config: RestoreConfig) -> BackupResult:
        """Run arangorestore. Returns BackupResult."""
        cmd = [
            "arangorestore",
            "--server.endpoint", self._host,
            "--server.username", self._username,
            "--server.password", self._password,
            "--input-directory", config.input_dir,
        ]

        if config.collections:
            for coll in config.collections:
                cmd.extend(["--collection", coll])

        if config.create_database:
            cmd.extend(["--create-database", "true"])

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=config.input_dir,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"arangorestore exited with code {returncode}",
            )

        return BackupResult(
            path=config.input_dir,
            timestamp=datetime.now(timezone.utc).isoformat(),
            collections=config.collections or [],
            success=True,
        )

    def list_backups(self, base_dir: str) -> list[BackupResult]:
        """List existing backup directories."""
        results: list[BackupResult] = []

        if not os.path.isdir(base_dir):
            return results

        for entry in sorted(os.listdir(base_dir)):
            full_path = os.path.join(base_dir, entry)
            if os.path.isdir(full_path):
                size = self._dir_size(full_path)
                results.append(
                    BackupResult(
                        path=full_path,
                        timestamp=entry,
                        size_bytes=size,
                        success=True,
                    )
                )

        return results

    def verify_backup(self, path: str) -> bool:
        """Check backup directory has expected structure."""
        if not os.path.isdir(path):
            return False

        entries = os.listdir(path)
        if not entries:
            return False

        has_dump_files = any(
            e.endswith(".data.json") or e.endswith(".data.json.gz")
            for e in entries
        )
        has_structure = any(
            e.endswith(".structure.json") for e in entries
        )

        return has_dump_files or has_structure

    def _dir_size(self, path: str) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    total += os.path.getsize(fpath)
                except OSError:
                    pass
        return total

    def _run_command(
        self, cmd: list[str], timeout: int = 600
    ) -> tuple[int, str, str]:
        """Run shell command and return (returncode, stdout, stderr)."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
