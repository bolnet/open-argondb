"""Filesystem snapshot helpers for point-in-time backup."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone

from open_argondb.models import BackupResult, SnapshotConfig

_SUPPORTED_METHODS = {"lvm", "zfs", "aws_ebs", "gcp_disk", "azure_disk"}


class SnapshotHelper:
    """Filesystem snapshot helpers for point-in-time backup."""

    def create_snapshot(self, config: SnapshotConfig) -> BackupResult:
        """Create a filesystem snapshot using the configured method."""
        if config.method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown snapshot method '{config.method}'. "
                f"Supported: {sorted(_SUPPORTED_METHODS)}"
            )

        dispatch = {
            "lvm": self._lvm_snapshot,
            "zfs": self._zfs_snapshot,
            "aws_ebs": self._aws_ebs_snapshot,
            "gcp_disk": self._gcp_disk_snapshot,
            "azure_disk": self._azure_disk_snapshot,
        }

        return dispatch[config.method](config)

    def _lvm_snapshot(self, config: SnapshotConfig) -> BackupResult:
        """Create an LVM snapshot."""
        label = config.label or f"snap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        cmd = [
            "lvcreate",
            "--snapshot",
            "--name", label,
            "--size", "10G",
        ]
        if config.volume:
            cmd.append(config.volume)

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=label,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"lvcreate exited with code {returncode}",
            )

        return BackupResult(
            path=label,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=True,
        )

    def _zfs_snapshot(self, config: SnapshotConfig) -> BackupResult:
        """Create a ZFS snapshot."""
        label = config.label or f"snap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        volume = config.volume or "pool/dataset"
        snapshot_name = f"{volume}@{label}"

        cmd = ["zfs", "snapshot", snapshot_name]
        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=snapshot_name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"zfs snapshot exited with code {returncode}",
            )

        return BackupResult(
            path=snapshot_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=True,
        )

    def _aws_ebs_snapshot(self, config: SnapshotConfig) -> BackupResult:
        """Create an AWS EBS snapshot."""
        if not config.volume:
            return BackupResult(
                path="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error="volume (EBS volume ID) is required for aws_ebs snapshots",
            )

        description = config.label or "open-argondb-snapshot"
        cmd = [
            "aws", "ec2", "create-snapshot",
            "--volume-id", config.volume,
            "--description", description,
        ]

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=config.volume,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"aws ec2 create-snapshot exited with code {returncode}",
            )

        return BackupResult(
            path=config.volume,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=True,
        )

    def _gcp_disk_snapshot(self, config: SnapshotConfig) -> BackupResult:
        """Create a GCP disk snapshot."""
        if not config.volume:
            return BackupResult(
                path="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error="volume (disk name) is required for gcp_disk snapshots",
            )

        label = config.label or f"snap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        cmd = [
            "gcloud", "compute", "disks", "snapshot",
            config.volume,
            "--snapshot-names", label,
        ]

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=config.volume,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"gcloud snapshot exited with code {returncode}",
            )

        return BackupResult(
            path=config.volume,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=True,
        )

    def _azure_disk_snapshot(self, config: SnapshotConfig) -> BackupResult:
        """Create an Azure managed disk snapshot."""
        if not config.volume:
            return BackupResult(
                path="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error="volume (disk resource ID) is required for azure_disk snapshots",
            )

        label = config.label or f"snap-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        cmd = [
            "az", "snapshot", "create",
            "--name", label,
            "--source", config.volume,
        ]

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return BackupResult(
                path=config.volume,
                timestamp=datetime.now(timezone.utc).isoformat(),
                success=False,
                error=stderr or f"az snapshot exited with code {returncode}",
            )

        return BackupResult(
            path=config.volume,
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=True,
        )

    def detect_snapshot_method(self) -> str | None:
        """Auto-detect available snapshot tool."""
        checks = [
            ("lvcreate", "lvm"),
            ("zfs", "zfs"),
            ("aws", "aws_ebs"),
            ("gcloud", "gcp_disk"),
            ("az", "azure_disk"),
        ]
        for binary, method in checks:
            if shutil.which(binary) is not None:
                return method
        return None

    def _run_command(
        self, cmd: list[str], timeout: int = 300
    ) -> tuple[int, str, str]:
        """Run shell command -- mockable for tests."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
