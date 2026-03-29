"""Validates OS-level encryption at rest."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone

from open_argondb.models import EncryptionRequirement, EncryptionStatus


class EncryptionValidator:
    """Validates OS-level encryption at rest (Enterprise ArangoDB equivalent)."""

    def check(self, data_directory: str | None = None) -> EncryptionStatus:
        """Auto-detect OS and check encryption status."""
        system = platform.system().lower()
        path = data_directory or "/"

        if system == "linux":
            return self._check_linux(path)
        if system == "darwin":
            return self._check_macos(path)
        if system == "windows":
            return self._check_windows(path)

        return EncryptionStatus(
            encrypted=False,
            method=None,
            details={"error": f"Unsupported platform: {system}"},
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _check_linux(self, path: str) -> EncryptionStatus:
        """Check LUKS/dm-crypt via lsblk and cryptsetup."""
        returncode, stdout, stderr = self._run_command(
            ["lsblk", "-o", "NAME,FSTYPE,MOUNTPOINT", "-J"]
        )

        if returncode != 0:
            return EncryptionStatus(
                encrypted=False,
                details={"error": stderr or "lsblk failed"},
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

        has_crypt = "crypt" in stdout.lower()

        return EncryptionStatus(
            encrypted=has_crypt,
            method="luks" if has_crypt else None,
            details={"lsblk_output": stdout, "path": path},
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _check_macos(self, path: str) -> EncryptionStatus:
        """Check FileVault via fdesetup and diskutil."""
        returncode, stdout, stderr = self._run_command(["fdesetup", "status"])

        if returncode != 0:
            return EncryptionStatus(
                encrypted=False,
                details={"error": stderr or "fdesetup failed"},
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

        is_on = "FileVault is On" in stdout

        return EncryptionStatus(
            encrypted=is_on,
            method="filevault" if is_on else None,
            details={"fdesetup_output": stdout, "path": path},
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def _check_windows(self, path: str) -> EncryptionStatus:
        """Check BitLocker via manage-bde."""
        returncode, stdout, stderr = self._run_command(
            ["manage-bde", "-status", path]
        )

        if returncode != 0:
            return EncryptionStatus(
                encrypted=False,
                details={"error": stderr or "manage-bde failed"},
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

        is_on = "Protection On" in stdout

        return EncryptionStatus(
            encrypted=is_on,
            method="bitlocker" if is_on else None,
            details={"manage_bde_output": stdout, "path": path},
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def check_cloud_aws(self, volume_id: str) -> EncryptionStatus:
        """Check AWS EBS encryption."""
        cmd = [
            "aws", "ec2", "describe-volumes",
            "--volume-ids", volume_id,
            "--query", "Volumes[0].Encrypted",
            "--output", "text",
        ]

        returncode, stdout, stderr = self._run_command(cmd)

        if returncode != 0:
            return EncryptionStatus(
                encrypted=False,
                details={"error": stderr or "aws ec2 describe-volumes failed"},
                checked_at=datetime.now(timezone.utc).isoformat(),
            )

        is_encrypted = stdout.strip().lower() == "true"

        return EncryptionStatus(
            encrypted=is_encrypted,
            method="aws_ebs" if is_encrypted else None,
            details={"volume_id": volume_id, "raw_output": stdout.strip()},
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

    def validate(
        self,
        status: EncryptionStatus,
        requirement: EncryptionRequirement,
    ) -> tuple[bool, str]:
        """Check if encryption status meets requirements."""
        if not requirement.require_encryption:
            return True, "Encryption not required"

        if not status.encrypted:
            return False, "Encryption is required but not detected"

        if status.method and status.method not in requirement.allowed_methods:
            return (
                False,
                f"Encryption method '{status.method}' is not in allowed methods: "
                f"{requirement.allowed_methods}",
            )

        return True, "Encryption requirement satisfied"

    def _run_command(
        self, cmd: list[str], timeout: int = 30
    ) -> tuple[int, str, str]:
        """Mockable subprocess wrapper."""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
