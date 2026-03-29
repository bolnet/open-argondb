"""Tests for EncryptionValidator."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from open_arangodb.encryption.validator import EncryptionValidator
from open_arangodb.models import EncryptionRequirement, EncryptionStatus


@pytest.fixture()
def validator() -> EncryptionValidator:
    return EncryptionValidator()


class TestCheckMacos:
    def test_filevault_on(self, validator: EncryptionValidator) -> None:
        """Detects FileVault when fdesetup reports On."""
        with (
            patch("open_arangodb.encryption.validator.platform") as mock_platform,
            patch.object(
                validator,
                "_run_command",
                return_value=(0, "FileVault is On.", ""),
            ),
        ):
            mock_platform.system.return_value = "Darwin"
            status = validator.check("/var/lib/arangodb3")

        assert status.encrypted is True
        assert status.method == "filevault"

    def test_filevault_off(self, validator: EncryptionValidator) -> None:
        """Detects no encryption when FileVault is Off."""
        with (
            patch("open_arangodb.encryption.validator.platform") as mock_platform,
            patch.object(
                validator,
                "_run_command",
                return_value=(0, "FileVault is Off.", ""),
            ),
        ):
            mock_platform.system.return_value = "Darwin"
            status = validator.check("/data")

        assert status.encrypted is False
        assert status.method is None


class TestCheckLinux:
    def test_luks_detected(self, validator: EncryptionValidator) -> None:
        """Detects LUKS when lsblk reports crypt."""
        lsblk_output = '{"blockdevices": [{"name":"sda1","fstype":"crypto_LUKS","mountpoint":null}]}'

        with (
            patch("open_arangodb.encryption.validator.platform") as mock_platform,
            patch.object(
                validator,
                "_run_command",
                return_value=(0, lsblk_output, ""),
            ),
        ):
            mock_platform.system.return_value = "Linux"
            status = validator.check("/data")

        assert status.encrypted is True
        assert status.method == "luks"

    def test_no_encryption(self, validator: EncryptionValidator) -> None:
        """No encryption detected on plain filesystem."""
        lsblk_output = '{"blockdevices": [{"name":"sda1","fstype":"ext4","mountpoint":"/"}]}'

        with (
            patch("open_arangodb.encryption.validator.platform") as mock_platform,
            patch.object(
                validator,
                "_run_command",
                return_value=(0, lsblk_output, ""),
            ),
        ):
            mock_platform.system.return_value = "Linux"
            status = validator.check("/data")

        assert status.encrypted is False
        assert status.method is None


class TestCheckAwsEncrypted:
    def test_aws_encrypted_volume(self, validator: EncryptionValidator) -> None:
        """Detects encrypted AWS EBS volume."""
        with patch.object(
            validator, "_run_command", return_value=(0, "True\n", "")
        ):
            status = validator.check_cloud_aws("vol-abc123")

        assert status.encrypted is True
        assert status.method == "aws_ebs"
        assert status.details["volume_id"] == "vol-abc123"

    def test_aws_unencrypted_volume(self, validator: EncryptionValidator) -> None:
        """Detects unencrypted AWS EBS volume."""
        with patch.object(
            validator, "_run_command", return_value=(0, "False\n", "")
        ):
            status = validator.check_cloud_aws("vol-xyz")

        assert status.encrypted is False


class TestValidate:
    def test_validate_passes(self, validator: EncryptionValidator) -> None:
        """Encrypted status with allowed method passes."""
        status = EncryptionStatus(encrypted=True, method="luks")
        requirement = EncryptionRequirement(require_encryption=True)

        ok, msg = validator.validate(status, requirement)

        assert ok is True
        assert "satisfied" in msg.lower()

    def test_validate_fails_not_encrypted(self, validator: EncryptionValidator) -> None:
        """Fails when encryption is required but not detected."""
        status = EncryptionStatus(encrypted=False)
        requirement = EncryptionRequirement(require_encryption=True)

        ok, msg = validator.validate(status, requirement)

        assert ok is False
        assert "not detected" in msg.lower()

    def test_validate_fails_wrong_method(self, validator: EncryptionValidator) -> None:
        """Fails when encryption method is not in allowed list."""
        status = EncryptionStatus(encrypted=True, method="custom_cipher")
        requirement = EncryptionRequirement(
            require_encryption=True,
            allowed_methods=["luks", "filevault"],
        )

        ok, msg = validator.validate(status, requirement)

        assert ok is False
        assert "custom_cipher" in msg

    def test_validate_not_required(self, validator: EncryptionValidator) -> None:
        """Always passes when encryption is not required."""
        status = EncryptionStatus(encrypted=False)
        requirement = EncryptionRequirement(require_encryption=False)

        ok, msg = validator.validate(status, requirement)

        assert ok is True
        assert "not required" in msg.lower()
