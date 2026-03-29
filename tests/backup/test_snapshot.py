"""Tests for SnapshotHelper."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from open_arangodb.backup.snapshot import SnapshotHelper
from open_arangodb.models import SnapshotConfig


@pytest.fixture()
def helper() -> SnapshotHelper:
    return SnapshotHelper()


class TestLvmSnapshot:
    def test_lvm_snapshot_success(self, helper: SnapshotHelper) -> None:
        """LVM snapshot succeeds with correct command."""
        config = SnapshotConfig(method="lvm", volume="/dev/vg0/data", label="test-snap")

        with patch.object(helper, "_run_command", return_value=(0, "", "")) as mock_cmd:
            result = helper.create_snapshot(config)

        assert result.success is True
        assert result.path == "test-snap"

        cmd = mock_cmd.call_args[0][0]
        assert "lvcreate" in cmd
        assert "--snapshot" in cmd
        assert "/dev/vg0/data" in cmd

    def test_lvm_snapshot_failure(self, helper: SnapshotHelper) -> None:
        """LVM snapshot returns error on failure."""
        config = SnapshotConfig(method="lvm", volume="/dev/vg0/data")

        with patch.object(
            helper, "_run_command", return_value=(1, "", "insufficient space")
        ):
            result = helper.create_snapshot(config)

        assert result.success is False
        assert "insufficient space" in (result.error or "")


class TestZfsSnapshot:
    def test_zfs_snapshot_success(self, helper: SnapshotHelper) -> None:
        """ZFS snapshot succeeds with correct command."""
        config = SnapshotConfig(method="zfs", volume="tank/data", label="snap1")

        with patch.object(helper, "_run_command", return_value=(0, "", "")) as mock_cmd:
            result = helper.create_snapshot(config)

        assert result.success is True
        assert result.path == "tank/data@snap1"

        cmd = mock_cmd.call_args[0][0]
        assert cmd == ["zfs", "snapshot", "tank/data@snap1"]


class TestAwsEbsSnapshot:
    def test_aws_ebs_snapshot_success(self, helper: SnapshotHelper) -> None:
        """AWS EBS snapshot succeeds."""
        config = SnapshotConfig(
            method="aws_ebs", volume="vol-abc123", label="my-snap"
        )

        with patch.object(helper, "_run_command", return_value=(0, "{}", "")) as mock_cmd:
            result = helper.create_snapshot(config)

        assert result.success is True
        cmd = mock_cmd.call_args[0][0]
        assert "aws" in cmd
        assert "vol-abc123" in cmd

    def test_aws_ebs_no_volume(self, helper: SnapshotHelper) -> None:
        """AWS EBS snapshot fails without volume ID."""
        config = SnapshotConfig(method="aws_ebs")

        result = helper.create_snapshot(config)

        assert result.success is False
        assert "volume" in (result.error or "").lower()


class TestDetectSnapshotMethod:
    def test_detect_lvm(self, helper: SnapshotHelper) -> None:
        """Detects LVM when lvcreate is available."""
        with patch("shutil.which", side_effect=lambda b: "/usr/sbin/lvcreate" if b == "lvcreate" else None):
            assert helper.detect_snapshot_method() == "lvm"

    def test_detect_zfs(self, helper: SnapshotHelper) -> None:
        """Detects ZFS when zfs binary is available."""
        with patch("shutil.which", side_effect=lambda b: "/usr/sbin/zfs" if b == "zfs" else None):
            assert helper.detect_snapshot_method() == "zfs"

    def test_detect_none(self, helper: SnapshotHelper) -> None:
        """Returns None when no snapshot tools found."""
        with patch("shutil.which", return_value=None):
            assert helper.detect_snapshot_method() is None


class TestUnknownMethod:
    def test_unknown_method_raises(self, helper: SnapshotHelper) -> None:
        """Unknown snapshot method raises ValueError."""
        config = SnapshotConfig(method="btrfs")

        with pytest.raises(ValueError, match="Unknown snapshot method"):
            helper.create_snapshot(config)
