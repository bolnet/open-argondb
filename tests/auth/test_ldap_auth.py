"""Tests for LDAPAuthenticator."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from open_argondb.auth.ldap_auth import LDAPAuthenticator
from open_argondb.models import AuthResult, LDAPConfig, RoleMapping


@pytest.fixture
def ldap_config() -> LDAPConfig:
    return LDAPConfig(
        server_url="ldap://localhost:389",
        base_dn="dc=example,dc=com",
        bind_dn="cn=admin,dc=example,dc=com",
        bind_password="admin_pass",
        tls=False,
    )


@pytest.fixture
def role_mappings() -> list[RoleMapping]:
    return [
        RoleMapping(ldap_group="db-admins", argondb_role="admin"),
        RoleMapping(ldap_group="db-readers", argondb_role="reader"),
        RoleMapping(ldap_group="db-writers", argondb_role="writer"),
    ]


@pytest.fixture
def authenticator(ldap_config, role_mappings) -> LDAPAuthenticator:
    return LDAPAuthenticator(ldap_config, role_mappings)


def _make_mock_ldap3(
    user_found: bool = True,
    bind_succeeds: bool = True,
    groups: list[str] | None = None,
) -> MagicMock:
    """Create a mock ldap3 module."""
    mock_ldap3 = MagicMock()

    # Mock Server
    mock_server = MagicMock()
    mock_ldap3.Server.return_value = mock_server

    # Mock entry for user search
    if user_found:
        mock_entry = MagicMock()
        mock_entry.entry_dn = "uid=testuser,ou=people,dc=example,dc=com"
        entries = [mock_entry]
    else:
        entries = []

    # Mock admin connection (for search)
    mock_admin_conn = MagicMock()
    mock_admin_conn.entries = entries

    # Mock group entries
    group_entries = []
    for g in (groups or []):
        ge = MagicMock()
        ge.cn = g
        group_entries.append(ge)

    # Track connection calls to return admin conn first, then user conn
    call_count = {"n": 0}
    original_entries = entries

    def connection_side_effect(*args, **kwargs):
        call_count["n"] += 1
        conn = MagicMock()
        if call_count["n"] == 1:
            # Admin bind for user search
            conn.entries = original_entries
        elif call_count["n"] == 2:
            # User bind for password verification
            if not bind_succeeds:
                raise Exception("Invalid credentials")
        else:
            # Group search connection
            conn.entries = group_entries
        return conn

    mock_ldap3.Connection = MagicMock(side_effect=connection_side_effect)

    return mock_ldap3


class TestLDAPAuthenticator:
    def test_authenticate_success(self, authenticator) -> None:
        """Successful auth should return authenticated=True with groups/roles."""
        mock_ldap3 = _make_mock_ldap3(
            user_found=True, bind_succeeds=True, groups=["db-admins", "db-readers"]
        )
        with patch.dict(sys.modules, {"ldap3": mock_ldap3}):
            result = authenticator.authenticate("testuser", "password123")

        assert isinstance(result, AuthResult)
        assert result.authenticated is True
        assert result.username == "testuser"
        assert "admin" in result.roles
        assert "reader" in result.roles

    def test_authenticate_user_not_found(self, authenticator) -> None:
        """Auth should fail when user is not found in LDAP."""
        mock_ldap3 = _make_mock_ldap3(user_found=False)
        with patch.dict(sys.modules, {"ldap3": mock_ldap3}):
            result = authenticator.authenticate("nobody", "password")

        assert result.authenticated is False
        assert result.error == "User not found"

    def test_authenticate_wrong_password(self, authenticator) -> None:
        """Auth should fail when user bind raises an exception."""
        mock_ldap3 = _make_mock_ldap3(user_found=True, bind_succeeds=False)
        with patch.dict(sys.modules, {"ldap3": mock_ldap3}):
            result = authenticator.authenticate("testuser", "wrong_pass")

        assert result.authenticated is False
        assert result.error is not None
        assert "Invalid credentials" in result.error

    def test_get_groups(self, authenticator) -> None:
        """get_groups should return group CNs."""
        mock_ldap3 = _make_mock_ldap3(groups=["db-admins", "dev-team"])

        # Reset call counter by creating fresh mock for group lookup
        mock_ldap3_groups = MagicMock()
        mock_server = MagicMock()
        mock_ldap3_groups.Server.return_value = mock_server

        mock_conn = MagicMock()
        ge1 = MagicMock()
        ge1.cn = "db-admins"
        ge2 = MagicMock()
        ge2.cn = "dev-team"
        mock_conn.entries = [ge1, ge2]
        mock_ldap3_groups.Connection.return_value = mock_conn

        with patch.dict(sys.modules, {"ldap3": mock_ldap3_groups}):
            groups = authenticator.get_groups(
                "uid=testuser,ou=people,dc=example,dc=com"
            )

        assert groups == ["db-admins", "dev-team"]

    def test_map_roles(self, authenticator) -> None:
        """map_roles should return mapped roles for matching groups."""
        roles = authenticator.map_roles(["db-admins", "db-readers"])
        assert "admin" in roles
        assert "reader" in roles
        assert len(roles) == 2

    def test_map_roles_no_match(self, authenticator) -> None:
        """map_roles should return empty list when no groups match."""
        roles = authenticator.map_roles(["unknown-group", "other-group"])
        assert roles == []

    def test_ldap3_not_installed(self, ldap_config) -> None:
        """Auth should return error when ldap3 is not installed."""
        auth = LDAPAuthenticator(ldap_config)

        # Remove ldap3 from sys.modules to simulate it not being installed
        with patch.dict(sys.modules, {"ldap3": None}):
            # Also patch builtins __import__ to raise ImportError for ldap3
            original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

            def mock_import(name, *args, **kwargs):
                if name == "ldap3":
                    raise ImportError("No module named 'ldap3'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = auth.authenticate("testuser", "password")

        assert result.authenticated is False
        assert "ldap3 not installed" in (result.error or "")
