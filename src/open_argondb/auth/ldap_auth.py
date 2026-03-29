"""LDAP authenticator — app-layer LDAP auth for ArangoDB Community."""

from __future__ import annotations

import logging

from open_argondb.models import AuthResult, LDAPConfig, RoleMapping

logger = logging.getLogger("open_argondb")


class LDAPAuthenticator:
    """App-layer LDAP auth — Enterprise ArangoDB equivalent."""

    def __init__(
        self,
        config: LDAPConfig,
        role_mappings: list[RoleMapping] | None = None,
    ) -> None:
        self._config = config
        self._role_mappings = role_mappings or []

    def authenticate(self, username: str, password: str) -> AuthResult:
        """Authenticate user against LDAP server."""
        try:
            import ldap3

            # Connect to LDAP
            server = ldap3.Server(self._config.server_url, use_ssl=self._config.tls)

            # Search for user
            user_filter = self._config.user_filter.replace("{username}", username)
            conn = ldap3.Connection(
                server,
                user=self._config.bind_dn,
                password=self._config.bind_password,
                auto_bind=True,
            )
            conn.search(self._config.base_dn, user_filter, attributes=["dn"])

            if not conn.entries:
                return AuthResult(
                    authenticated=False, username=username, error="User not found"
                )

            user_dn = str(conn.entries[0].entry_dn)
            conn.unbind()

            # Bind as user to verify password
            user_conn = ldap3.Connection(
                server, user=user_dn, password=password, auto_bind=True
            )
            user_conn.unbind()

            # Get groups
            groups = self.get_groups(user_dn)
            roles = self.map_roles(groups)

            return AuthResult(
                authenticated=True, username=username, groups=groups, roles=roles
            )

        except ImportError:
            return AuthResult(
                authenticated=False,
                username=username,
                error="ldap3 not installed",
            )
        except Exception as e:
            return AuthResult(
                authenticated=False, username=username, error=str(e)
            )

    def get_groups(self, user_dn: str) -> list[str]:
        """Retrieve groups for a user DN."""
        try:
            import ldap3

            server = ldap3.Server(self._config.server_url, use_ssl=self._config.tls)
            conn = ldap3.Connection(
                server,
                user=self._config.bind_dn,
                password=self._config.bind_password,
                auto_bind=True,
            )
            group_filter = self._config.group_filter.replace("{user_dn}", user_dn)
            conn.search(self._config.base_dn, group_filter, attributes=["cn"])
            groups = [str(entry.cn) for entry in conn.entries]
            conn.unbind()
            return groups
        except Exception as e:
            logger.error("LDAP group lookup failed: %s", e)
            return []

    def map_roles(self, groups: list[str]) -> list[str]:
        """Map LDAP groups to application roles."""
        roles = []
        for mapping in self._role_mappings:
            if mapping.ldap_group in groups:
                roles.append(mapping.argondb_role)
        return roles
