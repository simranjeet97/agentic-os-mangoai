"""
guardrails/network_policy.py — NetworkPolicy

Whitelist-based outbound network filter. Validates URLs before agent network
calls and logs all external requests to the audit system.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from urllib.parse import urlparse

import yaml

from guardrails.models import NetworkPolicyResult

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_GUARDRAILS_CONFIG = Path(__file__).parent.parent / "config" / "guardrails.yaml"

_DEFAULT_ALLOWED_DOMAINS = [
    "api.duckduckgo.com",
    "duckduckgo.com",
    "pypi.org",
    "files.pythonhosted.org",
    "github.com",
    "raw.githubusercontent.com",
    "api.github.com",
    "ollama.local",
    "localhost",
    "127.0.0.1",
    "serpapi.com",
    "google.com",
    "googleapis.com",
    "huggingface.co",
]

# RFC 1918 private ranges + loopback + link-local
_PRIVATE_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]

_TOR_PATTERN = re.compile(r"\.onion$", re.IGNORECASE)


class NetworkCallLog:
    """Lightweight in-memory log of network calls for the current session."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(
        self,
        url: str,
        agent_id: str,
        allowed: bool,
        reason: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        self._entries.append(
            {
                "timestamp": (timestamp or datetime.utcnow()).isoformat() + "Z",
                "agent_id": agent_id,
                "url": url,
                "allowed": allowed,
                "reason": reason,
            }
        )

    def get_all(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def get_blocked(self) -> list[dict[str, Any]]:
        return [e for e in self._entries if not e["allowed"]]


class NetworkPolicy:
    """
    Whitelist-based outbound network filter.

    Rules applied in order:
    1. Tor (.onion) → always blocked
    2. Private IP ranges (RFC 1918, unless localhost is whitelisted) → blocked
    3. Domain matches whitelist → allowed
    4. Default → blocked

    Usage:
        policy = NetworkPolicy()
        result = policy.check_url("https://api.duckduckgo.com/search")
        if not result.allowed:
            raise NetworkPolicyViolation(result.reason)
        policy.log_external_call(url, agent_id, outcome="allowed")
    """

    def __init__(self, extra_allowed_domains: Optional[list[str]] = None) -> None:
        cfg = self._load_config()
        config_domains: list[str] = (
            cfg.get("guardrails", {})
               .get("network_policy", {})
               .get("allowed_domains", [])
        )
        self._allowed_domains: set[str] = set(
            _DEFAULT_ALLOWED_DOMAINS + config_domains + (extra_allowed_domains or [])
        )
        self._call_log = NetworkCallLog()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_url(self, url: str) -> NetworkPolicyResult:
        """
        Validate whether `url` is permitted by the network policy.

        Args:
            url: The full URL the agent wants to reach.

        Returns:
            NetworkPolicyResult with allowed=True/False and a human-readable reason.
        """
        try:
            parsed = urlparse(url)
        except Exception as exc:
            return NetworkPolicyResult(
                url=url,
                domain="",
                allowed=False,
                reason=f"URL parse error: {exc}",
            )

        domain = (parsed.hostname or "").lower()
        if not domain:
            return NetworkPolicyResult(
                url=url,
                domain=domain,
                allowed=False,
                reason="URL has no hostname",
            )

        # Rule 1: Tor
        if _TOR_PATTERN.search(domain):
            return NetworkPolicyResult(
                url=url,
                domain=domain,
                allowed=False,
                reason="Tor (.onion) addresses are blocked",
                matched_rule="tor_block",
            )

        # Rule 2: Private IP ranges (strict block if domain resolves to private IP)
        private_block = self._check_private_ip(domain)
        if private_block and domain not in self._allowed_domains:
            return NetworkPolicyResult(
                url=url,
                domain=domain,
                allowed=False,
                reason=f"Private/loopback IP range blocked: {domain}",
                matched_rule="rfc1918_block",
            )

        # Rule 3: Whitelist match (exact domain or parent domain)
        matched_rule = self._matches_whitelist(domain)
        if matched_rule:
            return NetworkPolicyResult(
                url=url,
                domain=domain,
                allowed=True,
                reason=f"Domain matches whitelist rule: {matched_rule}",
                matched_rule=matched_rule,
            )

        # Rule 4: Default deny
        return NetworkPolicyResult(
            url=url,
            domain=domain,
            allowed=False,
            reason=(
                f"Domain {domain!r} is not in the network whitelist. "
                "Add it to config/guardrails.yaml → network_policy.allowed_domains."
            ),
            matched_rule="default_deny",
        )

    def log_external_call(
        self,
        url: str,
        agent_id: str,
        outcome: str = "allowed",
    ) -> None:
        """
        Record an outbound network call to the in-memory session log.

        For persistent audit storage, the GuardrailMiddleware also writes
        to AuditLogger separately.
        """
        self._call_log.record(
            url=url,
            agent_id=agent_id,
            allowed=(outcome == "allowed"),
            reason=outcome,
        )

    def add_allowed_domain(self, domain: str) -> None:
        """Dynamically add a domain to the whitelist (runtime only)."""
        self._allowed_domains.add(domain.lower())

    def get_call_log(self) -> list[dict[str, Any]]:
        """Return all network calls logged in this session."""
        return self._call_log.get_all()

    def get_blocked_calls(self) -> list[dict[str, Any]]:
        """Return only blocked network calls from this session."""
        return self._call_log.get_blocked()

    @property
    def allowed_domains(self) -> set[str]:
        return frozenset(self._allowed_domains)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _matches_whitelist(self, domain: str) -> Optional[str]:
        """
        Return the matched whitelist rule if domain is allowed, else None.
        Supports exact match and parent-domain match (e.g. "github.com"
        also covers "api.github.com").
        """
        # Exact match
        if domain in self._allowed_domains:
            return domain

        # Parent-domain match
        parts = domain.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[i:])
            if parent in self._allowed_domains:
                return parent

        return None

    @staticmethod
    def _check_private_ip(host: str) -> bool:
        """Return True if host is a private or loopback IP address."""
        try:
            addr = ipaddress.ip_address(host)
            return any(addr in net for net in _PRIVATE_RANGES)
        except ValueError:
            return False  # Not an IP address (it's a hostname)

    @staticmethod
    def _load_config() -> dict[str, Any]:
        if _GUARDRAILS_CONFIG.exists():
            with open(_GUARDRAILS_CONFIG) as f:
                return yaml.safe_load(f) or {}
        return {}
