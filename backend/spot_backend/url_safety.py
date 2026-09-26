"""SSRF guards for user-supplied Ollama base URLs."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# AWS/GCP-style link-local metadata (always blocked).
_METADATA_IPV4 = ipaddress.ip_address("169.254.169.254")


class OllamaUrlNotAllowedError(ValueError):
    """Raised when an Ollama URL points at a disallowed network target."""


def _is_link_local(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv4Address):
        return ip in ipaddress.ip_network("169.254.0.0/16")
    return ip.is_link_local or ip in ipaddress.ip_network("fe80::/10")


def _is_metadata(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv4Address):
        return ip == _METADATA_IPV4
    return False


def _is_loopback(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(ip.is_loopback)


def _is_rfc1918(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if not isinstance(ip, ipaddress.IPv4Address):
        return False
    return bool(
        ip in ipaddress.ip_network("10.0.0.0/8")
        or ip in ipaddress.ip_network("172.16.0.0/12")
        or ip in ipaddress.ip_network("192.168.0.0/16")
    )


def _classify_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    if _is_metadata(ip) or _is_link_local(ip):
        return "blocked"
    if _is_loopback(ip) or _is_rfc1918(ip):
        return "allowed_private"
    return "public"


def resolve_host_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    if not hostname.strip():
        raise OllamaUrlNotAllowedError("host is required")
    host = hostname.strip().lower()
    if host == "localhost":
        return [ipaddress.ip_address("127.0.0.1")]
    try:
        literal = ipaddress.ip_address(host)
        return [literal]
    except ValueError:
        pass
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        for family, _type, _proto, _canon, sockaddr in socket.getaddrinfo(host, None):
            if family == socket.AF_INET:
                ips.append(ipaddress.ip_address(sockaddr[0]))
            elif family == socket.AF_INET6:
                ips.append(ipaddress.ip_address(sockaddr[0]))
    except OSError as e:
        raise OllamaUrlNotAllowedError(f"could not resolve host: {host}") from e
    if not ips:
        raise OllamaUrlNotAllowedError(f"could not resolve host: {host}")
    return ips


def validate_ollama_base_url(url: str, *, allow_public: bool = False) -> str:
    """Return normalized base URL (no trailing slash) or raise."""
    raw = (url or "").strip()
    if not raw:
        raise OllamaUrlNotAllowedError("Ollama URL is required")
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        raise OllamaUrlNotAllowedError("Ollama URL must use http or https")
    if not parsed.hostname:
        raise OllamaUrlNotAllowedError("Ollama URL must include a hostname")
    if parsed.username or parsed.password:
        raise OllamaUrlNotAllowedError("Ollama URL must not include credentials")

    ips = resolve_host_ips(parsed.hostname)
    classes = {_classify_ip(ip) for ip in ips}
    if "blocked" in classes:
        raise OllamaUrlNotAllowedError(
            "That Ollama URL targets a blocked address (link-local or cloud metadata)"
        )
    if "public" in classes and not allow_public:
        raise OllamaUrlNotAllowedError(
            "That Ollama host is on the public internet. Confirm external access in the setup wizard."
        )

    if parsed.port:
        netloc = f"{parsed.hostname}:{parsed.port}"
    else:
        netloc = parsed.hostname
    return f"{parsed.scheme}://{netloc}".rstrip("/")
