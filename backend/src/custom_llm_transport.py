"""Pinned-DNS transport for administrator-configured model endpoints.

DNS is resolved and checked at connection time, then the checked IP is passed
to the network backend. TLS still verifies the original hostname. This avoids
the DNS rebinding gap of checking a hostname and resolving it again to connect.
"""

from __future__ import annotations

import ipaddress
import ssl
from urllib.parse import urlsplit

import anyio
import httpcore
import httpx


def validate_base_url(value: str, trusted_local_hosts: set[str] | None = None) -> str:
    trusted = trusted_local_hosts or set()
    try:
        parsed = urlsplit(value.strip())
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid endpoint URL") from exc
    if not host or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain credentials, query or fragment")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and host in trusted):
        raise ValueError("Endpoint must use HTTPS unless explicitly trusted locally")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip and (ip.is_link_local or ip.is_multicast or ip.is_unspecified or (not ip.is_global and host not in trusted)):
        raise ValueError("Endpoint address is not allowed")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid endpoint port")
    return value.strip().rstrip("/")


class CheckedNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, hostname: str, port: int, trusted_local: bool):
        self.hostname = hostname.lower()
        self.port = port
        self.trusted_local = trusted_local
        self.delegate = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        if host.lower() != self.hostname or port != self.port:
            raise ValueError("Endpoint host changed")
        records = await anyio.getaddrinfo(host, port, type=1)
        addresses = list(dict.fromkeys(item[4][0] for item in records))
        if not addresses:
            raise ValueError("Endpoint has no address")
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if ip.is_link_local or ip.is_multicast or ip.is_unspecified or (not ip.is_global and not self.trusted_local):
                raise ValueError("Endpoint resolved to a disallowed address")
        # Connection is made to the checked literal IP, never to the hostname.
        return await self.delegate.connect_tcp(addresses[0], port, timeout=timeout, local_address=local_address, socket_options=socket_options)

    async def connect_unix_socket(self, *args, **kwargs):
        raise ValueError("Unix sockets are not supported")

    async def sleep(self, seconds):
        await self.delegate.sleep(seconds)


class CheckedTransport(httpx.AsyncBaseTransport):
    def __init__(self, base_url: str, trusted_local_hosts: set[str] | None = None):
        trusted = trusted_local_hosts or set()
        url = httpx.URL(validate_base_url(base_url, trusted))
        self.origin = (url.scheme, url.host.lower(), url.port or (443 if url.scheme == "https" else 80))
        self._inner = httpx.AsyncHTTPTransport(verify=ssl.create_default_context(), trust_env=False)
        # httpx 0.28's underlying httpcore pool accepts an explicit network
        # backend. Fail closed if a future HTTP client changes this contract.
        pool = self._inner._pool
        if not isinstance(pool, httpcore.AsyncConnectionPool) or not hasattr(pool, "_network_backend"):
            raise RuntimeError("Unsupported HTTP transport version")
        pool._network_backend = CheckedNetworkBackend(self.origin[1], self.origin[2], self.origin[1] in trusted)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        origin = (url.scheme, url.host.lower(), url.port or (443 if url.scheme == "https" else 80))
        if origin != self.origin:
            raise ValueError("Endpoint host changed")
        response = await self._inner.handle_async_request(request)
        if 300 <= response.status_code < 400:
            await response.aclose()
            raise ValueError("Endpoint redirects are not permitted")
        return response

    async def aclose(self):
        await self._inner.aclose()
