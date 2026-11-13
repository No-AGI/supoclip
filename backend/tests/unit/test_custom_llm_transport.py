import pytest
import httpx
import asyncio
import datetime
import ssl
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from src.custom_llm_transport import validate_base_url, CheckedNetworkBackend, CheckedTransport


def test_reject_unsafe_urls():
    for value in ["http://example.com/v1", "https://user:pass@example.com/v1", "https://example.com/v1?key=secret", "https://127.0.0.1/v1", "https://169.254.169.254/v1"]:
        with pytest.raises(ValueError):
            validate_base_url(value)
    assert validate_base_url("https://example.com/v1/") == "https://example.com/v1"


def test_local_access_requires_explicit_config():
    assert validate_base_url("http://127.0.0.1:11434/v1", {"127.0.0.1"}) == "http://127.0.0.1:11434/v1"
    with pytest.raises(ValueError):
        validate_base_url("http://127.0.0.1:11434/v1")


@pytest.mark.asyncio
async def test_dns_checked_and_connection_pinned(monkeypatch):
    captured = []
    async def resolve(*args, **kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]
    async def connect(host, port, **kwargs):
        captured.append(host)
        return object()
    monkeypatch.setattr("src.custom_llm_transport.anyio.getaddrinfo", resolve)
    backend = CheckedNetworkBackend("example.com", 443, False)
    monkeypatch.setattr(backend.delegate, "connect_tcp", connect)
    await backend.connect_tcp("example.com", 443)
    assert captured == ["93.184.216.34"]
    async def rebinding(*args, **kwargs):
        return [(2, 1, 6, "", ("127.0.0.1", 443))]
    monkeypatch.setattr("src.custom_llm_transport.anyio.getaddrinfo", rebinding)
    with pytest.raises(ValueError):
        await backend.connect_tcp("example.com", 443)
    assert captured == ["93.184.216.34"]


@pytest.mark.asyncio
async def test_redirect_response_rejected():
    transport = CheckedTransport("https://example.com/v1")
    async def response(request):
        return httpx.Response(302, headers={"location": "https://other.example/"}, request=request)
    transport._inner.handle_async_request = response
    with pytest.raises(ValueError):
        await transport.handle_async_request(httpx.Request("POST", "https://example.com/v1/chat/completions"))
    await transport.aclose()


@pytest.mark.asyncio
async def test_real_tls_hostname_sni_and_redirect(monkeypatch, tmp_path):
    """Use real httpx/httpcore, local TLS only, and a temporary test CA."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.datetime.now(datetime.timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "model.test")])
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("model.test")]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    seen_sni = []
    server_context.set_servername_callback(lambda _socket, server_name, _context: seen_sni.append(server_name))
    async def respond(reader, writer):
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            status = b"302 Found\r\nLocation: https://elsewhere.test/" if b"/redirect" in request else b"200 OK"
            writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
    server = await asyncio.start_server(respond, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    original_resolve = __import__("src.custom_llm_transport", fromlist=["anyio"]).anyio.getaddrinfo
    async def resolve(host, port, **kwargs):
        return await original_resolve("127.0.0.1", port, **kwargs)
    monkeypatch.setattr("src.custom_llm_transport.anyio.getaddrinfo", resolve)
    original_context = ssl.create_default_context
    monkeypatch.setattr("src.custom_llm_transport.ssl.create_default_context", lambda: original_context(cafile=str(cert_path)))
    try:
        async with httpx.AsyncClient(transport=CheckedTransport(f"https://model.test:{port}/v1", {"model.test"}), trust_env=False) as client:
            assert (await client.get(f"https://model.test:{port}/v1/check")).status_code == 200
            with pytest.raises(ValueError, match="redirect"):
                await client.get(f"https://model.test:{port}/redirect")
        assert seen_sni and all(host == "model.test" for host in seen_sni)
        async with httpx.AsyncClient(transport=CheckedTransport(f"https://wrong.test:{port}/v1", {"wrong.test"}), trust_env=False) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get(f"https://wrong.test:{port}/v1/check")
    finally:
        server.close()
        await server.wait_closed()
