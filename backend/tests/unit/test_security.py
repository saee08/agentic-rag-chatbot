from __future__ import annotations

from app.core.security import RateLimiter, hash_password, hash_token, new_access_token, verify_password
from app.services.document_service import safe_display_name


def test_password_hash_roundtrip_and_salting():
    h1, h2 = hash_password("s3cret-password"), hash_password("s3cret-password")
    assert h1 != h2  # random salt
    assert h1.startswith("scrypt$") and "s3cret" not in h1
    assert verify_password("s3cret-password", h1)
    assert not verify_password("wrong-password", h1)


def test_verify_rejects_malformed_hashes():
    for bad in ("", "plain", "bcrypt$1$2$3$4$5", "scrypt$x$y$z$salt$hash"):
        assert not verify_password("anything", bad)


def test_tokens_are_random_and_stored_hashed():
    t1, t2 = new_access_token(), new_access_token()
    assert t1 != t2 and len(t1) >= 40
    assert hash_token(t1) != t1 and len(hash_token(t1)) == 64


def test_rate_limiter_blocks_then_reports_retry_after():
    limiter = RateLimiter(limit=3, window_seconds=60)
    assert [limiter.hit("k") for _ in range(3)] == [None, None, None]
    retry = limiter.hit("k")
    assert retry is not None and 1 <= retry <= 61
    assert limiter.hit("other") is None  # keys are independent


def test_rate_limiter_bounded_memory():
    limiter = RateLimiter(limit=1, window_seconds=60, max_keys=10)
    for i in range(50):
        limiter.hit(f"k{i}")
    assert len(limiter._hits) <= 10


def test_display_names_cannot_carry_paths():
    assert safe_display_name("../../etc/passwd") == "passwd"
    assert safe_display_name("C:\\Windows\\system32\\evil.pdf") == "evil.pdf"
    assert safe_display_name("/abs/path/report.pdf") == "report.pdf"
    assert safe_display_name("..") == "document"
    assert safe_display_name("") == "document"
    assert safe_display_name("bad\x00name\n.pdf") == "badname.pdf"
    assert len(safe_display_name("a" * 500 + ".pdf")) <= 200
    assert safe_display_name("a" * 500 + ".pdf").endswith(".pdf")
