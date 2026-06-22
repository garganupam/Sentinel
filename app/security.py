import hashlib
import hmac
from typing import Optional

def verify_signature(
    payload_body: bytes,
    secret: str,
    signature_header: Optional[str],
) -> bool:
    """Verify a GitHub webhook payload against its X-Hub-Signature-256 header."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False

    expected = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    received = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, received)