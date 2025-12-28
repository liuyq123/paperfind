"""Email digest module for paperfind."""

from paperfind.digest.digest import run_digest
from paperfind.digest.email import send_email
from paperfind.digest.template import render_digest

__all__ = [
    "run_digest",
    "send_email",
    "render_digest",
]
