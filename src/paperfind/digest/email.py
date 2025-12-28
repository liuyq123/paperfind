"""SMTP email sending for digest."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List

from paperfind.config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    EMAIL_FROM,
)


def send_email(
    subject: str,
    html_body: str,
    to_addresses: List[str],
) -> None:
    """
    Send an HTML email via SMTP.

    Args:
        subject: Email subject line
        html_body: HTML content of the email
        to_addresses: List of recipient email addresses
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        raise ValueError(
            "SMTP_USER and SMTP_PASSWORD must be set in .env to send emails"
        )

    if not EMAIL_FROM:
        raise ValueError("EMAIL_FROM must be set in .env")

    if not to_addresses:
        raise ValueError("No recipient addresses provided")

    # Create message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_addresses)

    # Attach HTML body
    html_part = MIMEText(html_body, "html")
    msg.attach(html_part)

    # Send via SMTP
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(EMAIL_FROM, to_addresses, msg.as_string())

    print(f"Email sent to {', '.join(to_addresses)}")
