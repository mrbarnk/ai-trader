from __future__ import annotations

from email.message import EmailMessage
import smtplib

from .settings import (
    MAIL_FROM_ADDRESS,
    MAIL_LOG_ENABLED,
    SMTP_HOST,
    SMTP_PASSWORD,
    SMTP_PORT,
    SMTP_USE_SSL,
    SMTP_USE_TLS,
    SMTP_USERNAME,
)


def send_email(to_address: str, subject: str, body: str) -> bool:
    if MAIL_LOG_ENABLED:
        print(
            "\n".join(
                [
                    "---- EMAIL SEND (LOG ONLY) ----",
                    f"From: {MAIL_FROM_ADDRESS}",
                    f"To: {to_address}",
                    f"Subject: {subject}",
                    "",
                    body,
                    "---- END EMAIL ----",
                ]
            )
        )
    if not SMTP_HOST:
        return False
    message = EmailMessage()
    message["From"] = MAIL_FROM_ADDRESS
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(body)
    try:
        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        try:
            server.ehlo()
            if SMTP_USE_TLS and not SMTP_USE_SSL:
                server.starttls()
                server.ehlo()
            if SMTP_USERNAME and SMTP_PASSWORD:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
        finally:
            server.quit()
    except Exception as exc:  # pragma: no cover - transport errors are environment-specific
        if MAIL_LOG_ENABLED:
            print(f"SMTP send failed: {exc}")
        return False
    return True
