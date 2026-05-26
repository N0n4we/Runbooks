#!/usr/bin/env python3
"""Send a notification email via SMTP.

Reads SMTP configuration from environment variables:
    SMTP_HOST       (required)
    SMTP_PORT       (optional, default 587, STARTTLS)
    SMTP_USER       (required)
    SMTP_PASS       (required)
    NOTIFY_EMAIL    (required, recipient)

Usage:
    notify_email.py SUBJECT BODY
    notify_email.py SUBJECT < body.txt
    echo "body" | notify_email.py SUBJECT

Exit codes:
    0  email sent OR SMTP not configured (treated as no-op)
    1  attempted to send but failed (network, auth, etc.)
    2  bad arguments
"""

from __future__ import annotations

import os
import smtplib
import sys
from email.mime.text import MIMEText


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: notify_email.py SUBJECT [BODY]", file=sys.stderr)
        return 2

    subject = argv[1]
    if len(argv) >= 3:
        body = argv[2]
    else:
        body = sys.stdin.read()

    host = os.environ.get("SMTP_HOST")
    port_str = os.environ.get("SMTP_PORT", "587")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    to_email = os.environ.get("NOTIFY_EMAIL")

    if not all([host, user, password, to_email]):
        print("[notify] SMTP not configured, skipping email", file=sys.stderr)
        return 0

    try:
        port = int(port_str)
    except ValueError:
        print(f"[notify] Invalid SMTP_PORT: {port_str!r}", file=sys.stderr)
        return 1

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
    except Exception as exc:
        print(f"[notify] Failed to send email: {exc}", file=sys.stderr)
        return 1

    print(f"[notify] Email sent to {to_email}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
