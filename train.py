#!/usr/bin/env python3
"""
LoraCI - Anima LoRA training entry point with email notification.

Usage:
    python train.py --config_file ~/models/topic.toml

Email notification is triggered on both success and failure. SMTP is
configured via environment variables (see scripts/notify_email.py for the
full list).

Why try/except wraps imports too: train_util.read_config_from_file calls
sys.exit(1) on bad config, which raises SystemExit (a BaseException, not
an Exception). To make sure failures during import, argument parsing, or
config loading still produce a notification email, the entire main path
runs inside `try ... except BaseException`.
"""

import os
import smtplib
import socket
import sys
import traceback
from datetime import datetime
from email.mime.text import MIMEText


def send_email(subject: str, body: str) -> None:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port_str = os.environ.get("SMTP_PORT", "587")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    to_email = os.environ.get("NOTIFY_EMAIL")

    if not all([smtp_host, smtp_user, smtp_pass, to_email]):
        print("[notify] SMTP not configured, skipping email notification", file=sys.stderr)
        return

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        print(f"[notify] Invalid SMTP_PORT: {smtp_port_str!r}", file=sys.stderr)
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[notify] Email sent to {to_email}", file=sys.stderr)
    except Exception as exc:
        # Never let a notification failure mask the original error.
        print(f"[notify] Failed to send email: {exc}", file=sys.stderr)


if __name__ == "__main__":
    config_name = "unknown"
    config_file_for_msg = "<not parsed>"
    output_for_msg = "<unknown>"
    start_time = datetime.now()

    try:
        # Imports are inside the try so an ImportError still triggers an
        # email instead of dying silently at module load.
        from anima_train_network import AnimaNetworkTrainer, setup_parser
        import library.train_util as train_util

        parser = setup_parser()
        args = parser.parse_args()
        train_util.verify_command_line_training_args(args)

        # Capture config_file for the email subject BEFORE read_config_from_file
        # mutates args (it strips the .toml extension and may exit on errors).
        if getattr(args, "config_file", None):
            config_name = os.path.splitext(os.path.basename(args.config_file))[0]
            config_file_for_msg = args.config_file

        args = train_util.read_config_from_file(args, parser)

        if getattr(args, "output_dir", None) and getattr(args, "output_name", None):
            output_for_msg = f"{args.output_dir}/{args.output_name}.safetensors"

        trainer = AnimaNetworkTrainer()
        trainer.train(args)

        elapsed = datetime.now() - start_time
        send_email(
            f"[LoraCI] {config_name} - training complete",
            f"Host: {socket.gethostname()}\n"
            f"Config: {config_file_for_msg}\n"
            f"Duration: {elapsed}\n"
            f"Output: {output_for_msg}\n",
        )
    except BaseException as e:
        # KeyboardInterrupt should never silently mail and continue; just
        # propagate so an interactive user can ctrl+c cleanly.
        if isinstance(e, KeyboardInterrupt):
            raise

        # A clean SystemExit (code 0 / None) is not a failure. argparse --help
        # and --output_config both reach this path; do not email or rewrite
        # the exit code.
        if isinstance(e, SystemExit) and (e.code is None or e.code == 0):
            raise

        elapsed = datetime.now() - start_time
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        send_email(
            f"[LoraCI] {config_name} - training failed",
            f"Host: {socket.gethostname()}\n"
            f"Config: {config_file_for_msg}\n"
            f"Duration: {elapsed}\n"
            f"Error: {type(e).__name__}: {e}\n\n"
            f"Traceback:\n{tb}\n",
        )

        # Preserve the original exit code when SystemExit carried one;
        # otherwise default to 1.
        if isinstance(e, SystemExit) and isinstance(e.code, int):
            sys.exit(e.code)
        sys.exit(1)
