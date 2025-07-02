"""
plimai.telemetry

This package contains telemetry and logging utilities.
"""

def log_event(event_name, details=None):
    """Log a telemetry event."""
    print(f"[Telemetry] {event_name}: {details}") 