"""Telemetry module for logging and analytics."""
 
def log_event(event_name, data=None):
    """Log a telemetry event."""
    print(f"[Telemetry] {event_name}: {data}") 