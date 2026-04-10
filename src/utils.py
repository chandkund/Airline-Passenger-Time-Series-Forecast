def ensure_directory(path):
    """Ensure a path exists."""
    import os

    os.makedirs(path, exist_ok=True)
