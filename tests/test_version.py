from src.plimai.utils import get_project_version

def test_version_format():
    version = get_project_version()
    parts = version.split('.')
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
