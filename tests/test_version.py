from plimai.utils import get_project_version, parse_version

def test_version_format():
    version = get_project_version()
    parts = version.split('.')
    assert len(parts) == 3
    assert all(part.isdigit() or part[:-1].isdigit() for part in parts)

def test_parse_version():
    # Standard version
    assert parse_version('1.2.3') == (1, 2, 3, '')
    # Alpha version
    assert parse_version('1.2.3a1') == (1, 2, 3, 'a1')
    # Beta version
    assert parse_version('1.2.3b2') == (1, 2, 3, 'b2')
    # Release candidate
    assert parse_version('1.2.3rc3') == (1, 2, 3, 'rc3')
