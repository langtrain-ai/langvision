from plimai.example import hello, get_version

def test_hello():
    assert hello("World") == "Hello, World!"
    assert hello("Plim") == "Hello, Plim!"

def test_get_version():
    version = get_version()
    assert isinstance(version, str)
    assert len(version.split('.')) == 3  # Should be in format x.y.z 