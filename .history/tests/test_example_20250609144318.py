from plimai.example import hello

def test_hello():
    assert hello("World") == "Hello, World!"
    assert hello("Plim") == "Hello, Plim!" 