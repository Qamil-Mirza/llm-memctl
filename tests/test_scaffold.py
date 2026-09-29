"""Keeps `pytest` (the container's default command) green until real tests exist."""


def test_package_imports():
    import memctl

    assert memctl.__doc__
