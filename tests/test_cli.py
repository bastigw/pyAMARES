import subprocess


def test_import():
    import pyAMARES

    assert pyAMARES.__version__ is not None


def test_cli_entrypoint():
    # Attempt to call the entry point script
    result = subprocess.run(["amaresFit", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
