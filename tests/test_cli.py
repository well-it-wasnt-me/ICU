import subprocess


def test_help_output():
    """
    Run the program with --help and verify that the help message is printed.
    """
    cmd = ["python", "main.py", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 0, "Help command failed."
    # Check that the output contains expected text (e.g., usage information)
    assert "usage:" in result.stdout.lower(), "Help text does not contain usage information."
