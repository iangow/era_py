from pathlib import Path
import subprocess
import sys
import tomllib


def test_ibis_is_an_optional_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert not any(
        dependency.startswith("ibis-framework")
        for dependency in pyproject["project"]["dependencies"]
    )
    assert pyproject["project"]["optional-dependencies"]["ibis"] == [
        "ibis-framework[postgres]"
    ]


def test_importing_base_package_does_not_import_ibis():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, era_py; assert 'ibis' not in sys.modules",
        ],
        check=False,
    )

    assert result.returncode == 0
