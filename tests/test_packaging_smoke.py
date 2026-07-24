import email
import json
import os
import site
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from email.message import Message
from pathlib import Path
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional

import pytest
from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_NAME = "carefree-core"
RUN_PACKAGING_SMOKE = os.environ.get("CFCORE_RUN_PACKAGING_SMOKE") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_PACKAGING_SMOKE,
    reason="set CFCORE_RUN_PACKAGING_SMOKE=1 to run packaging smoke tests",
)


class BuiltDistributions(NamedTuple):
    source: Path
    sdist: Path
    wheel: Path
    metadata: Message


def _offline_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PIP_EXTRA_INDEX_URL", None)
    env.pop("PIP_FIND_LINKS", None)
    env.pop("PIP_INDEX_URL", None)
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_CACHE_DIR"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run(
    command: List[str],
    *,
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {' '.join(command)}\n"
        f"{completed.stdout}"
    )
    return completed


def _copy_source_tree(destination: Path) -> None:
    destination.mkdir()
    for filename in [
        "pyproject.toml",
        "README.md",
        "LICENSE",
    ]:
        source = PROJECT_ROOT / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)
    shutil.copytree(
        PROJECT_ROOT / "core",
        destination / "core",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    sentinel = destination / "tests" / "packaging_sentinel" / "__init__.py"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text(
        "# This package must not be present in the wheel.\n", encoding="utf-8"
    )


def _message_from_bytes(payload: bytes) -> Message:
    return email.message_from_bytes(payload)


def _wheel_metadata(wheel: Path) -> Message:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_files) == 1
        return _message_from_bytes(archive.read(metadata_files[0]))


def _sdist_metadata(sdist: Path) -> Message:
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.count("/") == 1 and member.name.endswith("/PKG-INFO")
        ]
        assert len(members) == 1
        extracted = archive.extractfile(members[0])
        assert extracted is not None
        return _message_from_bytes(extracted.read())


@pytest.fixture(scope="session")
def built_distributions(
    tmp_path_factory: pytest.TempPathFactory,
) -> BuiltDistributions:
    workdir = tmp_path_factory.mktemp("packaging")
    source = workdir / "source"
    _copy_source_tree(source)

    configured_dist = os.environ.get("CFCORE_DIST_DIR")
    if configured_dist is None:
        dist = workdir / "dist"
        dist.mkdir()
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--sdist",
                "--wheel",
                "--outdir",
                str(dist),
                str(source),
            ],
            cwd=workdir,
            env=_offline_env(),
        )
    else:
        dist = Path(configured_dist)
        if not dist.is_absolute():
            dist = PROJECT_ROOT / dist
        dist = dist.resolve()
        assert dist.is_dir(), f"CFCORE_DIST_DIR is not a directory: {dist}"

    sdists = list(dist.glob("*.tar.gz"))
    wheels = list(dist.glob("*.whl"))
    assert len(sdists) == 1
    assert len(wheels) == 1
    metadata = _wheel_metadata(wheels[0])
    assert metadata["Name"] == DIST_NAME
    assert metadata["Version"]
    return BuiltDistributions(source, sdists[0], wheels[0], metadata)


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _create_dependency_sharing_venv(path: Path) -> Path:
    # CI installs one constrained dependency set before running this test. A
    # child venv does not inherit its parent venv's site-packages merely from
    # system_site_packages=True, so explicitly add the current interpreter's
    # non-user site-packages after the child's own purelib.
    # Use the constrained parent pip/setuptools/wheel through the .pth below.
    # Bundled ensurepip versions differ by Python release and can predate PEP
    # 621; installing with one of those would not exercise our pinned backend.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(path)
    python = _venv_python(path)
    completed = _run(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        cwd=path,
        env=_offline_env(),
    )
    child_purelib = Path(completed.stdout.strip()).resolve()
    parent_site_packages = []
    for raw_path in site.getsitepackages():
        site_packages = Path(raw_path).resolve()
        if site_packages != child_purelib and site_packages.is_dir():
            parent_site_packages.append(site_packages)
    assert parent_site_packages, "current interpreter has no site-packages"
    pth_contents = "".join(
        f"{site_packages}\n" for site_packages in dict.fromkeys(parent_site_packages)
    )
    (child_purelib / "_cfcore_parent_site_packages.pth").write_text(
        pth_contents,
        encoding="utf-8",
    )
    return python


def _install_local(
    python: Path,
    target: Path,
    *,
    editable: bool = False,
) -> None:
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        "--ignore-installed",
    ]
    if editable:
        command.append("--editable")
    command.append(str(target))
    _run(command, cwd=target.parent, env=_offline_env())


def _probe_installed_package(
    python: Path,
    *,
    cwd: Path,
    expected_core: Path,
    expected_version: str,
    check_metadata: bool,
) -> None:
    probe = """
import importlib.metadata
import json
from pathlib import Path

import core
import core.learn

module_dir = Path(core.__file__).resolve().parent
expected_core = Path({expected_core}).resolve()
assert module_dir == expected_core, (module_dir, expected_core)
if {check_metadata}:
    actual_version = importlib.metadata.version({dist_name})
    assert actual_version == {expected_version}, (actual_version, {expected_version})
print(json.dumps({{"core": str(module_dir)}}))
""".format(
        expected_core=json.dumps(str(expected_core)),
        check_metadata=repr(check_metadata),
        dist_name=json.dumps(DIST_NAME),
        expected_version=json.dumps(expected_version),
    )
    _run([str(python), "-c", probe], cwd=cwd, env=_offline_env())


def test_distribution_metadata_and_wheel_contents(
    built_distributions: BuiltDistributions,
) -> None:
    wheel_metadata = built_distributions.metadata
    sdist_metadata = _sdist_metadata(built_distributions.sdist)
    for field in ["Name", "Version", "Requires-Python"]:
        assert sdist_metadata[field] == wheel_metadata[field]
    assert wheel_metadata["Requires-Python"] == ">=3.8"
    assert sorted(
        str(Requirement(requirement))
        for requirement in sdist_metadata.get_all("Requires-Dist") or []
    ) == sorted(
        str(Requirement(requirement))
        for requirement in wheel_metadata.get_all("Requires-Dist") or []
    )

    with zipfile.ZipFile(built_distributions.wheel) as archive:
        names = archive.namelist()
    assert "core/__init__.py" in names
    assert "core/py.typed" in names
    assert any(
        ".dist-info/" in name and name.rsplit("/", 1)[-1] == "LICENSE" for name in names
    )
    assert not any(name.startswith("tests/") for name in names)
    assert not any(".egg-info/" in name for name in names)
    assert not any("__pycache__/" in name or name.endswith(".pyc") for name in names)


@pytest.mark.parametrize("artifact_name", ["sdist", "wheel"])
def test_clean_artifact_install(
    artifact_name: str,
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    python = _create_dependency_sharing_venv(environment)
    artifact = getattr(built_distributions, artifact_name)
    _install_local(python, artifact)
    _run(
        [str(python), "-m", "pip", "check"],
        cwd=tmp_path,
        env=_offline_env(),
    )

    purelib_probe = (
        "import json, sysconfig; " "print(json.dumps(sysconfig.get_path('purelib')))"
    )
    completed = _run(
        [str(python), "-c", purelib_probe],
        cwd=tmp_path,
        env=_offline_env(),
    )
    purelib = Path(json.loads(completed.stdout.strip()))
    _probe_installed_package(
        python,
        cwd=tmp_path,
        expected_core=purelib / "core",
        expected_version=built_distributions.metadata["Version"],
        check_metadata=True,
    )


def test_editable_install(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    python = _create_dependency_sharing_venv(environment)
    _install_local(python, built_distributions.source, editable=True)
    _run(
        [str(python), "-m", "pip", "check"],
        cwd=tmp_path,
        env=_offline_env(),
    )
    _probe_installed_package(
        python,
        cwd=tmp_path,
        expected_core=built_distributions.source / "core",
        expected_version=built_distributions.metadata["Version"],
        check_metadata=True,
    )


def test_copy_in_import(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    project = tmp_path / "copy_in_project"
    project.mkdir()
    copied_core = project / "core"
    shutil.copytree(
        PROJECT_ROOT / "core",
        copied_core,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _probe_installed_package(
        Path(sys.executable),
        cwd=project,
        expected_core=copied_core,
        expected_version=built_distributions.metadata["Version"],
        check_metadata=False,
    )
