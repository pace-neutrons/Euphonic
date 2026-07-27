"""Shared pytest fixtures for the Euphonic test suite."""

import platform

import pytest


@pytest.fixture
def brille_or_skip_if_unsupported():
    """Import brille, or skip this test on a known-unsupported platform.

    At the moment that platform is Linux ARM: wheels are not available and
    we're having trouble building them in CI.

    On other platforms, a missing brille import is treated as a real failure
    rather than skipped silently. To avoid ModuleNotFoundError on those
    platforms if brille was deliberately not installed, use "-m not brille" to
    deselect these tests instead.

    """
    try:
        import brille
    except ModuleNotFoundError:
        if platform.system() == 'Linux' and platform.machine() in (
            'aarch64',
            'arm64',
        ):
            pytest.skip('brille is not supported on Linux ARM')
        raise
    return brille
