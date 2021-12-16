from importlib_resources import open_text

import pytest

import euphonic


class TestInstalledFiles:

    def test_license_is_installed(self):
        with open_text(euphonic, 'LICENSE') as fp:
            license_data = fp.readlines()
        assert 'GNU GENERAL PUBLIC LICENSE' in license_data[0]

    def test_citation_cff_is_installed(self):
        # yaml dependency is optional
        try:
            import yaml
        except ModuleNotFoundError:
            pytest.skip()
        with open_text(euphonic, 'CITATION.cff') as fp:
            citation_data = yaml.safe_load(fp)
        assert 'cff-version' in citation_data
