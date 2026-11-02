"""凭据与网关成对选择；离线配置检查不冒充账号鉴权。"""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import config
from tools.seedance_api import SeedanceAPI


@pytest.mark.parametrize("overrides,expected_key,expected_url", [
    ({}, "shared-test-key", "https://ark.cn-beijing.volces.com/api/plan/v3"),
    ({"ARK_BASE_URL": "https://ark.cn-beijing.volces.com/api/v3"},
     "shared-test-key", "https://ark.cn-beijing.volces.com/api/v3"),
    ({"ARK_BASE_URL_SEEDANCE": "https://video.example.test/api/v3"},
     "shared-test-key", "https://video.example.test/api/v3"),
    ({"ARK_API_KEY_SEEDANCE": "independent-test-key"},
     "independent-test-key", "https://ark.cn-beijing.volces.com/api/v3"),
    ({"ARK_API_KEY_SEEDANCE": "independent-test-key",
      "ARK_BASE_URL_SEEDANCE": "https://ark.cn-beijing.volces.com/api/plan/v3"},
     "independent-test-key", "https://ark.cn-beijing.volces.com/api/plan/v3"),
])
def test_video_credentials_follow_matching_gateway(monkeypatch, overrides, expected_key, expected_url):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    for name in ("ARK_BASE_URL", "ARK_API_KEY_SEEDANCE", "ARK_BASE_URL_SEEDANCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARK_API_KEY", "shared-test-key")
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    settings = runpy.run_path(str(Path(config.__file__)))
    assert settings["ARK_API_KEY_SEEDANCE"] == expected_key
    assert settings["ARK_BASE_URL_SEEDANCE"] == expected_url


@pytest.mark.parametrize("shared_key", [None, "shared-test-key"])
def test_client_uses_effective_video_key_and_url_without_network(monkeypatch, shared_key):
    ark = Mock()
    monkeypatch.setitem(sys.modules, "volcenginesdkarkruntime", SimpleNamespace(Ark=ark))
    monkeypatch.setattr(config, "ARK_API_KEY", shared_key)
    monkeypatch.setattr(config, "ARK_API_KEY_SEEDANCE", "video-test-key")
    monkeypatch.setattr(config, "ARK_BASE_URL_SEEDANCE", "https://ark.cn-beijing.volces.com/api/plan/v3")
    SeedanceAPI()
    ark.assert_called_once_with(api_key="video-test-key", base_url=config.ARK_BASE_URL_SEEDANCE)


def test_missing_effective_key_never_initializes_sdk(monkeypatch):
    ark = Mock()
    monkeypatch.setitem(sys.modules, "volcenginesdkarkruntime", SimpleNamespace(Ark=ark))
    monkeypatch.setattr(config, "ARK_API_KEY_SEEDANCE", None)
    with pytest.raises(RuntimeError, match="凭据"):
        SeedanceAPI()
    ark.assert_not_called()
