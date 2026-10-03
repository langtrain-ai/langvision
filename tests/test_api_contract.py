"""
The remote clients must only call routes the Langtrain API server has, with
X-API-Key and the server's field names. See fake_langtrain_api.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from fake_langtrain_api import API_KEY, FakeAPI  # noqa: E402

from langvision.fast_model import FastVisionModel, LangvisionServerClient, VisionRemoteJob  # noqa: E402
from langvision.api.client import LangvisionClient, ServerConfig  # noqa: E402


@pytest.fixture
def api():
    server = FakeAPI()
    yield server
    server.close()
    assert server.unknown == [], f"called routes the server doesn't have: {server.unknown}"


def test_remote_job_lifecycle(api):
    client = LangvisionServerClient(API_KEY, api.url)
    job = client.create_job({"base_model": "m", "dataset_id": "ds", "training_method": "qlora", "task": "vision"})
    remote = VisionRemoteJob(job["id"], client)
    assert remote.status()["status"] == "completed"
    assert remote.cancel()
    assert remote.export("you/model")["export_id"] == "exp-1"
    assert client.get_telemetry("job-1")[0]["step"] == 10
    assert all(call[4] == API_KEY for call in api.calls)


def test_upload_with_api_key_explains_what_to_do(api, tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    with pytest.raises(PermissionError, match="dashboard"):
        LangvisionServerClient(API_KEY, api.url).upload_dataset(str(data))


def test_langvision_client(api):
    client = LangvisionClient(api_key=API_KEY, config=ServerConfig(base_url=api.url))
    assert client.validate()["valid"]
    job = client.get_job_status("job-1")
    assert job.job_id == "job-1"
    assert client.list_jobs()[0].job_id == "job-1"


def test_env_key_alone_does_not_switch_to_remote(monkeypatch):
    monkeypatch.delenv("LANGTRAIN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="remote=True"):
        FastVisionModel.from_pretrained("m", remote=True)
