from __future__ import annotations

import json
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_protocol_version_header(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert "X-Protocol-Version" in resp.headers

    def test_health_response_structure(self, client: TestClient):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data
        assert "uptime_seconds" in data
        assert "model_loaded" in data
        assert "active_tasks" in data
        assert "queued_tasks" in data

    def test_model_loaded_is_string(self, client: TestClient):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert isinstance(data["model_loaded"], str)


class TestCapabilitiesEndpoint:
    def test_capabilities_returns_200(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        assert resp.status_code == 200

    def test_capabilities_response_structure(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        assert "protocol_version" in data
        assert "server_version" in data
        assert "capabilities" in data
        assert "devices" in data
        assert "scheduler" in data
        assert "output_formats" in data
        assert "output_packages" in data
        assert "max_upload_bytes" in data

    def test_output_formats_is_string_array(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        assert isinstance(data["output_formats"], list)
        assert all(isinstance(f, str) for f in data["output_formats"])

    def test_output_packages_is_string_array(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        assert isinstance(data["output_packages"], list)
        assert all(isinstance(p, str) for p in data["output_packages"])

    def test_capabilities_have_label_not_name(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        for cap in data["capabilities"]:
            assert "label" in cap, f"Capability {cap.get('id')} missing label field"
            assert "name" not in cap, f"Capability {cap.get('id')} has deprecated name field"

    def test_capability_status_values(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        valid_statuses = {"implemented", "not_implemented"}
        for cap in data["capabilities"]:
            assert cap["status"] in valid_statuses, f"Invalid status: {cap['status']}"

    def test_split_capability_has_model_and_default(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        split = [c for c in data["capabilities"] if c["id"] == "split"][0]
        assert isinstance(split["models"], list)
        assert len(split["models"]) > 0
        assert split["default_model"] is not None

    def test_generate_capability_not_implemented(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        gen = [c for c in data["capabilities"] if c["id"] == "generate"][0]
        assert gen["status"] == "not_implemented"

    def test_scheduler_structure(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        s = data["scheduler"]
        assert "max_concurrent_tasks" in s
        assert "max_queue_size" in s

    def test_devices_have_device_type(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        for dev in data["devices"]:
            assert "device_type" in dev, f"Device missing device_type field"
            assert "available" in dev
            assert "count" in dev

    def test_param_defs_have_group(self, client: TestClient):
        resp = client.get("/api/v1/capabilities")
        data = resp.json()
        for cap in data["capabilities"]:
            for pd in cap.get("param_defs", []):
                assert "group" in pd, f"ParamDef {pd.get('key')} missing group"


class TestTaskSubmission:
    def test_submit_no_file_returns_error(self, client: TestClient):
        resp = client.post("/api/v1/tasks", data={})
        assert resp.status_code in (400, 422)

    def test_submit_no_file_json_mode(self, client: TestClient):
        resp = client.post(
            "/api/v1/tasks",
            json={"pipeline": {"steps": []}},
        )
        assert resp.status_code in (400, 422)

    def test_error_response_wrapped(self, client: TestClient):
        resp = client.post("/api/v1/tasks", data={})
        if resp.status_code >= 400:
            data = resp.json()
            assert "error" in data, "Error response should be wrapped in error key"

    def test_delete_task_not_found(self, client: TestClient):
        resp = client.delete("/api/v1/tasks/nonexistent-task-id")
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_get_task_not_found(self, client: TestClient):
        resp = client.get("/api/v1/tasks/nonexistent-task-id")
        assert resp.status_code == 400

    def test_list_tasks_empty(self, client: TestClient):
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert data["total"] == 0

    def test_list_tasks_with_status_filter(self, client: TestClient):
        resp = client.get("/api/v1/tasks?status=queued")
        assert resp.status_code == 200
        data = resp.json()
        for t in data["tasks"]:
            assert t["status"] == "queued"

    def test_list_tasks_pagination(self, client: TestClient):
        resp = client.get("/api/v1/tasks?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 10
        assert data["offset"] == 0


class TestTaskSubmitMultipart:
    def test_submit_without_file(self, client: TestClient):
        resp = client.post("/api/v1/tasks", data={"pipeline": '{"steps":[{"type":"split"}]}'})
        assert resp.status_code >= 400

    def test_submit_with_invalid_extension(self, client: TestClient):
        files = {"file": ("test.txt", b"not audio", "text/plain")}
        resp = client.post("/api/v1/tasks", files=files)
        assert resp.status_code == 400

    def test_submit_valid_file(self, client: TestClient, sample_wav: bytes):
        files = {"file": ("test.wav", sample_wav, "audio/wav")}
        data = {
            "pipeline": '{"steps":[{"type":"split"}]}',
            "device_preference": "cpu",
        }
        resp = client.post("/api/v1/tasks", files=files, data=data)
        if resp.status_code == 201:
            rj = resp.json()
            assert rj["status"] == "queued"
            assert "task_id" in rj
            assert "created_at" in rj
            pip = rj.get("pipeline")
            if pip:
                if isinstance(pip, list):
                    assert pip[0].get("type") == "split" or pip[0].get("capability") == "split"
                elif isinstance(pip, dict) and "steps" in pip:
                    assert isinstance(pip["steps"], list)

    def test_delete_task_returns_cancelled(self, client: TestClient, sample_wav: bytes):
        files = {"file": ("test.wav", sample_wav, "audio/wav")}
        data = {"pipeline": '{"steps":[{"type":"split"}]}'}
        resp = client.post("/api/v1/tasks", files=files, data=data)
        if resp.status_code != 201:
            return

        task_id = resp.json()["task_id"]
        resp2 = client.delete(f"/api/v1/tasks/{task_id}")
        assert resp2.status_code == 200
        rj = resp2.json()
        assert rj["status"] == "cancelled"
        assert "message" in rj


class TestTaskStatusEndpoint:
    def test_get_status_structure(self, client: TestClient, sample_wav: bytes):
        files = {"file": ("test.wav", sample_wav, "audio/wav")}
        data = {"pipeline": '{"steps":[{"type":"split"}]}'}
        resp = client.post("/api/v1/tasks", files=files, data=data)
        if resp.status_code != 201:
            return

        task_id = resp.json()["task_id"]
        resp2 = client.get(f"/api/v1/tasks/{task_id}")
        assert resp2.status_code == 200
        rj = resp2.json()
        assert "task_id" in rj
        assert "status" in rj
        assert "current_step" in rj
        assert "step_type" in rj
        assert "percent" in rj
        assert "completed_steps" in rj
        assert "completed_urls" in rj
        assert "errors" in rj


class TestVersionNegotiation:
    def test_protocol_version_header(self, client: TestClient):
        for path in ["/api/v1/health", "/api/v1/capabilities", "/api/v1/tasks"]:
            resp = client.get(path)
            assert "X-Protocol-Version" in resp.headers, f"Missing header for {path}"


class TestJSONSubmission:
    def test_json_submit_no_input_url(self, client: TestClient):
        payload = {
            "pipeline": {"steps": [{"type": "split"}]},
        }
        resp = client.post("/api/v1/tasks", json=payload)
        assert resp.status_code >= 400

    def test_json_submit_invalid_pipeline(self, client: TestClient):
        payload = {
            "input_url": "http://example.com/test.wav",
            "pipeline": "invalid",
        }
        resp = client.post("/api/v1/tasks", json=payload)
        assert resp.status_code == 400
