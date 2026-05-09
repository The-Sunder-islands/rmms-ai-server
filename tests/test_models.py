from __future__ import annotations

import pytest
from rmms_ai_server.models.protocol import (
    CapabilityStatus,
    TaskStatus,
    ParamType,
    FinalStatus,
    ProgressStatus,
    ParamDef,
    Capability,
    CapabilitiesResponse,
    HealthResponse,
    PipelineStep,
    StepInput,
    StepResultURL,
    StepError,
    Track,
    PartialResultEvent,
    ProgressSSEEvent,
    FinalResultEvent,
    TaskInfo,
    DeviceInfo,
    DeviceUnit,
    SchedulerInfo,
)
from rmms_ai_server.models.errors import (
    ErrorCode,
    RMMSAIError,
    InputError,
    ModelError,
    DeviceError,
    ServerError,
    AuthError,
    QuotaError,
    PipelineError,
    CapabilityError,
)


class TestCapabilityStatus:
    def test_implemented_value(self):
        assert CapabilityStatus.IMPLEMENTED.value == "implemented"

    def test_not_implemented_value(self):
        assert CapabilityStatus.NOT_IMPLEMENTED.value == "not_implemented"

    def test_no_degraded(self):
        assert not hasattr(CapabilityStatus, "DEGRADED")
        assert not hasattr(CapabilityStatus, "AVAILABLE")


class TestTaskStatus:
    def test_status_values(self):
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.PROCESSING.value == "processing"
        assert TaskStatus.DONE.value == "done"
        assert TaskStatus.PARTIAL_ERROR.value == "partial_error"
        assert TaskStatus.ERROR.value == "error"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_no_old_values(self):
        assert not hasattr(TaskStatus, "RUNNING")
        assert not hasattr(TaskStatus, "COMPLETED")
        assert not hasattr(TaskStatus, "FAILED")


class TestParamType:
    def test_type_values(self):
        assert ParamType.INT.value == "int"
        assert ParamType.FLOAT.value == "float"
        assert ParamType.STRING.value == "string"
        assert ParamType.BOOL.value == "bool"
        assert ParamType.ENUM.value == "enum"
        assert ParamType.MULTI_ENUM.value == "multi_enum"

    def test_no_old_types(self):
        assert not hasattr(ParamType, "INTEGER")
        assert not hasattr(ParamType, "BOOLEAN")
        assert not hasattr(ParamType, "ARRAY")


class TestFinalStatus:
    def test_final_statuses(self):
        assert FinalStatus.DONE.value == "done"
        assert FinalStatus.PARTIAL_ERROR.value == "partial_error"
        assert FinalStatus.ERROR.value == "error"
        assert FinalStatus.CANCELLED.value == "cancelled"


class TestParamDef:
    def test_int_param(self):
        p = ParamDef(key="shifts", type=ParamType.INT, label="Shifts", default=1, min_val=1, max_val=20, step=1, group="advanced")
        d = p.model_dump()
        assert d["key"] == "shifts"
        assert d["type"] == "int"
        assert d["min"] == 1
        assert d["max"] == 20

    def test_enum_param_with_options(self):
        p = ParamDef(
            key="model", type=ParamType.ENUM, label="Model", default="htdemucs_6s",
            options=[
                {"value": "htdemucs", "label": "HTDemucs (4 stems)"},
                {"value": "htdemucs_6s", "label": "HTDemucs 6-stem"},
            ],
            group="basic",
        )
        d = p.model_dump()
        assert len(d["options"]) == 2

    def test_multi_enum_param(self):
        p = ParamDef(key="stems", type=ParamType.MULTI_ENUM, label="Stems", default=["vocals"], options=[{"value": "vocals", "label": "Vocals"}], group="basic")
        assert p.type == ParamType.MULTI_ENUM

    def test_group_default(self):
        p = ParamDef(key="x", type=ParamType.INT, label="X")
        assert p.group == ""


class TestCapability:
    def test_field_names(self):
        c = Capability(id="split", label="Stem Separation", status=CapabilityStatus.IMPLEMENTED)
        d = c.model_dump()
        assert "label" in d
        assert "name" not in d

    def test_default_model_null(self):
        c = Capability(id="x", label="X")
        assert c.default_model is None

    def test_no_old_fields(self):
        c = Capability(id="x", label="X")
        d = c.model_dump()
        assert "input_types" not in d
        assert "output_types" not in d


class TestCapabilitiesResponse:
    def test_full_response(self):
        r = CapabilitiesResponse(
            protocol_version="1.0.0",
            server_version="1.0.0a1",
            capabilities=[],
            devices=[],
            scheduler=SchedulerInfo(max_concurrent_tasks=4, max_queue_size=20),
            output_formats=["wav", "flac", "mp3"],
            output_packages=["separate", "zip"],
            max_upload_bytes=524288000,
        )
        d = r.model_dump()
        assert d["protocol_version"] == "1.0.0"
        assert d["server_version"] == "1.0.0a1"
        assert "output_packages" in d
        assert "max_upload_bytes" in d
        assert isinstance(d["output_formats"], list)
        assert isinstance(d["output_formats"][0], str)


class TestHealthResponse:
    def test_model_loaded_is_string(self):
        h = HealthResponse(model_loaded="htdemucs_6s")
        d = h.model_dump()
        assert isinstance(d["model_loaded"], str)

    def test_model_loaded_default(self):
        h = HealthResponse()
        assert h.model_loaded == ""


class TestPipelineStep:
    def test_capability_alias_to_type(self):
        step = PipelineStep(capability="split", model="htdemucs_6s", params={"shifts": 1})
        d = step.model_dump(by_alias=True)
        assert d["type"] == "split"
        assert "capability" not in d or d.get("capability") is None

    def test_validate_with_type_key(self):
        step = PipelineStep(**{"type": "split"})
        assert step.capability == "split"

    def test_validate_with_capability_key(self):
        step = PipelineStep(capability="split")
        assert step.capability == "split"

    def test_model_field(self):
        step = PipelineStep(capability="split", model="htdemucs")
        assert step.model == "htdemucs"

    def test_model_params_field(self):
        step = PipelineStep(capability="midi", model_params={"precision": "float32"})
        assert step.model_params == {"precision": "float32"}

    def test_input_null_by_default(self):
        step = PipelineStep(capability="split")
        assert step.input is None

    def test_step_input(self):
        step = PipelineStep(capability="midi", input=StepInput(from_step=0, stem="vocals"))
        assert step.input.from_step == 0
        assert step.input.stem == "vocals"


class TestStepError:
    def test_error_is_dict(self):
        e = StepError(step_index=0, step_type="split", error={"code": "TEST", "message": "test"})
        d = e.model_dump()
        assert isinstance(d["error"], dict)
        assert d["error"]["code"] == "TEST"


class TestTrack:
    def test_full_track(self):
        t = Track(
            track_type="audio", stem="vocals", label="Vocals",
            url="/api/v1/files/abc/vocals.wav", format="wav",
            sample_rate=44100, duration=180.5, size_bytes=63580444,
        )
        d = t.model_dump()
        assert d["track_type"] == "audio"
        assert d["stem"] == "vocals"
        assert d["sample_rate"] == 44100


class TestPartialResultEvent:
    def test_track_inline(self):
        track = Track(track_type="audio", stem="vocals", label="V", url="/f/v.wav", format="wav")
        evt = PartialResultEvent(task_id="x", step_index=0, step_type="split", track=track)
        d = evt.model_dump(by_alias=True)
        assert "track" in d
        assert d["track"]["stem"] == "vocals"
        assert "data" not in d

    def test_track_none(self):
        evt = PartialResultEvent()
        assert evt.track is None


class TestProgressSSEEvent:
    def test_urls_is_flat_strings(self):
        evt = ProgressSSEEvent(status=ProgressStatus.COMPLETED, urls=["/api/v1/files/x/a.wav", "/api/v1/files/x/b.wav"])
        d = evt.model_dump()
        assert isinstance(d["urls"], list)
        assert all(isinstance(u, str) for u in d["urls"])


class TestFinalResultEvent:
    def test_completed_steps_and_failed_steps(self):
        evt = FinalResultEvent(
            status=FinalStatus.PARTIAL_ERROR,
            completed_steps=[0, 2], failed_steps=[1],
        )
        d = evt.model_dump()
        assert d["completed_steps"] == [0, 2]
        assert d["failed_steps"] == [1]


class TestDeviceInfo:
    def test_new_format(self):
        d = DeviceInfo(
            device_type="cuda", available=True, count=1,
            units=[DeviceUnit(device_index=0, name="RTX 4090", memory_total_mb=24564)],
            install_hint=None,
        )
        j = d.model_dump()
        assert j["device_type"] == "cuda"
        assert j["count"] == 1
        assert "type" not in j
        assert "name" not in j
        assert j["install_hint"] is None

    def test_install_hint(self):
        d = DeviceInfo(
            device_type="cuda", available=False, count=0, units=[],
            install_hint="pip install torch --index-url https://download.pytorch.org/whl/cu128",
        )
        assert d.install_hint is not None


class TestDeviceUnit:
    def test_new_format(self):
        u = DeviceUnit(device_index=0, name="RTX 4090", memory_total_mb=24564)
        j = u.model_dump()
        assert j["device_index"] == 0
        assert "id" not in j


class TestSchedulerInfo:
    def test_new_format(self):
        s = SchedulerInfo(max_concurrent_tasks=4, max_queue_size=20)
        j = s.model_dump()
        assert j["max_concurrent_tasks"] == 4
        assert j["max_queue_size"] == 20
        assert "type" not in j


class TestErrorCodes:
    def test_input_error_codes(self):
        assert ErrorCode.INPUT_FORMAT_UNSUPPORTED.value == "INPUT_FORMAT_UNSUPPORTED"
        assert ErrorCode.INPUT_TOO_LARGE.value == "INPUT_TOO_LARGE"
        assert ErrorCode.INPUT_CORRUPT.value == "INPUT_CORRUPT"

    def test_no_old_input_codes(self):
        assert not hasattr(ErrorCode, "INPUT_INVALID_FORMAT")
        assert not hasattr(ErrorCode, "INPUT_FILE_TOO_LARGE")
        assert not hasattr(ErrorCode, "INPUT_UNSUPPORTED_TYPE")

    def test_device_error_codes(self):
        assert ErrorCode.DEVICE_UNAVAILABLE.value == "DEVICE_UNAVAILABLE"
        assert ErrorCode.DEVICE_OOM.value == "DEVICE_OOM"
        assert not hasattr(ErrorCode, "DEVICE_NOT_AVAILABLE")
        assert not hasattr(ErrorCode, "DEVICE_BUSY")

    def test_auth_error_codes(self):
        assert ErrorCode.AUTH_REQUIRED.value == "AUTH_REQUIRED"
        assert ErrorCode.AUTH_INVALID.value == "AUTH_INVALID"
        assert not hasattr(ErrorCode, "AUTH_MISSING_KEY")
        assert not hasattr(ErrorCode, "AUTH_INVALID_KEY")

    def test_server_error_codes(self):
        assert ErrorCode.SERVER_BUSY.value == "SERVER_BUSY"
        assert ErrorCode.SERVER_ERROR.value == "SERVER_ERROR"
        assert not hasattr(ErrorCode, "SERVER_OVERLOADED")

    def test_capability_error_codes(self):
        assert ErrorCode.CAPABILITY_NOT_IMPLEMENTED.value == "CAPABILITY_NOT_IMPLEMENTED"
        assert not hasattr(ErrorCode, "CAPABILITY_NOT_AVAILABLE")

    def test_pipeline_error_codes(self):
        assert ErrorCode.PIPELINE_CIRCULAR_DEPENDENCY.value == "PIPELINE_CIRCULAR_DEPENDENCY"
        assert ErrorCode.PIPELINE_MISSING_INPUT.value == "PIPELINE_MISSING_INPUT"

    def test_quota_error_codes(self):
        assert ErrorCode.QUOTA_RATE_LIMIT.value == "QUOTA_RATE_LIMIT"


class TestErrorClasses:
    def test_rmms_error_to_dict(self):
        e = RMMSAIError(ErrorCode.SERVER_ERROR, "test", details={"x": 1}, http_status=500)
        d = e.to_dict()
        assert d["code"] == "SERVER_ERROR"
        assert d["message"] == "test"
        assert d["details"] == {"x": 1}

    def test_input_error_http_status(self):
        e = InputError(ErrorCode.INPUT_TOO_LARGE, "too large")
        assert e.http_status == 400

    def test_auth_required_401(self):
        e = AuthError(ErrorCode.AUTH_REQUIRED, "key required")
        assert e.http_status == 401

    def test_auth_invalid_403(self):
        e = AuthError(ErrorCode.AUTH_INVALID, "invalid key")
        assert e.http_status == 403

    def test_quota_error_429(self):
        e = QuotaError(ErrorCode.QUOTA_EXCEEDED, "quota")
        assert e.http_status == 429

    def test_device_error_503(self):
        e = DeviceError(ErrorCode.DEVICE_UNAVAILABLE, "device")
        assert e.http_status == 503

    def test_model_error_500(self):
        e = ModelError(ErrorCode.MODEL_LOAD_FAILED, "model")
        assert e.http_status == 500

    def test_capability_error_501(self):
        e = CapabilityError(ErrorCode.CAPABILITY_NOT_IMPLEMENTED, "cap")
        assert e.http_status == 501
