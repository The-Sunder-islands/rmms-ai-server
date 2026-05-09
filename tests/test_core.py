from __future__ import annotations

import json
import os
import tempfile
import pytest
from pathlib import Path

from rmms_ai_server.core.pipeline_defs import resolve_pipeline, PRESET_PIPELINES
from rmms_ai_server.core.cache_manager import CacheManager
from rmms_ai_server.models.protocol import PipelineStep, StepInput
from rmms_ai_server.models.errors import (
    ErrorCode, RMMSAIError, InputError, AuthError, QuotaError,
)


class TestResolvePipeline:
    def test_resolve_split_preset(self):
        steps = resolve_pipeline(None, "split")
        assert len(steps) == 1
        assert steps[0].capability == "split"

    def test_resolve_full_preset(self):
        steps = resolve_pipeline(None, "full")
        assert len(steps) == 3
        assert steps[0].capability == "split"
        assert steps[1].capability == "midi"
        assert steps[2].capability == "detect"

    def test_resolve_split_midi_preset_input(self):
        steps = resolve_pipeline(None, "split+midi")
        assert len(steps) == 2
        assert steps[1].input is not None
        assert steps[1].input.from_step == 0

    def test_resolve_unknown_preset_falls_back(self):
        steps = resolve_pipeline(None, "unknown")
        assert len(steps) == 1
        assert steps[0].capability == "split"

    def test_resolve_from_dict_list(self):
        steps = resolve_pipeline([{"capability": "split"}, {"capability": "midi"}], None)
        assert len(steps) == 2
        assert isinstance(steps[0], PipelineStep)

    def test_resolve_from_type_alias(self):
        steps = resolve_pipeline([{"type": "split"}, {"type": "midi"}], None)
        assert len(steps) == 2
        assert steps[0].capability == "split"
        assert steps[1].capability == "midi"

    def test_resolve_from_steps_dict(self):
        steps = resolve_pipeline({"steps": [{"type": "split"}]}, None)
        assert len(steps) == 1
        assert steps[0].capability == "split"

    def test_resolve_none_returns_split(self):
        steps = resolve_pipeline(None, None)
        assert len(steps) == 1
        assert steps[0].capability == "split"

    def test_pipeline_step_model_field(self):
        steps = resolve_pipeline([{"type": "split", "model": "htdemucs"}], None)
        assert steps[0].model == "htdemucs"

    def test_pipeline_step_model_params(self):
        steps = resolve_pipeline([{"type": "midi", "model_params": {"precision": "float32"}}], None)
        assert steps[0].model_params == {"precision": "float32"}

    def test_pipeline_step_params(self):
        steps = resolve_pipeline([{"type": "split", "params": {"shifts": 5}}], None)
        assert steps[0].params == {"shifts": 5}

    def test_preset_pipelines_have_correct_structure(self):
        for name, steps_def in PRESET_PIPELINES.items():
            assert isinstance(steps_def, list)
            for step_def in steps_def:
                assert "capability" in step_def


class TestCacheManager:
    def test_put_and_get(self):
        cm = CacheManager()
        h = "abc123"
        cm.put(h, {"pipeline": "split"}, {"task_id": "task-1"})
        result = cm.get(h, {"pipeline": "split"})
        assert result is not None
        assert result["task_id"] == "task-1"

    def test_different_params_different_keys(self):
        cm = CacheManager()
        h = "abc123"
        cm.put(h, {"pipeline": "split"}, {"task_id": "task-1"})
        result = cm.get(h, {"pipeline": "midi"})
        assert result is None

    def test_different_hash_different_keys(self):
        cm = CacheManager()
        cm.put("aaa", {"x": 1}, {"v": 1})
        cm.put("bbb", {"x": 1}, {"v": 2})
        assert cm.get("aaa", {"x": 1})["v"] == 1
        assert cm.get("bbb", {"x": 1})["v"] == 2

    def test_invalidate(self):
        cm = CacheManager()
        cm.put("key", {"params": "x"}, {"result": "y"})
        assert cm.get("key", {"params": "x"}) is not None
        cm.invalidate("key", {"params": "x"})
        assert cm.get("key", {"params": "x"}) is None

    def test_clear(self):
        cm = CacheManager()
        cm.put("a", {}, {"x": 1})
        cm.put("b", {}, {"x": 2})
        cm.clear()
        assert cm.get("a", {}) is None
        assert cm.get("b", {}) is None

    def test_put_includes_created_at(self):
        cm = CacheManager()
        cm.put("key", {}, {"task_id": "t1"})
        result = cm.get("key", {})
        assert "created_at" in result
        assert result["created_at"] != ""

    def test_compute_file_hash(self):
        cm = CacheManager()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"test audio content")
            f.flush()
            fpath = f.name
        try:
            h1 = cm.compute_file_hash(fpath)
            h2 = cm.compute_file_hash(fpath)
            assert h1 == h2
            assert len(h1) == 32
            assert all(c in "0123456789abcdef" for c in h1)
        finally:
            os.unlink(fpath)

    def test_compute_file_hash_different_content(self):
        cm = CacheManager()
        fpath1 = None
        fpath2 = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"content a")
                fpath1 = f.name
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(b"content b")
                fpath2 = f.name
            assert cm.compute_file_hash(fpath1) != cm.compute_file_hash(fpath2)
        finally:
            if fpath1:
                os.unlink(fpath1)
            if fpath2:
                os.unlink(fpath2)


class TestPipelineStepModelDump:
    def test_json_alias_type(self):
        step = PipelineStep(capability="split", model="htdemucs")
        d = step.model_dump(by_alias=True)
        assert d["type"] == "split"
        assert "capability" not in d

    def test_roundtrip_via_json(self):
        step1 = PipelineStep(capability="split", model="htdemucs", params={"shifts": 1})
        j = step1.model_dump_json(by_alias=True)
        step2 = PipelineStep.model_validate_json(j)
        assert step2.capability == "split"
        assert step2.model == "htdemucs"

    def test_model_dump_includes_model(self):
        step = PipelineStep(capability="midi", model="basic-pitch")
        d = step.model_dump(by_alias=True)
        assert d["model"] == "basic-pitch"

    def test_model_dump_includes_null_input(self):
        step = PipelineStep(capability="split")
        d = step.model_dump(by_alias=True)
        assert d.get("input") is None

    def test_model_dump_includes_model_params(self):
        step = PipelineStep(capability="detect", model_params={"foo": "bar"})
        d = step.model_dump(by_alias=True)
        assert d["model_params"] == {"foo": "bar"}


class TestErrorRMMSAI:
    def test_to_dict_structure(self):
        e = RMMSAIError(ErrorCode.SERVER_ERROR, "msg", details={"k": "v"})
        d = e.to_dict()
        assert d == {"code": "SERVER_ERROR", "message": "msg", "details": {"k": "v"}}

    def test_input_error_details(self):
        e = InputError(ErrorCode.INPUT_TOO_LARGE, "too large", details={"size": 100})
        assert e.http_status == 400
        assert e.details["size"] == 100

    def test_quota_error_with_retry(self):
        e = QuotaError(
            ErrorCode.QUOTA_EXCEEDED, "queue full",
            details={"retry_after": 30, "queue_position": 5},
        )
        d = e.to_dict()
        assert d["details"]["retry_after"] == 30
        assert d["details"]["queue_position"] == 5

    def test_error_code_values_match_protocol(self):
        assert ErrorCode.INPUT_FORMAT_UNSUPPORTED.value == "INPUT_FORMAT_UNSUPPORTED"
        assert ErrorCode.INPUT_TOO_LARGE.value == "INPUT_TOO_LARGE"
        assert ErrorCode.MODEL_LOAD_FAILED.value == "MODEL_LOAD_FAILED"
        assert ErrorCode.DEVICE_UNAVAILABLE.value == "DEVICE_UNAVAILABLE"
        assert ErrorCode.DEVICE_OOM.value == "DEVICE_OOM"
        assert ErrorCode.SERVER_BUSY.value == "SERVER_BUSY"
        assert ErrorCode.QUOTA_EXCEEDED.value == "QUOTA_EXCEEDED"
        assert ErrorCode.PIPELINE_INVALID.value == "PIPELINE_INVALID"
        assert ErrorCode.CAPABILITY_NOT_IMPLEMENTED.value == "CAPABILITY_NOT_IMPLEMENTED"
        assert ErrorCode.AUTH_REQUIRED.value == "AUTH_REQUIRED"
        assert ErrorCode.AUTH_INVALID.value == "AUTH_INVALID"
