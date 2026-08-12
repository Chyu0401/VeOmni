# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-runnable contract tests for gated-delta-rule varlen metadata."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch


GENERATED_NPU_MODELS = (
    "veomni/models/transformers/qwen3_5/generated/patched_modeling_qwen3_5_npu.py",
    "veomni/models/transformers/qwen3_5_moe/generated/patched_modeling_qwen3_5_moe_npu.py",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cu_seqlens(*lengths: int) -> torch.LongTensor:
    """Build a cumulative-sequence-lengths tensor from per-sequence lengths."""
    return torch.cumsum(torch.tensor((0,) + lengths, dtype=torch.long), dim=0)


# ---------------------------------------------------------------------------
# precompute_varlen_metadata contract
# ---------------------------------------------------------------------------


class TestPrecomputeVarlenMetadataContract:
    """Verify the backend-independent varlen metadata contract."""

    @pytest.fixture(scope="class")
    def module(self):
        from veomni.ops.kernels.gated_delta_rule import varlen_metadata as m

        return m

    def test_file_cold_import_does_not_load_ascend_dependencies(self) -> None:
        """The helper module itself must remain independent of Ascend backends."""
        module_path = Path(__file__).parents[2] / "veomni/ops/kernels/gated_delta_rule/varlen_metadata.py"
        script = textwrap.dedent(
            f"""
            import builtins
            import importlib.util

            original_import = builtins.__import__

            def reject_ascend_imports(name, *args, **kwargs):
                if name in {{"fla_npu", "torch_npu"}} or name.startswith("triton"):
                    raise AssertionError(f"unexpected Ascend dependency import: {{name}}")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = reject_ascend_imports
            spec = importlib.util.spec_from_file_location("varlen_metadata_cold_import", {str(module_path)!r})
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert callable(module.precompute_varlen_metadata)
            """
        )
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)

    def test_package_cold_import_without_fla_npu(self) -> None:
        """The generated model's full import path must work without the optional AscendC package."""
        script = textwrap.dedent(
            """
            import importlib
            import sys

            sys.modules["fla_npu"] = None
            module = importlib.import_module("veomni.ops.kernels.gated_delta_rule.varlen_metadata")
            assert callable(module.precompute_varlen_metadata)
            """
        )
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)

    @pytest.mark.parametrize(
        "lengths,chunk_size,num_heads",
        [
            ([128, 256], 64, 4),
            ([64, 128, 32], 64, 8),
        ],
    )
    def test_metadata_matches_ensure(
        self,
        module,
        lengths: list[int],
        chunk_size: int,
        num_heads: int,
    ) -> None:
        """Precomputed tensors and lists must match an independent Python reference."""
        cu_seqlens = _make_cu_seqlens(*lengths)
        cu_seqlens_list, chunk_indices, chunk_indices_list = module.precompute_varlen_metadata(
            cu_seqlens=cu_seqlens,
            num_heads=num_heads,
            chunk_size=chunk_size,
        )

        assert cu_seqlens_list == cu_seqlens.tolist()
        cumsum_block_size = 1 << (((1 << 17) // (num_heads * chunk_size)) - 1).bit_length()
        expected_sizes = {16, 32, 64, 128, 608 * 2, chunk_size, cumsum_block_size}
        assert set(chunk_indices) == {str(size) for size in expected_sizes}
        assert set(chunk_indices_list) == {str(size) for size in expected_sizes}

        for size in expected_sizes:
            expected: list[int] = []
            for seq_idx, length in enumerate(lengths):
                for index in range((length + size - 1) // size):
                    expected.extend([seq_idx, index])

            assert chunk_indices_list[str(size)] == expected
            assert chunk_indices[str(size)].tolist() == [expected[i : i + 2] for i in range(0, len(expected), 2)]


@pytest.mark.parametrize("relative_path", GENERATED_NPU_MODELS)
def test_generated_model_only_passes_precomputed_metadata_to_ascendc(relative_path: str) -> None:
    """Backend-specific metadata must not leak into third-party kernel signatures."""
    tree = ast.parse((Path(__file__).parents[2] / relative_path).read_text())
    forbidden_keywords = {"cu_seqlens_list", "chunk_indices", "chunk_indices_list"}

    kernel_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "chunk_gated_delta_rule"
    ]
    assert kernel_calls
    for call in kernel_calls:
        assert forbidden_keywords.isdisjoint(keyword.arg for keyword in call.keywords)
        assert any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "precomputed_metadata_kwargs"
            for keyword in call.keywords
        )

    ascendc_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "npu_ascendc"
            for expression in ast.walk(node.test)
            if isinstance(expression, ast.Compare)
            for comparator in expression.comparators
        )
    ]
    assert any(
        isinstance(descendant, ast.Subscript)
        and isinstance(descendant.value, ast.Name)
        and descendant.value.id == "precomputed_metadata_kwargs"
        and isinstance(descendant.ctx, ast.Store)
        for guard in ascendc_guards
        for statement in guard.body
        for descendant in ast.walk(statement)
    )
    assert any(
        isinstance(descendant, ast.Call)
        and isinstance(descendant.func, ast.Name)
        and descendant.func.id == "precompute_varlen_metadata"
        for guard in ascendc_guards
        for statement in guard.body
        for descendant in ast.walk(statement)
    )
