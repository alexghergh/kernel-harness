"""Describe the solver-editable candidate file and its starter template.

Workspace materialization imports this so every problem starts from the same
candidate filename and a small free-form stub instead of a large scaffold with
protected markers and embedded source strings.
"""

from __future__ import annotations

from textwrap import dedent


CANDIDATE_FILENAME = "candidate_model_new.py"


def candidate_template() -> str:
    return dedent(
        '''
        import torch
        import torch.nn as nn
        from torch.utils.cpp_extension import load_inline

        # You may rewrite this file freely.
        # Requirements:
        # - define ModelNew(nn.Module)
        # - build a custom CUDA/C++ extension via load_inline or load
        # - write your own raw CUDA kernel and <<<...>>> launch path
        # - do not replace the computation with torch.matmul / torch.mm / torch.bmm / torch.einsum / Python @
        # - do not call high-level kernel frameworks, ATen helpers, or vendor-library wrappers


        class ModelNew(nn.Module):
            def __init__(self, *init_inputs):
                super().__init__()
                self._module = load_inline(
                    name="replace_me_candidate",
                    cpp_sources="#include <torch/extension.h>\\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}",
                    cuda_sources="#include <cuda_runtime.h>\\n__global__ void placeholder_kernel() {}",
                    functions=[],
                    with_cuda=True,
                    verbose=False,
                )

            def forward(self, *inputs):
                raise NotImplementedError(
                    "Replace this stub with a custom CUDA extension built via load_inline."
                )
        '''
    ).strip() + "\n"
