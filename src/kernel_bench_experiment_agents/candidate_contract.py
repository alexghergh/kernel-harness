from __future__ import annotations

import re
from textwrap import dedent


CANDIDATE_FILENAME = "candidate_model_new.py"

CPP_BLOCK_START = "// BEGIN EDITABLE CPP SOURCE"
CPP_BLOCK_END = "// END EDITABLE CPP SOURCE"
CUDA_BLOCK_START = "// BEGIN EDITABLE CUDA SOURCE"
CUDA_BLOCK_END = "// END EDITABLE CUDA SOURCE"
INIT_BLOCK_START = "# BEGIN EDITABLE INIT"
INIT_BLOCK_END = "# END EDITABLE INIT"
FORWARD_BLOCK_START = "# BEGIN EDITABLE FORWARD"
FORWARD_BLOCK_END = "# END EDITABLE FORWARD"
EXTRA_CFLAGS_START = "# BEGIN EDITABLE EXTRA CFLAGS"
EXTRA_CFLAGS_END = "# END EDITABLE EXTRA CFLAGS"
EXTRA_CUDA_CFLAGS_START = "# BEGIN EDITABLE EXTRA CUDA CFLAGS"
EXTRA_CUDA_CFLAGS_END = "# END EDITABLE EXTRA CUDA CFLAGS"


EDITABLE_BLOCKS: tuple[tuple[str, str], ...] = (
    (CPP_BLOCK_START, CPP_BLOCK_END),
    (CUDA_BLOCK_START, CUDA_BLOCK_END),
    (INIT_BLOCK_START, INIT_BLOCK_END),
    (FORWARD_BLOCK_START, FORWARD_BLOCK_END),
    (EXTRA_CFLAGS_START, EXTRA_CFLAGS_END),
    (EXTRA_CUDA_CFLAGS_START, EXTRA_CUDA_CFLAGS_END),
)


def candidate_template() -> str:
    return dedent(
        f'''
        import hashlib
        import torch
        import torch.nn as nn
        from torch.utils.cpp_extension import load_inline

        # edit only the marked blocks in this file
        # do not add imports, helper functions, or classes outside the fixed scaffold
        # this experiment tests whether raw custom CUDA code can beat optimized PyTorch baselines
        # vendor-library wrappers, ATen compute helpers, and high-level kernel frameworks are forbidden

        CPP_SOURCE = r"""
        {CPP_BLOCK_START}
        #include <torch/extension.h>

        // write pybind-visible C++ bindings for your custom CUDA code here
        // keep this block limited to minimal extension glue and your own entrypoints

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
            // register your extension entrypoints here
        }}
        {CPP_BLOCK_END}
        """

        CUDA_SOURCE = r"""
        {CUDA_BLOCK_START}
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        // write raw CUDA kernels and launchers here
        // keep this block limited to your own raw CUDA implementation, not ATen helpers
        {CUDA_BLOCK_END}
        """

        EXTRA_CFLAGS = [
            {EXTRA_CFLAGS_START}
            "-O3",
            "-std=c++17",
            {EXTRA_CFLAGS_END}
        ]

        EXTRA_CUDA_CFLAGS = [
            {EXTRA_CUDA_CFLAGS_START}
            "-O3",
            "-std=c++17",
            {EXTRA_CUDA_CFLAGS_END}
        ]


        def _module_name() -> str:
            digest = hashlib.sha256((CPP_SOURCE + "\\n" + CUDA_SOURCE).encode("utf-8")).hexdigest()
            return "kbe_candidate_" + digest[:16]


        class ModelNew(nn.Module):
            _module = None

            def __init__(self, *init_inputs):
                super().__init__()
                if ModelNew._module is None:
                    ModelNew._module = load_inline(
                        name=_module_name(),
                        cpp_sources=CPP_SOURCE,
                        cuda_sources=CUDA_SOURCE,
                        functions=None,
                        extra_cflags=list(EXTRA_CFLAGS),
                        extra_cuda_cflags=list(EXTRA_CUDA_CFLAGS),
                        with_cuda=True,
                        verbose=False,
                    )
                {INIT_BLOCK_START}
                # initialize any module state you need here
                {INIT_BLOCK_END}

            def forward(self, *inputs):
                {FORWARD_BLOCK_START}
                raise NotImplementedError("Call your compiled extension entrypoint from forward.")
                {FORWARD_BLOCK_END}
        '''
    ).strip() + "\n"


def normalize_candidate_template(source: str) -> str:
    normalized = source
    for start_marker, end_marker in EDITABLE_BLOCKS:
        pattern = re.compile(
            re.escape(start_marker) + r".*?" + re.escape(end_marker),
            flags=re.DOTALL,
        )
        normalized, count = pattern.subn(
            f"{start_marker}\n<editable>\n{end_marker}",
            normalized,
            count=1,
        )
        if count != 1:
            raise ValueError(
                f"Expected exactly one editable block delimited by {start_marker!r} and {end_marker!r}."
            )
    return normalized
