"""Perform lightweight static validation on candidate source before the harness executes it.

The run and profile paths use this to reject obviously unsafe or malformed candidates early.
"""

from __future__ import annotations

import ast

from kernel_bench_experiment_agents.kernelbench.candidate.contract import CANDIDATE_FILENAME


FORBIDDEN_IMPORT_ROOTS = {
    "ctypes",
    "httpx",
    "importlib",
    "inspect",
    "nvidia",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "triton",
    "urllib",
    "problem_reference",
    "reference",
}

FORBIDDEN_DEFINITION_NAMES = {
    "Model",
    "get_inputs",
    "get_init_inputs",
}

FORBIDDEN_CALL_NAMES = {
    "compile",
    "eval",
    "exec",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
    "torch.compile",
    "torch.set_float32_matmul_precision",
    "torch.mm",
    "torch.matmul",
    "torch.bmm",
    "torch.ops.load_library",
    "torch.cuda.Stream",
    "torch.cuda.current_stream",
    "torch.cuda.default_stream",
    "torch.cuda.stream",
}

FORBIDDEN_CALL_SUFFIXES = {
    ".mm",
    ".matmul",
    ".bmm",
    ".register_buffer",
}

REQUIRED_LOADER_NAMES = {
    "load",
    "load_inline",
    "torch.utils.cpp_extension.load",
    "torch.utils.cpp_extension.load_inline",
}

FORBIDDEN_STRING_MARKERS = {
    "TORCH_EXTENSIONS_DIR",
    "build_directory",
    "cudaEventCreate",
    "cudaEventRecord",
    "cudaStreamCreate",
    "cudaStreamSynchronize",
    "cudaStreamWaitEvent",
    "cudaStream_t",
    "torch.cuda.Stream",
    "torch.cuda.stream",
}

FORBIDDEN_REBIND_NAMES = {
    "load",
    "load_inline",
}

FORBIDDEN_VENDOR_MARKERS = {
    "aten/cuda/cudablas",
    "aten/cuda/cudacontext.h",
    "aten/native",
    "at::cuda::blas",
    "at::cuda::getcurrentcudastream",
    "at::cuda::getdefaultcudastream",
    "at::matmul",
    "at::native",
    "at::_ops::",
    "aten::matmul",
    "cublas",
    "cublaslt",
    "cutlass",
    "libcublas",
    "torch_cudablas_check",
}


class CandidateValidationError(ValueError):
    """Raised when a generated candidate violates the solver contract."""


def validate_candidate_source(candidate_src: str) -> None:
    """Validate that candidate source matches the free-form KernelBench contract."""

    for marker in FORBIDDEN_STRING_MARKERS:
        if marker in candidate_src:
            raise CandidateValidationError(
                f"{CANDIDATE_FILENAME} may not set or reference forbidden runtime marker {marker!r}."
            )
    lowered = candidate_src.lower()
    for marker in FORBIDDEN_VENDOR_MARKERS:
        if marker in lowered:
            raise CandidateValidationError(
                f"Vendor-library shortcut {marker!r} is forbidden in {CANDIDATE_FILENAME}."
            )

    try:
        tree = ast.parse(candidate_src)
    except SyntaxError as exc:
        raise CandidateValidationError(f"Candidate is not valid Python: {exc}") from exc

    validator = _CandidateValidator()
    validator.visit(tree)
    validator.finalize()


def _node_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _node_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


class _CandidateValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self._saw_model_new = False
        self._saw_custom_loader = False

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                raise CandidateValidationError(
                    f"Importing {alias.name!r} is forbidden in {CANDIDATE_FILENAME}."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root in FORBIDDEN_IMPORT_ROOTS:
            raise CandidateValidationError(
                f"Importing from {module!r} is forbidden in {CANDIDATE_FILENAME}."
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == "ModelNew":
            self._saw_model_new = True
        elif node.name in FORBIDDEN_DEFINITION_NAMES:
            raise CandidateValidationError(
                f"Redefining {node.name!r} is forbidden. Keep only ModelNew as the model entrypoint in {CANDIDATE_FILENAME}."
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in FORBIDDEN_DEFINITION_NAMES:
            raise CandidateValidationError(
                f"Redefining {node.name!r} is forbidden. Keep reference helpers in problem_reference.py."
            )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name in FORBIDDEN_DEFINITION_NAMES:
            raise CandidateValidationError(
                f"Redefining {node.name!r} is forbidden. Keep reference helpers in problem_reference.py."
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._validate_assignment_targets(node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._validate_assignment_targets([node.target])
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = _node_name(node.func)
        if call_name in REQUIRED_LOADER_NAMES or (
            call_name is not None
            and (call_name.endswith(".load") or call_name.endswith(".load_inline"))
        ):
            self._saw_custom_loader = True

        if call_name in FORBIDDEN_CALL_NAMES:
            raise CandidateValidationError(
                f"Calling {call_name!r} is forbidden in {CANDIDATE_FILENAME}."
            )
        if call_name is not None:
            if call_name.startswith("torch.backends."):
                raise CandidateValidationError(
                    f"Mutating or querying torch backend flags is forbidden in {CANDIDATE_FILENAME}."
                )
            if any(call_name.endswith(suffix) for suffix in FORBIDDEN_CALL_SUFFIXES):
                raise CandidateValidationError(
                    f"Calling {call_name!r} is forbidden in {CANDIDATE_FILENAME}."
                )

        for keyword in node.keywords:
            if keyword.arg == "build_directory":
                raise CandidateValidationError(
                    "Setting build_directory is forbidden; extension build paths are owned by the harness."
                )
            if keyword.arg == "out":
                raise CandidateValidationError(
                    "Using out= in candidate ops is forbidden because it can reuse output buffers across evaluations."
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = _node_name(node)
        if name is not None:
            if name.startswith("os.environ"):
                raise CandidateValidationError(
                    f"Environment-variable access is forbidden in {CANDIDATE_FILENAME}."
                )
            if name.startswith("torch.backends."):
                raise CandidateValidationError(
                    f"Torch backend flag access is forbidden in {CANDIDATE_FILENAME}."
                )
        self.generic_visit(node)

    def finalize(self) -> None:
        if not self._saw_model_new:
            raise CandidateValidationError(
                f"{CANDIDATE_FILENAME} must define a class named ModelNew."
            )
        if not self._saw_custom_loader:
            raise CandidateValidationError(
                f"{CANDIDATE_FILENAME} must build a custom CUDA/C++ extension via load_inline or load."
            )

    def _validate_assignment_targets(self, targets: list[ast.expr]) -> None:
        for target in targets:
            name = _node_name(target)
            if name is None:
                continue
            if name in FORBIDDEN_REBIND_NAMES:
                raise CandidateValidationError(
                    f"Rebinding {name!r} is forbidden in {CANDIDATE_FILENAME}. Keep the extension loader itself untouched."
                )
            if name.startswith("os.environ"):
                raise CandidateValidationError(
                    f"Environment-variable mutation is forbidden in {CANDIDATE_FILENAME}."
                )
            if name.startswith("torch.backends."):
                raise CandidateValidationError(
                    f"Torch backend flag mutation is forbidden in {CANDIDATE_FILENAME}."
                )
