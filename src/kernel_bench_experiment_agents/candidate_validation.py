"""Perform lightweight static validation on candidate source before the harness executes it.

The run and profile paths use this to reject obviously unsafe or malformed candidates early.
"""

from __future__ import annotations

import ast

from .candidate_contract import candidate_template, normalize_candidate_template


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
    "open",
    "torch.compile",
    "torch.set_float32_matmul_precision",
    "torch.mm",
    "torch.matmul",
    "torch.bmm",
    "torch.ops.load_library",
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
    """Validate that candidate source matches the solution-only KernelBench contract."""

    try:
        normalized_candidate = normalize_candidate_template(candidate_src)
        normalized_template = normalize_candidate_template(candidate_template())
    except ValueError as exc:
        raise CandidateValidationError(str(exc)) from exc
    if normalized_candidate != normalized_template:
        raise CandidateValidationError(
            "candidate_model_new.py must keep the fixed scaffold unchanged and edit only the marked blocks."
        )

    for marker in FORBIDDEN_STRING_MARKERS:
        if marker in candidate_src:
            raise CandidateValidationError(
                f"Candidate may not set or reference forbidden runtime marker {marker!r}."
            )
    lowered = candidate_src.lower()
    for marker in FORBIDDEN_VENDOR_MARKERS:
        if marker in lowered:
            raise CandidateValidationError(
                f"Vendor-library shortcut {marker!r} is forbidden in candidate_model_new.py."
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
                    f"Importing {alias.name!r} is forbidden in candidate_model_new.py."
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root in FORBIDDEN_IMPORT_ROOTS:
            raise CandidateValidationError(
                f"Importing from {module!r} is forbidden in candidate_model_new.py."
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == "ModelNew":
            self._saw_model_new = True
        elif node.name in FORBIDDEN_DEFINITION_NAMES:
            raise CandidateValidationError(
                f"Redefining {node.name!r} is forbidden. Only ModelNew belongs in the candidate file."
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
                f"Calling {call_name!r} is forbidden in candidate_model_new.py."
            )
        if call_name is not None:
            if call_name.startswith("torch.backends."):
                raise CandidateValidationError(
                    "Mutating or querying torch backend flags is forbidden in candidate_model_new.py."
                )
            if any(call_name.endswith(suffix) for suffix in FORBIDDEN_CALL_SUFFIXES):
                raise CandidateValidationError(
                    f"Calling {call_name!r} is forbidden in candidate_model_new.py."
                )

        for keyword in node.keywords:
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
                    "Environment-variable access is forbidden in candidate_model_new.py."
                )
            if name.startswith("torch.backends."):
                raise CandidateValidationError(
                    "Torch backend flag access is forbidden in candidate_model_new.py."
                )
        self.generic_visit(node)

    def finalize(self) -> None:
        if not self._saw_model_new:
            raise CandidateValidationError(
                "candidate_model_new.py must define a class named ModelNew."
            )
        if not self._saw_custom_loader:
            raise CandidateValidationError(
                "candidate_model_new.py must build a custom CUDA/C++ extension via load_inline or load."
            )

    def _validate_assignment_targets(self, targets: list[ast.expr]) -> None:
        for target in targets:
            name = _node_name(target)
            if name is None:
                continue
            if name.startswith("os.environ"):
                raise CandidateValidationError(
                    "Environment-variable mutation is forbidden in candidate_model_new.py."
                )
            if name.startswith("torch.backends."):
                raise CandidateValidationError(
                    "Torch backend flag mutation is forbidden in candidate_model_new.py."
                )
