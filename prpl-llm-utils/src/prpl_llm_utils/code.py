"""Data structure and methods for code synthesis."""

import ast
import importlib
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import traceback
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.reprompting import (
    RepromptCheck,
    create_reprompt_from_error_message,
    query_with_reprompts,
)
from prpl_llm_utils.structs import Query, Response

# This speeds up the sandbox for code synthesis by a lot.
mp.set_start_method("fork")


class SynthesizedPythonFunctionRunError(Exception):
    """An exception raised during a call to SynthesizedPythonFunction.run()."""


@dataclass(frozen=True)
class SynthesizedPythonFunction:
    """Wraps a piece of Python code that contains a function with a given name.

    The typical flow is that an LLM outputs the code as a string, then
    we create one of these class instances, then invoke the function by
    calling run().

    If timeout is exceeded on run() call, a TimeoutError is raised.
    """

    function_name: str
    code_str: str
    timeout: float = 30.0  # max time in seconds that run() is allowed

    @cached_property
    def filepath(self) -> Path:
        """Get a file with the code string implemented in it."""
        filename = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".py").name)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.code_str)
        return filename

    def _load_module(self) -> Any:
        module_name = f"{self.filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, self.filepath)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        assert module is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def __call__(self, *input_args: Any) -> Any:
        """Alias for run()."""
        return self.run(*input_args)

    def __str__(self) -> str:
        return self.code_str

    def run(self, *input_args: Any) -> Any:
        """Run the function on an input (that will be unpacked)."""
        module = self._load_module()
        fn = getattr(module, self.function_name)

        # All of the code below is for handling the possibility of the function
        # call timing out.

        def _fn_in_place(result_dict: dict, *args: Any) -> Any:
            try:
                result_dict["fn_output"] = fn(*args)
            except BaseException as e:
                exception_msg = "\n".join(traceback.format_exception(e))
                error_msg = (
                    f"Given the input {input_args}, {self.function_name} raised an "
                    f"exception:\n{exception_msg}"
                )
                result_dict["error_msg"] = error_msg

        manager = mp.Manager()
        result_proxy_dict = manager.dict()
        p = mp.Process(
            target=_fn_in_place,
            args=(result_proxy_dict,) + tuple(input_args),
        )
        p.start()
        p.join(self.timeout)
        result_dict = dict(result_proxy_dict)
        # Timeout reached.
        if p.is_alive():
            # Treated like a KeyboardInterrupt.
            assert p.pid is not None
            os.kill(p.pid, signal.SIGINT)
            # Give it a few more seconds then kill for good.
            p.join(3)
            p.kill()
            raise SynthesizedPythonFunctionRunError("Possible infinite loop.")
        if "error_msg" in result_dict:
            raise SynthesizedPythonFunctionRunError(result_dict["error_msg"])
        return result_dict["fn_output"]


class SyntaxRepromptCheck(RepromptCheck):
    """Check the syntax of a response."""

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        python_code = parse_python_code_from_text(response.text)
        if python_code is None:
            error_msg = "No python code was found in the response."
        else:
            try:
                ast.parse(python_code)
                return None
            except SyntaxError as e:
                error_msg = "\n".join(traceback.format_exception(e))
        return create_reprompt_from_error_message(query, response, error_msg)


class FunctionOutputRepromptCheck(RepromptCheck):
    """Check whether the synthesized Python function produces valid output.

    It is up to the user of this class how "valid" is defined.
    """

    def __init__(
        self,
        function_name: str,
        inputs: list[Any],
        output_check_fns: list[Callable[[Any], bool]],
        function_timeout: float = 30.0,
    ) -> None:
        assert len(inputs) == len(
            output_check_fns
        ), "Expecting one check function per input"
        self._function_name = function_name
        self._inputs = inputs
        self._output_check_fns = output_check_fns
        self._function_timeout = function_timeout

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        python_code = parse_python_code_from_text(response.text)
        if python_code is None:
            raise RuntimeError("No python code found. Consider SyntaxRepromptCheck().")
        fn = SynthesizedPythonFunction(
            self._function_name, python_code, timeout=self._function_timeout
        )
        for fn_in, check_fn in zip(self._inputs, self._output_check_fns, strict=True):
            try:
                fn_out = fn.run(*fn_in)
            except SynthesizedPythonFunctionRunError as e:
                error_msg = e.args[0]
                return create_reprompt_from_error_message(query, response, error_msg)
            if not check_fn(fn_out):
                error_msg = (
                    f"Given the input {fn_in}, the output of {self._function_name} "
                    f"was {fn_out}, which is invalid"
                )
                return create_reprompt_from_error_message(query, response, error_msg)
        return None


def parse_python_code_from_text(text: str) -> str | None:
    """Parse Python code from text, assuming ```python tag."""
    # Parse out python code if it exists.
    python_code_prefix = "```python"
    if python_code_prefix in text:
        python_start = text.index(python_code_prefix)
        python_remainder = text[python_start + len(python_code_prefix) :]
        if "```" in python_remainder:
            python_end = python_remainder.index("```")
        else:
            python_end = len(python_remainder)
        python_response = python_remainder[:python_end]
        return python_response
    return None


def synthesize_python_function_with_llm(
    function_name: str,
    model: PretrainedLargeModel,
    query: Query,
    reprompt_checks: list[RepromptCheck] | None = None,
    max_attempts: int = 5,
    function_timeout: float = 30.0,
) -> SynthesizedPythonFunction:
    """Synthesize a Python function with an LLM."""
    if reprompt_checks is None:
        reprompt_checks = []
    response = query_with_reprompts(model, query, reprompt_checks, max_attempts)
    python_code = parse_python_code_from_text(response.text)
    if python_code is None:
        raise RuntimeError("No python code found. Consider SyntaxRepromptCheck().")
    return SynthesizedPythonFunction(
        function_name, python_code, timeout=function_timeout
    )
