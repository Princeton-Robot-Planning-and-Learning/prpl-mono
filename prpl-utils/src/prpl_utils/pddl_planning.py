"""Utilities for planning in PDDL problems."""

import logging
import os
import re
import subprocess
import tempfile
from typing import Optional

from pyperplan.planner import HEURISTICS, SEARCHES, search_plan


def run_pddl_planner(
    domain_str: str, problem_str: str, planner: str = "fd-sat"
) -> Optional[list[str]]:
    """Run a PDDL planner and return a list of ground operators, or None if no
    plan is found."""
    if planner == "fd-sat":
        return run_fastdownward_planning(domain_str, problem_str, alias="lama-first")
    if planner == "fd-opt":
        return run_fastdownward_planning(domain_str, problem_str, alias="seq-opt-lmcut")
    if planner == "pyperplan":
        return run_pyperplan_planning(domain_str, problem_str)
    raise NotImplementedError(f"Planner {planner} not implemented.")


def run_pyperplan_planning(
    domain_str: str,
    problem_str: str,
    heuristic: str = "hff",
    search: str = "gbf",
) -> Optional[list[str]]:
    """Find a plan with pyperplan."""
    search_fn = SEARCHES[search]
    heuristic_fn = HEURISTICS[heuristic]
    domain_file = tempfile.NamedTemporaryFile(delete=False).name
    problem_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(domain_file, "w", encoding="utf-8") as f:
        f.write(domain_str)
    with open(problem_file, "w", encoding="utf-8") as f:
        f.write(problem_str)
    # Quiet the pyperplan logging.
    logging.disable(logging.ERROR)
    pyperplan_plan = search_plan(
        domain_file,
        problem_file,
        search_fn,
        heuristic_fn,
    )
    logging.disable(logging.NOTSET)
    os.remove(domain_file)
    os.remove(problem_file)
    if pyperplan_plan is None:
        return None
    return [a.name for a in pyperplan_plan]


def run_fastdownward_planning(
    domain_str: str,
    problem_str: str,
    alias: Optional[str] = "lama-first",
    search: Optional[str] = None,
) -> Optional[list[str]]:
    """Find a plan with fast downward.

    Usage: Build and compile the Fast Downward planner, then set the environment
    variable FD_EXEC_PATH to point to the `downward` directory. For example:
    1) git clone https://github.com/aibasel/downward.git
    2) cd downward && ./build.py
    3) export FD_EXEC_PATH="<your path here>/downward"

    On MacOS, to use gtimeout:
    4) brew install coreutils
    """
    # Write out strings to files.
    domain_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(domain_file, "w", encoding="utf-8") as f:
        f.write(domain_str)
    problem_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(problem_file, "w", encoding="utf-8") as f:
        f.write(problem_str)
    # Specify either a search flag or an alias.
    assert (search is None) + (alias is None) == 1
    # The SAS file isn't actually used, but it's important that we give it a
    # name, because otherwise Fast Downward uses a fixed default name, which
    # will cause issues if you run multiple processes simultaneously.
    sas_file = tempfile.NamedTemporaryFile(delete=False).name
    # Run Fast Downward followed by cleanup. Capture the output.
    assert (
        "FD_EXEC_PATH" in os.environ
    ), "Please follow the instructions in the docstring of this method!"
    if alias is not None:
        alias_flag = f"--alias {alias}"
    else:
        alias_flag = ""
    if search is not None:
        search_flag = f"--search '{search}'"
    else:
        search_flag = ""
    fd_exec_path = os.environ["FD_EXEC_PATH"]
    exec_str = os.path.join(fd_exec_path, "fast-downward.py")
    cmd_str = (
        f'"{exec_str}" {alias_flag} '
        f"--sas-file {sas_file} "
        f'"{domain_file}" "{problem_file}" '
        f"{search_flag}"
    )
    output = subprocess.getoutput(cmd_str)
    cleanup_cmd_str = f"{exec_str} --cleanup"
    subprocess.getoutput(cleanup_cmd_str)
    # Extract the plan from the output, if one exists.
    if "Solution found!" not in output:
        return None
    if "Plan length: 0 step" in output:
        # Handle the special case where the plan is found to be trivial.
        return []
    plan_str = re.findall(r"(.+) \(\d+?\)", output)
    assert plan_str  # already handled empty plan case, so something went wrong
    plan = [f"({a})" for a in plan_str]
    return plan
