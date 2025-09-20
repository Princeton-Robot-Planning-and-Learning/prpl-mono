# PRPL LLM Utils

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/prpl-llm-utils/actions/workflows/ci.yml/badge.svg)

LLM utilities from the Princeton Robot Planning and Learning group.

The main feature is the ability to save and load previous responses. There are also some code synthesis utilities.

## Usage Examples

### Cache to SQLite3 Database (Recommended)
```python
# Make sure OPENAI_API_KEY is set first.
from pathlib import Path
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
cache = SQLite3PretrainedLargeModelCache(Path(".llm_cache.db"))
llm = OpenAIModel("gpt-4o-mini", cache)
response = llm.query("What's a funny one liner?", hyperparameters={"temperature": 1.0})
# Querying again loads from cache.
assert llm.query("What's a funny one liner?", hyperparameters={"temperature": 1.0}).text == response.text
# Querying with different hyperparameters can change the response.
response2 = llm.query("What's a funny one liner?", hyperparameters={"temperature": 0.5})
# Inspect .llm_cache.db, for example, using https://sqliteviewer.app/.
```

### Cache to files
```python
from pathlib import Path
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import FilePretrainedLargeModelCache
cache = FilePretrainedLargeModelCache(Path(".llm_cache"))
llm = OpenAIModel("gpt-4o-mini", cache)
response = llm.query("What's a funny one liner?")
# Inspect the files in .llm_cache.
```

### Synthesize a Python function
```python
from pathlib import Path
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    SynthesizedPythonFunction,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.structs import Query, Response

# Set up the LLM.
cache = SQLite3PretrainedLargeModelCache(Path(".llm_cache.db"))
llm = OpenAIModel("gpt-4o-mini", cache)

# Give the target function a name.
function_name = "count_vowels"

# Optionally, check syntax and reprompt in case of errors.
reprompt_checks = [SyntaxRepromptCheck()]

# Optionally, define examples to reprompt in case of errors.
inputs = [("books",), ("stormy",), ("farm",)]
output_check_fns = [lambda x: x == 2, lambda x: x in (1, 2), lambda x: x == 1]
reprompt_checks.append(
    FunctionOutputRepromptCheck(
        function_name, inputs, output_check_fns, function_timeout=1.0
    ),
)

# Define the query.
query = Query(
    """Generate a Python function of the form

def count_vowels(s: str) -> int:
    # your code here

Return only the function; do not give example usages.
"""
)

# Run synthesis.
count_vowels = synthesize_python_function_with_llm(
    function_name,
    llm,
    query,
    reprompt_checks=reprompt_checks,
)

# Inspect the synthesized function.
print(count_vowels)

# Use the synthesized function.
assert count_vowels("woohoo") == 4
```

## Requirements

- Python 3.10+
- Tested on MacOS Monterey and Ubuntu 22.04

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`

## Check Installation

Run `./run_ci_checks.sh`. It should complete with all green successes in 5-10 seconds.

## Acknowledgements

This code descends from [predicators](https://github.com/Learning-and-Intelligent-Systems/predicators) and includes contributions from a number of people, including especially Nishanth Kumar.
