# lockstep-rvm

A Python implementation of a Pike/Cox-style lockstep regex virtual machine, with DFA state caching

## Installation

```Bash
python3.12 -m venv env
source env/bin/activate
pip install -e .
```

## Execution

```Bash
python3.12 -m lockstep_rvm.vm tests/test1.rasm bbbbbbbbb info cache
python3.12 -m lockstep_rvm.vm tests/test1.rasm bbbbbbbbb debug cache
```
