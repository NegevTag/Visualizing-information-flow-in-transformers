# PR: Add `@overload` signatures to `LanguageModel.__init__`

## Repo
`ndif-team/nnsight`

## File to change
`src/nnsight/modeling/language.py`

## Problem

`LanguageModel.__init__` currently has the signature:

```python
def __init__(
    self,
    *args,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    automodel: Type[AutoModel] = AutoModelForCausalLM,
    **kwargs,
) -> None:
```

Because the first positional argument is absorbed into `*args` with no type annotation,
Pylance (and mypy) cannot tell that the first argument is supposed to be a `str` (model name/path)
or a `torch.nn.Module` (custom model). Instead, Pylance tries to match the positional string
against the first *named* parameter (`tokenizer`), producing a false-positive error:

```
Argument of type "str" cannot be assigned to parameter "tokenizer"
of type "PreTrainedTokenizer | None"
```

The error surface is actually larger than it first appears. `LanguageModel.__init__`
passes `*args` down to `TransformersModel.__init__`, which passes it further to
`Envoy.__init__` — which has a fully typed first param `module: torch.nn.Module`.
Pylance tries to match the positional `str` against each typed named param it
encounters along the `*args` passthrough chain, producing multiple false-positive errors
(`tokenizer`, `automodel`, and potentially `module`).

This is a pure static-analysis problem — the code runs correctly at runtime.

## Fix

Add two `@overload` signatures that document the two valid call shapes already described
in the class docstring, without touching the real implementation body at all.

### Imports to add/update

Add `overload` to the `typing` import line (already imports `Optional`, `Type`, `Any`, etc.).
`torch` is already imported.

### Overloads to add (immediately before the real `__init__`)

```python
@overload
def __init__(
    self,
    model: str,
    *,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    automodel: Type[AutoModel] = ...,
    **kwargs: Any,
) -> None: ...

@overload
def __init__(
    self,
    model: "torch.nn.Module",
    *,
    tokenizer: PreTrainedTokenizer,
    automodel: Type[AutoModel] = ...,
    **kwargs: Any,
) -> None: ...
```

The real `__init__` body stays completely unchanged.

## Why this is correct

- Zero runtime impact — `@overload` variants are erased at runtime.
- The two overloads match exactly what the docstring already says:
  - `LanguageModel("meta-llama/Llama-3-8b", token=...)` — string path, tokenizer optional
  - `LanguageModel(my_module, tokenizer=my_tok)` — custom module, tokenizer required
- No new dependencies.

## Branch name suggestion
`feat/languagemodel-init-overloads`

## Commit message
```
typing: add @overload signatures to LanguageModel.__init__

Pylance/mypy cannot infer the type of the first positional arg when it
is absorbed by *args, causing a false-positive error when passing a
model name string. Two @overload variants document the two valid call
shapes (str path vs nn.Module) without changing runtime behaviour.
```
