# Guards against scripts/train_kan*.py calling ClassicKANTrainer's stage
# methods on the *module* (e.g. `classic.train_kan_model(...)`) instead of on
# a `trainer` *instance* (`trainer.train_kan_model(...)`). classic_kan.py does
# not re-export these names at module scope, so a module-style call is a hard
# AttributeError at runtime -- this is exactly the bug that made
# scripts/train_kan_top.py and scripts/train_kan_qg.py silently stop mid-
# pipeline (and, for train_kan_qg.py, never train at all) before the fix.
import ast
import inspect
from pathlib import Path

from src.architectures.classic_kan import ClassicKANTrainer

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
SCRIPT_PATHS = [
    SCRIPTS_DIR / "train_kan.py",
    SCRIPTS_DIR / "train_kan_top.py",
    SCRIPTS_DIR / "train_kan_qg.py",
]

INSTANCE_METHOD_NAMES = {
    name
    for name, _ in inspect.getmembers(ClassicKANTrainer, predicate=inspect.isfunction)
    if not name.startswith("_")
}


def _module_alias(tree):
    """Local name bound to `import src.architectures.classic_kan as <alias>`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src.architectures.classic_kan":
                    return alias.asname or alias.name
    return None


def _trainer_var_names(tree, module_alias):
    """Local names assigned from `<module_alias>.ClassicKANTrainer(...)` or
    `ClassicKANTrainer(...)`."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        is_ctor_call = (
            isinstance(func, ast.Name) and func.id == "ClassicKANTrainer"
        ) or (
            isinstance(func, ast.Attribute)
            and func.attr == "ClassicKANTrainer"
            and isinstance(func.value, ast.Name)
            and func.value.id == module_alias
        )
        if is_ctor_call:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _illegal_module_style_calls(tree, module_alias):
    """`<module_alias>.<instance_method>(...)` call sites -- always a bug,
    since these names don't exist at module scope."""
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == module_alias
            and func.attr in INSTANCE_METHOD_NAMES
        ):
            offenders.append(f"{module_alias}.{func.attr}(...) at line {node.lineno}")
    return offenders


def _prune_call_kwargs(tree, trainer_vars):
    """Keyword-argument names used at each `<trainer>.prune_and_save_kan(...)`
    call site."""
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "prune_and_save_kan"
            and isinstance(func.value, ast.Name)
            and func.value.id in trainer_vars
        ):
            calls.append({kw.arg for kw in node.keywords if kw.arg is not None})
    return calls


def test_instance_method_set_is_nonempty():
    # Sanity check on the reflection itself, so a refactor that renames/removes
    # every method can't silently make every other assertion in this file vacuous.
    assert {"train_kan_model", "prune_and_save_kan", "retrain_pruned_kan",
            "simplify_and_save", "finetune_symbolic_model"} <= INSTANCE_METHOD_NAMES


def test_scripts_never_call_stage_methods_on_the_module():
    for path in SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        module_alias = _module_alias(tree)
        if module_alias is None:
            continue  # script imports ClassicKANTrainer directly, nothing to check here
        offenders = _illegal_module_style_calls(tree, module_alias)
        assert not offenders, (
            f"{path.name} calls ClassicKANTrainer instance methods on the module "
            f"object instead of a trainer instance: {offenders}"
        )


def test_scripts_instantiate_trainer_exactly_once():
    for path in SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        module_alias = _module_alias(tree)
        trainer_vars = _trainer_var_names(tree, module_alias)
        assert len(trainer_vars) == 1, (
            f"{path.name} should construct exactly one ClassicKANTrainer instance, "
            f"found assignments to: {trainer_vars or 'none'}"
        )


def test_prune_and_save_kan_call_sites_use_real_kwargs():
    real_params = set(inspect.signature(ClassicKANTrainer.prune_and_save_kan).parameters) - {"self"}
    for path in SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        module_alias = _module_alias(tree)
        trainer_vars = _trainer_var_names(tree, module_alias)
        for used_kwargs in _prune_call_kwargs(tree, trainer_vars):
            unknown = used_kwargs - real_params
            assert not unknown, (
                f"{path.name} calls prune_and_save_kan with unknown kwargs {unknown} "
                f"(valid: {sorted(real_params)})"
            )
