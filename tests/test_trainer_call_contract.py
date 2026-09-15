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


TRAINING_SCRIPT_PATHS = [
    SCRIPTS_DIR / "train_kan.py",
    SCRIPTS_DIR / "train_kan_top.py",
    SCRIPTS_DIR / "train_kan_qg.py",
    SCRIPTS_DIR / "train_rf.py",
]


def _calls_named(tree, name, *, with_attr=None):
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr == name and (with_attr is None or func.value.id == with_attr):
                calls.append(node)
        elif isinstance(func, ast.Name) and func.id == name:
            calls.append(node)
    return calls


def test_training_runs_use_run_seed_while_preprocessors_use_split_seed():
    for path in TRAINING_SCRIPT_PATHS:
        tree = ast.parse(path.read_text(), filename=str(path))
        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "set_seed"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "workspace"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Attribute)
            and call.args[0].attr == "seed"
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
        ), f"{path.name} should set the runtime RNG from args.seed"

        assert any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "load_and_preprocess_data"
            and any(
                kw.arg == "seed" and isinstance(kw.value, ast.Attribute) and kw.value.attr == "seed"
                for kw in call.keywords
            )
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
        ), f"{path.name} should select the data subset using args.seed"

    for path in (SCRIPTS_DIR.parent / "src" / "preprocessing" / "processor_top.py",
                 SCRIPTS_DIR.parent / "src" / "preprocessing" / "processor_qg.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        set_seed_calls = _calls_named(tree, "set_seed")
        assert any(
            len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "subset_split_seed"
            for call in set_seed_calls
        ), f"{path.name} should seed the canonical split with subset_split_seed"
        assert any(
            len(call.args) == 1
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "seed"
            for call in set_seed_calls
        ), f"{path.name} should restore the run-specific seed when selecting a subset"
