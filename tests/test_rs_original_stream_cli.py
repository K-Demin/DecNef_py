"""Exercise actual CLI declarations and config construction without scanner/UI startup."""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import rs_pca_runtime
from rt_global_settings import add_registration_arguments


@pytest.mark.parametrize('launcher', ['rs_realtime_parallel', 'rs_pca_score_all_rois'])
@pytest.mark.parametrize('enabled', [False, True])
def test_original_comparison_stream_is_explicitly_opt_in_and_forwarded(launcher, enabled):
    source = Path(__file__).resolve().parents[1] / f'{launcher}.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    declarations = []
    for node in main.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'parser'
                                               for target in node.targets):
            declarations.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            function = node.value.func
            if ((isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
                 and function.value.id == 'parser' and function.attr == 'add_argument')
                    or isinstance(function, ast.Name) and function.id == 'add_registration_arguments'):
                declarations.append(node)
    namespace = {'argparse': argparse, 'Path': Path, 'pca_rt': rs_pca_runtime,
                 'add_registration_arguments': add_registration_arguments}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(source), 'exec'), namespace)
    arguments = ['--sub', '001', '--day', 'day', '--run', '1', '--incoming-root', 'incoming']
    if enabled:
        arguments.append('--enable-original-score')
    args = namespace['parser'].parse_args(arguments)
    config_calls = [node for node in ast.walk(main)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'RTSessionConfig']
    assert len(config_calls) == 1
    namespace.update(args=args, RTSessionConfig=SimpleNamespace, sub='001',
                     incoming_root=Path('incoming'), base_data=Path('data'),
                     pca_reference_image=None, enable_scoring=False)
    config = eval(compile(ast.Expression(config_calls[0]), str(source), 'eval'), namespace)
    assert config.enable_original_score is enabled
