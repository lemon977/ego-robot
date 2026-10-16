"""Change only the image propagation split length in an isolated frozen body."""
from __future__ import annotations
import ast,hashlib,inspect,textwrap
from types import MethodType


def make_full_pixel_painter(parent):
    from chaoyang.pick_place_1008.visual import ProtectedProPainter
    from .v1_clean_context import make_context_painter,LOCAL
    source=textwrap.dedent(inspect.getsource(ProtectedProPainter.inpaint))
    tree=ast.parse(source);hits=[]
    for node in ast.walk(tree):
        if (isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id=='prop_len' and isinstance(node.value,ast.Call)
            and isinstance(node.value.func,ast.Name) and node.value.func.id=='min'
            and len(node.value.args)==2 and isinstance(node.value.args[0],ast.Constant)
            and node.value.args[0].value==100 and isinstance(node.value.args[1],ast.Name)
            and node.value.args[1].id=='sub'):
            hits.append(node)
    if len(hits)!=1:raise ValueError('FULL_PIXEL_CONTEXT_EXACT_ONE_FROZEN_NODE_REQUIRED')
    hits[0].value=ast.Name(id='length',ctx=ast.Load());ast.fix_missing_locations(tree)
    painter,log=make_context_painter(parent,LOCAL)
    namespace=dict(painter.inpaint.__func__.__globals__)
    exec(compile(tree,'<candidate-full-native-pixel-context>','exec'),namespace)
    painter.inpaint=MethodType(namespace['inpaint'],painter)
    log.update(single_change='IMAGE_PIXEL_PROPAGATION_CONTEXT_ONLY',original_body_reused=False,
        original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_ast_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),edited_ast_assignments=1,
        pixel_context_policy='FULL_NATIVE',flow_chunking_unchanged=True,neural_reference_indices_unchanged=True,
        settings_dictionary_unchanged=True,original_globals_modified=False,vendor_modified=False)
    return painter,log
