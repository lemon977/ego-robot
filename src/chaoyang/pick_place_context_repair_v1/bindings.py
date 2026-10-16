from __future__ import annotations
import copy
from chaoyang.pick_place_1008.pipeline import binding
from .approval import RECIPE,exact
def repair_binding(repo,stage,source_row,camera_ref,receipts,seed_ref,approval):
    upstream={} if stage=='source' else {'source':receipts['source']} if stage in ('motion','gpu') else {k:receipts[k] for k in ('source','motion','gpu')}
    value=binding(repo,stage,source_row,camera_ref,upstream,seed_ref if stage=='gpu' else None)
    if stage in ('source','motion'):return value
    if stage not in ('gpu','render'):raise ValueError('A_STAGE_DOMAIN')
    code={x['path']:x for x in value['code_and_dependencies']}
    for x in approval['exact_extra_runtime_code']:
        exact(x)
        if x['path'] in code:raise ValueError('A_CANNOT_REPLACE_BASELINE_DEPENDENCY')
        code[x['path']]=x
    value['code_and_dependencies']=sorted(code.values(),key=lambda x:x['path'])
    fields=approval['exact_extra_binding_fields'][stage]
    if set(fields)!=set(('repair_recipe',)):raise ValueError('A_ONLY_TYPED_REPAIR_FIELD')
    recipe=fields['repair_recipe']
    if recipe['recipe_id']!=RECIPE or recipe['contract']!=approval['contract_ref'] or recipe['session_id']!='042':
        raise ValueError('A_APPROVED_RECIPE_BINDING_IDENTITY')
    value.update(copy.deepcopy(fields))
    return value
