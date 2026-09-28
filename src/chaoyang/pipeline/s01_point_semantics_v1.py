"""Read-only projection provenance validation; never estimates geometry."""
POINTS = {'controller_tracked_origin', 'controller_housing_center',
          'anatomical_wrist_center', 'manus_virtual_root'}


def metadata_errors(config, installation_fit):
    errors = []
    flag = config.get('installation_fitted')
    if type(flag) is not bool:
        errors.append('INSTALLATION_FITTED_FLAG_UNKNOWN')
    sides = installation_fit.get('sides', [])
    if not isinstance(sides, list):
        return errors + ['INVALID_INSTALLATION_FIT_SCHEMA']
    if any(not isinstance(x,dict) or type(x.get('success')) is not bool for x in sides):
        errors.append('INVALID_INSTALLATION_SUCCESS_TYPE')
    successful = [x for x in sides if isinstance(x, dict) and x.get('success') is True]
    if successful and flag is False:
        errors.append('INSTALLATION_FIT_METADATA_CONFLICT')
    if flag is True and not successful:
        errors.append('INSTALLATION_FIT_SUCCESS_EVIDENCE_MISSING')
    return errors


def point_comparison(a, b, mapping=None):
    if a not in POINTS or b not in POINTS:
        raise ValueError('unknown point definition')
    if a == b:
        return 'SAME_POINT_DEFINITION_NOT_ACCURACY_PROOF'
    if not mapping or mapping.get('independently_verified') is not True:
        return 'NOT_COMPARABLE_AS_SAME_POINT'
    if mapping.get('from') != a or mapping.get('to') != b or not mapping.get('evidence_sha256'):
        return 'MAPPING_EVIDENCE_INCOMPLETE'
    return 'COMPARABLE_ONLY_THROUGH_REGISTERED_MAPPING'


def validate_contract(contract):
    if contract.get('schema_version') != 'S01_POINT_SEMANTICS_CONTRACT_V1':
        raise ValueError('wrong contract schema')
    if set(contract['point_definitions']) != POINTS:
        raise ValueError('missing point semantics')
    if contract['physical_calibration_verified'] is not False or contract['adoption'] != 'NOT_ADOPTED':
        raise ValueError('unjustified authority upgrade')
    if set(contract['sessions']) != {'097', '098', '101'}:
        raise ValueError('session scope escape')
    if contract['metadata_validation']['status'] != 'REJECT_LEGACY_CONFIG_AS_COHERENT_PROVENANCE':
        raise ValueError('known metadata conflict hidden')
    if 'INSTALLATION_FIT_METADATA_CONFLICT' not in contract['metadata_validation']['errors']:
        raise ValueError('known metadata conflict missing')
    return True
