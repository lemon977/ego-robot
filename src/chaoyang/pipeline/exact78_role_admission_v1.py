"""Fail early on complete required-role loss; this is not a semantic mask pass."""


def check_required_roles(mask_result, required_roles, frame_count):
    if not required_roles or len(set(required_roles)) != len(required_roles):
        raise ValueError('EXPLICIT_REQUIRED_MASK_ROLES_NEEDED')
    rows = mask_result.get('rows', [])
    if not frame_count or [r.get('frame_id') for r in rows] != list(range(frame_count)):
        raise ValueError('MASK_FRAME_SET_MISMATCH')
    present = {}
    for role in required_roles:
        counts = [r.get('instance_counts', {}).get(role) for r in rows]
        if any(type(c) is not int or c < 0 for c in counts):
            raise ValueError('ROLE_COUNTS_MISSING_OR_INVALID:' + role)
        present[role] = [i for i, count in enumerate(counts) if count > 0]
    missing = [role for role, frames in present.items() if not frames]
    return dict(status='HOLD_REQUIRED_ROLE_ABSENT' if missing else 'NONEMPTY_ONLY_NOT_QUALITY_ACCEPTED',
                missing_required_roles=missing, present_frames=present,
                visual_quality_pass=False, claim_limit='Only rejects all-frame role loss; presence alone cannot prove correct masks.')
