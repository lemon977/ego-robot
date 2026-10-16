"""Read-only mount adaptation for frozen V1 seed evidence, not segmentation."""
from contextlib import contextmanager
from pathlib import Path

from chaoyang.humanego_ict.project_mount_paths import resolve_input_path


@contextmanager
def legacy_source_evidence_paths(project_root):
    """Keep old evidence ownership spelling while preserving physical scope.

    The frozen installation validator compares its own __file__ and historical
    producer paths. Its metadata was signed under MNT. Adapt only these modules'
    Path bindings during Source; never rewrite a file, matrix or reference.
    All original producer/pixel/byte checks continue to execute.
    """
    import importlib
    from chaoyang.humanego_ict.project_mount_paths import PROJECT_ALIASES
    legacy = PROJECT_ALIASES[0]
    root = Path(project_root).resolve(strict=True)
    resolve_input_path(root, legacy)
    native_path = Path
    class LegacyEvidencePath(type(native_path())):
        def __new__(cls, *parts):
            value = native_path(*parts)
            if value.is_absolute() and any(value.is_relative_to(a) for a in PROJECT_ALIASES):
                value = resolve_input_path(value, legacy, exists=False)
            return super().__new__(cls, value)
    names = ('source', 'audit', 'installation_fixed_k', 'mask_probe')
    modules = [importlib.import_module('chaoyang.pick_place_1008.' + name) for name in names]
    saved = [(module, getattr(module, 'Path', None), getattr(module, 'PROJECT_ROOT', None)) for module in modules]
    try:
        for module, old_path, old_root in saved:
            if old_path is not None:
                module.Path = LegacyEvidencePath
            if old_root is not None:
                module.PROJECT_ROOT = legacy
        yield
    finally:
        for module, old_path, old_root in saved:
            if old_path is not None:
                module.Path = old_path
            if old_root is not None:
                module.PROJECT_ROOT = old_root


@contextmanager
def project_read_alias_audit(audit_module, project_root):
    """Scoped exact-reference read adapter for local imports in raw Source."""
    native_verify = audit_module._verify_exact_ref
    root = Path(project_root).resolve(strict=True)
    def verify(reference, project_only=False):
        if project_only:
            local = resolve_input_path(reference['path'], root)
            if not local.is_relative_to(root) or local == root:
                raise ValueError('AUTOMATION_EVIDENCE_OUTSIDE_PHYSICAL_PROJECT')
        return native_verify(reference, project_only=False)
    audit_module._verify_exact_ref = verify
    try:
        yield
    finally:
        audit_module._verify_exact_ref = native_verify


@contextmanager
def seed_evidence_alias_reader(seed_module, project_root):
    """Keep every original reference/SHA and all original pixel checks.

    The native audit checks lexical project scope. Check the identical physical
    project through the existing mount resolver, then delegate its exact-byte
    check. Only this seed module's read-only audit binding is scoped; the audit
    module, its output guards, all files and model behavior stay unchanged.
    """
    native = seed_module.audit
    root = Path(project_root).resolve(strict=True)

    class SeedReadAudit:
        def __getattr__(self, name):
            return getattr(native, name)

        def _verify_exact_ref(self, reference, project_only=False):
            if project_only:
                local = resolve_input_path(reference['path'], root)
                if not local.is_relative_to(root) or local == root:
                    raise ValueError('AUTOMATION_SEED_EVIDENCE_OUTSIDE_PHYSICAL_PROJECT')
            return native._verify_exact_ref(reference, project_only=False)

    seed_module.audit = SeedReadAudit()
    try:
        yield
    finally:
        seed_module.audit = native
