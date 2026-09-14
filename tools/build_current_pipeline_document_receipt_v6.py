from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT / "docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md"
REGISTRY = ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"
RECEIPT = ROOT / "docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.receipt.json"


def ref(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    text = DOCUMENT.read_text(encoding="utf-8")
    required = ["HaWoR", "FoundationStereo", "SAM3.1", "Object6D", "ProPainter", "Contact", "Robot Visual", "HumanEgo"]
    missing = [token for token in required if token not in text]
    value = {
        "schema_version": "pipeline-technical-document-receipt-v3",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_CURRENT_ONLY_DOCUMENT" if not missing else "FAIL_MISSING_REQUIRED_SECTION",
        "document": ref(DOCUMENT),
        "baseline_registry_at_validation": ref(REGISTRY),
        "governance_revision": registry["governance_revision"],
        "required_tokens_missing": missing,
        "mutable_runtime_counts_copied": False,
        "claim_limit": "Technical-document validation only; no algorithm, contact, Robot or training authority promotion.",
    }
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{RECEIPT.name}.", suffix=".tmp", dir=RECEIPT.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, RECEIPT)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(value, ensure_ascii=False))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
