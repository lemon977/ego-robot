import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cutie_asset_pin_closes_local_files() -> None:
    pin = json.loads((ROOT / "assets/models/cutie/ASSET_PIN.json").read_text(encoding="utf-8"))
    assert pin["document_status"] == "REGISTERED_PIN_RESEARCH_CANARY_ONLY"
    for stem in ("model", "config", "weight", "resnet50", "resnet18", "license"):
        path = ROOT / pin["local"][f"{stem}_path"]
        data = path.read_bytes()
        assert len(data) == pin["local"][f"{stem}_bytes"]
        assert hashlib.sha256(data).hexdigest() == pin["local"][f"{stem}_sha256"]
    assert pin["load_smoke"]["status"] == "PASS_CPU_MODEL_AND_WEIGHT_LOAD"
    assert pin["execution"]["formal_production_allowed"] is False
    assert pin["license_boundary"]["commercial_use_authorized"] is False
