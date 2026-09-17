from __future__ import annotations

import pytest

from chaoyang.ops.audit_kaihand_adapter_step_v1 import parse_step_text


STEP = """ISO-10303-21;
HEADER;
FILE_SCHEMA (('CONFIG_CONTROL_DESIGN'));
ENDSEC;
DATA;
#1=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));
#2=CARTESIAN_POINT('',(-1.,2.,3.));
#3=CARTESIAN_POINT('',(4.,8.,12.));
#4=CLOSED_SHELL('',());
#5=MANIFOLD_SOLID_BREP('adapter',#4);
#6=CYLINDRICAL_SURFACE('',#7,2.5);
#7=CIRCLE('',#8,5.0);
ENDSEC;
END-ISO-10303-21;
"""


def test_step_static_audit_records_units_solids_and_envelope() -> None:
    value = parse_step_text(STEP)
    assert value["length_unit"] == "millimetre"
    assert value["manifold_solid_brep_names"] == ["adapter"]
    assert value["declared_control_point_envelope"]["extent"] == [5.0, 6.0, 9.0]
    assert value["declared_cylindrical_radii"] == [2.5]


def test_incomplete_step_fails_closed() -> None:
    with pytest.raises(ValueError, match="complete"):
        parse_step_text(STEP.replace("END-ISO-10303-21;", ""))
