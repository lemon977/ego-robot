PYTHON ?= python

.PHONY: check check-portable check-server handoff

# Safe default: synthetic CPU contracts only. No production or model dispatch.
check: check-portable

check-portable:
	$(PYTHON) -B scripts/maintenance/run_project_checks_v1.py --profile portable

# Server evidence is SHA-bound and intentionally unavailable in a clean clone.
check-server:
	$(PYTHON) -B scripts/maintenance/run_project_checks_v1.py --profile server

handoff:
	$(PYTHON) -B scripts/maintenance/project_handoff_check_v1.py
