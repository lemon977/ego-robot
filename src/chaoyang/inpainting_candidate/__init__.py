"""Candidate-only generic inpainting components; never the default V1 producer.

Importing this package does not load models, register a Task, or run algorithms.
Real execution requires a separately frozen algorithm contract and finite packet.
"""

CANDIDATE_ONLY = True
PARENT_BASELINE = "INPAINTING_RAW_TO_ROBOT_V1"
