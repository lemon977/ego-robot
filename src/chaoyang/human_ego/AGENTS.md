# HumanEgo Visual Aux execution rules

1. Read the receipt-bound current status and the active Visual Aux task packet before any run.
2. Chips and Poker are independent checkpoint pairs. A blocked pair must not block the other pair.
3. Each pair contains exactly `HUMAN_RAW_RGB` and `ROBOTIZED_RGB`; session, source frame, split, seed, H50 label, and valid mask must match.
4. Training input must be `CAUSAL_TRAINING_INPUT`. A target frame may consume only donor frames with `donor_frame_id <= target_frame_id`.
5. The target is only `future_2d_xy` plus `future_2d_valid`, with `H=50`.
6. Never consume digital retarget `q_arm` or `q_hand` as real Robot action supervision.
7. Every checkpoint and visual trajectory sidecar must declare `control_ground_truth=false` and `physical_deployment_authorized=false`.
8. Run eligibility and epoch-0 before a full training attempt. Data-volume failure is `BLOCKED_DATA_VOLUME`, not an empty checkpoint.
9. Publish `best.pt`, `last.pt`, loss CSV/PNG, ADE/FDE/PCK, identity error, valid coverage, temporal smoothness, and code/config/data SHA.
10. One pair receives at most three runtime attempts and 12 GPU hours per branch. Quality failure is not retried automatically.
11. A single-seed result is an engineering comparison, not statistical significance or final policy authority.
