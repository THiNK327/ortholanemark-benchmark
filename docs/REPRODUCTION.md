# OrthoLaneMark reproduction interfaces

Start with the commands in the README. `tools/rebuild_tables.py` reads retained per-image scores only. It recomputes pooled presence counts/F1, joint presence/category accuracy, GRA image means, and wheelpath image means; localization summary fields are taken from saved reports and converted to millimeters using 4 mm/pixel. It checks all learning summary mean/std fields at absolute tolerance 1e-12. It writes a new JSON/CSV under `verification/` and does not overwrite archived reports.

`tools/evaluate_saved.py` evaluates saved numeric predictions against companion annotations using the unchanged evaluator. It does not build a model, select a threshold, or load checkpoint weights. Example for one retained learning run:

```sh
python tools/evaluate_saved.py --dataset-root ../dataset --manifest ../dataset/manifests/manifest_paper.json --predictions artifacts/learning/clrnet_faithful_seed0/predictions_test.npz --output work/clrnet_seed0_metrics.json
```

Prediction IDs and order must exactly match the manifest test split; the utility fails on disagreement. For raw confidence files, use the stored selected test predictions for this paper instead of introducing a new threshold search.

## Historical model interfaces

The evaluated source is retained for later reproduction. Nine files have added comment-only redistribution notices; [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) explains how their original bytes and hashes are recovered and verified. The nine methods are Canny-Hough, LSD, Steger ridge, U-Net, SCNN, UFLDv2, PolyLaneNet, LaneATT, and CLRNet. The method registry is `lcms_lane_benchmark.literature.build_method`. Classical constructors accept the effective keyword arguments in `configs/traditional/*_effective.json`. Learning constructors accept `ckpt_path` and device. Their `predict_image(grayscale_array)` output exposes `pred_x_L`, `pred_x_R`, continuous confidences, and native presence. Apply the stored final threshold for the paper; native adapter decisions and checkpoint-selection decisions are not interchangeable.

The original per-method `train.py` files and dataset readers remain present for traceability. They are not invoked by any preparation or offline check command. Existing defaults and historical scripts can refer to old run paths and should not be treated as the selected configurations; use `configs/learning/<method>_seed<seed>.json` and the run manifest. Original `scripts/reselect_eval.py` and `wheelpath_diagnostic.py` include historical inference/selection entry points; the README offline commands do not call those entry points. Their helper functions are imported solely to preserve evaluation behavior.

The LaneATT anchor-frequency tensor referenced in selected metadata is retained and hashed. No anchor recalibration was performed. Checkpoint files are in the separate sibling `checkpoints/` candidate. Use a hash-verified copy at the run manifest's expected weights path before any future model reproduction; redistribution remains subject to [the checkpoint rights review](CHECKPOINT_RIGHTS.md) and the authors' final release decision.
