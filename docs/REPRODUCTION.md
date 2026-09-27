# Reproducing the benchmark results

The repository supports three workflows: reconstructing result tables from saved scores, evaluating saved predictions against the companion annotations, and using the original model interfaces. The first two do not require checkpoint weights or a GPU.

Run commands from the repository root. Install the offline dependencies with:

```sh
python -m pip install -r requirements-verification.txt
```

## Reconstruct the result tables

```sh
python tools/verify_integrity.py
python tools/verify_sources.py
python tools/rebuild_tables.py --output work/reconstructed_tables.json
```

The first two commands check package integrity, current source hashes, and retained attribution notices. `rebuild_tables.py` recomputes pooled presence counts/F1, joint and category presence accuracy, mean GRA, and mean wheelpath disagreement from saved per-image scores. Localization summaries come from the saved reports, with MAE converted using 4 mm/pixel. All learning-method mean and population-standard-deviation fields are checked against the saved summary at an absolute tolerance of 1e-12.

The command writes `work/reconstructed_tables.json` and `.csv`, leaving the packaged verification records and source reports unchanged. Use `--output <path>.json` to choose a different output location.

## Evaluate saved predictions

Place the companion dataset in `../dataset/`, or substitute its location in the commands below. Resolve the portable manifest for the original dataset readers:

```sh
python tools/prepare_manifest.py --dataset-root ../dataset --manifest ../dataset/manifests/manifest_paper.json --output work/manifest_resolved.json
```

The generated manifest contains paths for your machine. Shared dataset manifests retain relative paths.

Evaluate one saved CLRNet run against the annotations:

```sh
python tools/evaluate_saved.py --dataset-root ../dataset --manifest ../dataset/manifests/manifest_paper.json --predictions artifacts/learning/clrnet_faithful_seed0/predictions_test.npz --output work/clrnet_seed0_metrics.json
```

`evaluate_saved.py` uses the original metric helpers and the saved final presence decisions. It does not load a model or checkpoint or select a new confidence threshold. Prediction IDs and order must exactly match the manifest test split; the utility fails on disagreement. Use `predictions_test.npz` for the paper's selected decisions. See [the dataset interface](DATASET_AND_ANNOTATIONS.md) for the NPZ schema and [the protocol](PROTOCOL.md) for metric definitions.

## Check evaluation behavior

```sh
python tools/run_synthetic_checks.py
python tools/check_edge_cases.py
```

These commands check ground-truth handling, region rasterization, presence gating, and geometric edge cases using synthetic inputs. With the model dependencies installed, `python tools/run_synthetic_checks.py --decoders` also exercises the retained decoder stubs. Those stubs use synthetic arrays and do not run models on benchmark images.

## Model interfaces

The evaluated source includes Canny-Hough, LSD, Steger ridge, U-Net, SCNN, UFLDv2, PolyLaneNet, LaneATT, and CLRNet. The registry is `ortholanemark.literature.build_method`. Traditional constructors accept the effective keyword arguments in `configs/traditional/*_effective.json`. Learning constructors accept `ckpt_path` and device. Each method's `predict_image(grayscale_array)` output exposes `pred_x_L`, `pred_x_R`, continuous confidence scores, and native presence decisions. To reproduce the paper's decisions, apply the final threshold recorded for that run; native adapter decisions and checkpoint-selection decisions serve different stages of the protocol.

Use `configs/learning/<method>_seed<seed>.json` and [run_manifest.json](../provenance/run_manifest.json) for the selected configurations. Original per-method `train.py` files and dataset readers are included, but some default paths refer to the original experiments and require adjustment. The source scripts `reselect_eval.py` and `wheelpath_diagnostic.py` also contain inference or selection entry points. The offline commands above import their evaluation helpers without invoking those entry points.

The LaneATT anchor-frequency tensor used in the selected configuration is included at `ortholanemark/literature/laneatt_faithful/anchor_frequencies.pt` and hashed. Checkpoint weights are distributed separately in the companion `checkpoints/` package. Verify their hashes and copy them to the expected `weights/<run_id>/best_gra.pth` paths in the run manifest before using the learning models. See [checkpoint provenance and licensing](CHECKPOINT_RIGHTS.md), [method adaptations](METHOD_ADAPTATIONS.md), and [environment records](ENVIRONMENTS.md).

The package is distributed as `ortholanemark`, with matching import and resource paths. Nine source files also contain attribution comment prefixes. [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) explains the current-source verification and the historical evaluated hashes retained as audit references. Original source reconstruction records are held in a separate author archive. The offline checks establish saved-result reproducibility; they do not constitute a fresh training or inference validation of the installation specifications.
