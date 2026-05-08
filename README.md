# Active Lesion Localization with DQN and PPO in LITS 

Public repository link: https://github.com/Dannyism11/Lession_Detection_in_RL

This repository contains the final notebooks for a deep reinforcement learning project on 2D active lesion localization in CT slices. The project follows a three-stage experimental flow:

1. **Baseline**: DQN-style active localization.
2. **Improvement 1**: Stabilized PPO localizer.
3. **Improvement 2**: PPO with multi-start confidence reporting.


## Group Information

| Field | Value |
|---|---|
| Name 1 | Muhammad Hamdan Sikandar |
| Roll 1 | 27100326 |
| Name 2 | Hadi Shahzad Khan |
| Roll 2 | 27100475 |

## Repository Contents

| File | Role |
|---|---|
| `Baseline-Final.ipynb` | Final baseline notebook using DQN-style active localization. |
| `Improvement_1.ipynb` | Improvement 1 notebook using a stabilized PPO training setup. |
| `Improvement_2.ipynb` | Improvement 2 notebook using multi-start confidence reporting on top of PPO. |

Generated checkpoints, metrics, and figures are written inside `/kaggle/working/...` during execution and are not required to be committed to GitHub.

## Dataset

The notebooks expect the LiTS dataset in the following Kaggle path:

```text
/kaggle/input/datasets/javariatahir123/lits17-liver-tumor-segmentation
```

Expected subdirectories:

```text
CT_Vol/CT_Vol
CT_Mask/CT_Mask
```


## Method Summary

### Baseline

The baseline uses an active localization setup where an agent starts with a bounding box and applies discrete actions to move, resize, or trigger a final lesion localization decision. It provides the reference performance for the project.

Main characteristics:

- DQN-style action-value policy.
- 2D lesion records built from 3D LiTS annotations.
- Maximum of 40 localization steps per case.
- Training capped at 300 records.
- Validation capped at 100 records.

### Improvement 1: Stabilized PPO Localizer

The first improvement replaces the DQN-style policy with PPO and adds a more stable training protocol. The goal is to improve the search behavior and increase the best lesion overlap reached during the localization trajectory.

Main characteristics:

- PPO actor-critic policy.
- Frozen VGG16-based visual feature extractor.
- Behavior cloning warmup.
- Action masking to avoid invalid or immediately reversing moves.
- Curriculum learning over lesion sizes.
- Clipped value loss and trigger auxiliary loss.
- Training capped at 300 records.
- Validation capped at 100 records.

### Improvement 2: Multi-start Confidence Reporting

The second improvement keeps the trained PPO policy but changes the reporting strategy. Instead of relying on a single start box, the same policy is evaluated from multiple deterministic valid start boxes. The final reported box is selected using learned stop-confidence information.

Main characteristics:

- Uses the same PPO localization policy.
- Runs evaluation from 7 deterministic start boxes.
- Selects the reported box using stop-confidence scoring.
- Does not use ground-truth labels at deployment time for selecting the final box.
- Improves final reported IoU and best trajectory IoU over the baseline.

## Experimental Configuration

| Setting | Baseline | Improvement 1 | Improvement 2 |
|---|---:|---:|---:|
| Max training records | 300 | 300 | 300 |
| Max validation records | 100 | 100 | 100 |
| Max steps per episode | 40 | 40 | 40 |
| PPO timesteps | N/A | 80,000 | 80,000 |
| Curriculum | No | Yes | Yes |
| Multi-start reporting | No | No | Yes |
| Number of start boxes | 1 | 1 | 7 |

## Final Reported Results

The following values are taken from the executed final notebook cells.

| Model | Split | n | Mean Final IoU | Mean Best IoU | Final@0.25 | Final@0.5 | Best@0.25 | Best@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | Validation | 100 | 0.1339 | 0.2026 | 0.2100 | 0.0100 | 0.3200 | 0.0500 |
| Baseline | Test | 100 | 0.0569 | 0.1287 | 0.0547 | 0.0078 | 0.2266 | 0.0469 |
| Improvement 1 | Validation | 100 | 0.1094 | 0.4639 | 0.1700 | 0.0500 | 0.7200 | 0.5800 |
| Improvement 1 | Test | 100 | 0.0444 | 0.1344 | 0.0600 | 0.0200 | 0.2200 | 0.0400 |
| Improvement 2 | Validation | 100 | 0.1578 | 0.5283 | 0.2300 | 0.0400 | 0.7800 | 0.6500 |
| Improvement 2 | Test | 100 | 0.0816 | 0.2342 | 0.1200 | 0.0500 | 0.3500 | 0.1600 |

Interpretation:

- Improvement 1 mainly improves the best localization reached during the trajectory.
- Improvement 2 improves both the final reported localization and the best reached localization.

## How to Run

The recommended environment is Kaggle Notebook with GPU enabled.

1. Upload or open the repository notebooks in Kaggle.
2. Attach the LiTS dataset at the expected path.
3. Run the notebooks in this order:

```text
Baseline-Final.ipynb
Improvement_1.ipynb
Improvement_2.ipynb
```

4. Check the final metric tables and trajectory visualizations at the bottom of each notebook.

For local editing only:

```bash
python -m pip install torch torchvision numpy scipy nibabel matplotlib pandas jupyter
jupyter lab .
```

## Output Files

The notebooks write outputs to:

| Notebook | Output Directory |
|---|---|
| Baseline | `/kaggle/working/caicedo_baseline_final` |
| Improvement 1 | `/kaggle/working/improvement_1` |
| Improvement 2 | `/kaggle/working/improvement_2` |

Typical generated files include:

- Best model checkpoint files, such as `improvement_2_best.pt`.
- Final metric CSV files.
- Training log CSV files.
- Qualitative rollout visualizations.
- JSON metric summaries.

