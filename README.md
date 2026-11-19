This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.”

# HabCam Orientation Bias Quantification Extraction Pipeline

**Facility:** WHOI Sundance Test Tank (October 26–27, 2026)  
**Project:** HabCam V4 Length-Measurement Bias Experiment  
**Organization:** NOAA NEFSC PEMAD ESB MDAT HabCam 

---

## Overview

This repository contains the end-to-end data extraction and quality-control pipeline for quantifying optical measurement bias in the HabCam V4 benthic stereo imaging system. Across $2.4\text{ hours}$ of active submerged tank operations, the pipeline processes **42,429** raw composite stereo images ($3872 \times 1216\text{ px}$ at $6\text{ Hz}$), extracts subpixel checkerboard target vectors across six known physical lengths ($80.0, 113.1, 160.0, 178.9, 226.3,$ and $240.0\text{ mm}$) using OpenCV, synchronizes frames with vehicle attitude and altitude telemetry ($\theta_{\text{pitch}}$, zero-referenced $\theta_{\text{roll,adj}} = \text{roll} - 137.0^\circ$, and `alt1`), and evaluates empirical measurement error under the operational field-of-view (`FOV()`) projection model.

After computer-vision screening, telemetry filtering, and geometric quality control, the pipeline outputs an analysis-ready dataset of **16,102,423 cleaned segments** across **33,386 stereo frames** (**63,040** camera detections) to train and validate Generalized Additive Mixed Models (`mgcv::bam()`) for operational length-bias correction.

---

## 1. Project Directory Structure

```text
HabCam_Project/
│
├── habcam_measurement_pipeline.R                # Master batch extraction, QC & audit pipeline
├── fit_habcam_bias_gamm.R                       # Downstream GAM/GAMM statistical modeling script
│
├── images/                                      # [USER PROVIDED] Raw 3872x1216 stereo imagery (42,429 pairs)
│
├── Extraction_Checkpoints/                      # Auto-generated chunk caches (safe to delete between runs)
│   ├── Previews/                                # Corner-detection diagnostic image overlays
│   ├── manifest_chunk_*.rds                     # Interim extraction logs
│   └── segments_chunk_*.rds                     # [GITIGNORE] Interim measurement data
│
├── Data/
│   ├── All_img_data.csv                         # Raw vehicle telemetry log (61,568 packets)
│   ├── calibrated_segments_raw.rds              # [GITHUB RELEASE v1.0-data] Pre-QC extracted segments
│   ├── calibrated_segments_clean.rds            # [GITHUB RELEASE v1.0-data] Final QC-filtered dataset (16,102,423 segments)
│   ├── frame_processing_manifest.rds            # Frame-by-frame CV screening & blur audit
│   ├── qc_audit_ledger.csv                      # QC tracking ledger
│   ├── habcam_bias_gamm_*.rds                   # Fitted Random-Effects GAMM models (Port/Starboard)
│   ├── habcam_bias_interaction_gam_*.rds        # Fitted Tilt x Sensor Interaction GAM models (Port/Starboard)
│   ├── habcam_bias_fixed_gam_*.rds              # Fitted Additive Fixed-Effects GAM models (Port/Starboard)
│   ├── habcam_validation_all_models.rds         # [GITHUB RELEASE v1.0-data] Multi-model OOS validation dataset
│   ├── habcam_validation_evaluated_clean.rds    # [GITHUB RELEASE v1.0-data] RE-GAMM OOS validation dataset
│   └── output_*.csv                             # Statistical and attitude-tier summaries
│
├── Tables/
│   ├── Table_1_Data_Cleaning_Summary.[csv|xlsx]
│   ├── Table_2_Cleaned_Data_by_Atitude.[csv|xlsx]
│   ├── Table_3_Model_Fit_Summary.[csv|xlsx]
│   ├── Table_4_OOS_Validation_Performance.[csv|xlsx]
│   ├── Table_5_Operational_Regime_Breakdown.[csv|xlsx]
│   └── Table_6_Size_Stratified_Millimeter_Accuracy.[csv|xlsx]
│
└── Figures/
    ├── Figure_1_Atiitude_Coverage_by_Altitude.png
    ├── Figure_2_Bias_Surface.png
    ├── Figure_3_Single_Axis_Transects.png
    ├── Figure_4_Orientation_Bias.png
    ├── Figure_5_Bias_Histograms.png
    ├── Figure_6_Fitted_Attitude_Altitude_Surfaces.png
    ├── Figure_7_Dynamic_Sensor_Plane_Distortion.png
    ├── Figure_8_Target_Orientation_Effect.png
    ├── Figure_9_OOS_Segment_Bias_Combo_Histogram.png
    ├── Figure_10_OOS_Segment_Bias_Attitude_Strata.png
    ├── Figure_11_GAMM_vs_GAM_Surface_Comparison_2m.png
    ├── Figure_12_Size_Stratified_Millimeter_Error.png
    └── Figure_13_Attitude_Regime_Millimeter_Error.png
```

---

## 2. Installation and Environment Setup

This pipeline uses a hybrid R–Python architecture. Telemetry synchronization, geometric quality control, and statistical summaries are executed in R, while subpixel checkerboard detection and image quality screening are offloaded to a localized Python virtual environment via `reticulate`.

### Step 1: R Packages
Install the required R packages:
```R
install.packages(c("reticulate", "dplyr", "tidyr", "readr", "tibble", "ggplot2", "scales", "mgcv"))
```

### Step 2: Python Virtual Environment
Create a dedicated Python virtual environment named `habcam_env` to prevent dependency conflicts with system Python installations. Run once in the R console:
```R
library(reticulate)
virtualenv_create("habcam_env")
virtualenv_install("habcam_env", packages = c("opencv-python", "numpy", "pandas"))
```

*Note: `habcam_measurement_pipeline.R` binds explicitly to this environment via `use_virtualenv("habcam_env", required = TRUE)`.*

### Step 3: Download Pre-Extracted Datasets (`v1.0-data` Release)
Because the serialized segment and validation datasets exceed GitHub's 100 MB per-file repository limit, they are hosted as `.zip` archives on the **[GitHub Releases (`v1.0-data`)](https://github.com/ncalabrese1/PEMAD-ESB-HabCam_Pitch_Roll/releases/tag/v1.0-data)** page rather than tracked directly in the Git tree.

* **If you only need the fitted correction models:** The compressed GAM and Random-Effects GAMM objects (`Data/habcam_bias_*.rds`) are already included directly in `Data/` and can be loaded immediately with `readRDS()`.
* **If you want to run `fit_habcam_bias_gamm.R` or inspect the segment-level data:** Download the required `.zip` archives from **[Releases (`v1.0-data`)](https://github.com/ncalabrese1/PEMAD-ESB-HabCam_Pitch_Roll/releases/tag/v1.0-data)** and extract the `.rds` files directly into your local `Data/` folder:

| Release Asset | Compressed Size | Extracted Path | Description |
| :--- | :---: | :--- | :--- |
| `calibrated_segments_clean.rds.zip` | 595.8 MB | `Data/calibrated_segments_clean.rds` | Primary QC-verified dataset (16,102,423 segments); **required** to run `fit_habcam_bias_gamm.R`. |
| `calibrated_segments_raw.rds.zip` | 616.6 MB | `Data/calibrated_segments_raw.rds` | Complete pre-QC segment pool prior to altimeter, attitude, and grid-rigidity filtering. |
| `habcam_validation_all_models.rds.zip` | 281.0 MB | `Data/habcam_validation_all_models.rds` | 20% stratified unseen holdout predictions across all 3 candidate models (Port & Starboard). |
| `habcam_validation_evaluated_clean.rds.zip` | 209.5 MB | `Data/habcam_validation_evaluated_clean.rds` | Out-of-sample holdout predictions and millimeter error evaluations for the RE-GAMMs. |

---

## 3. Data Dictionary & Quality-Control Pipeline

### Required Inputs
- **`images/`**: Uncompressed HabCam stereo composites ($3872 \times 1216\text{ px}$; `.jpg`, `.png`, or `.tif`). All 42,429 archived image pairs across 37 subfolders are indexed in `frame_processing_manifest.rds`. *(Not required if using the pre-extracted `.rds` datasets from Release `v1.0-data`.)*
- **`Data/All_img_data.csv`**: Continuous vehicle IMU and altimeter log. Required columns:
  - `imagename`: Filename matching the stereo image archive.
  - `pitch`: Vehicle pitch angle in degrees ($-22.7^\circ$ to $21.4^\circ$ in water).
  - `roll`: Raw vehicle roll angle in degrees (zero-referenced via $\theta_{\text{roll,adj}} = \text{roll} - 137.0^\circ$, spanning $-31.0^\circ$ to $33.2^\circ$).
  - `alt1`: Downward acoustic altitude in meters ($1.35$ to $3.79\text{ m}$ in water).

### Primary Output: `Data/calibrated_segments_clean.rds` *(via Release `v1.0-data`)*
Serialized dataframe containing **16,102,423** QC-verified target segments (**8,084,838** Port; **8,017,585** Starboard) across **33,386** stereo frames:
- `img_key`: Base frame identifier without file extension.
- `camera`: Camera channel (`"port"` or `"starboard"`).
- `segment_type`: Vector geometry class (1-square orthogonal, diagonal $\sqrt{2}$, 2-square orthogonal, knight-move $\sqrt{5}$, $2\times 2$ diagonal, and 3-square orthogonal).
- `pos_x`, `pos_y`: Normalized segment centroid on the rectified sensor plane ($0.0$ to $1.0$).
- `theta` / orientation angle: Segment orientation folded onto the $0^\circ\text{--}90^\circ$ domain.
- `pixel_len`: Subpixel Euclidean segment length measured on the rectified image matrix.
- `true_len_mm`: Known physical target length ($80.0, 113.1, 160.0, 178.9, 226.3,$ or $240.0\text{ mm}$).
- `fov_m2`, `mm_per_pixel`: Operational footprint area and nominal scalar resolution.
- `est_len_mm`: Uncorrected size estimate (`pixel_len` $\times$ `mm_per_pixel`).
- `bias_pct`: Uncorrected percentage length error ($(\text{est} - \text{true}) / \text{true} \times 100$).

---

## 4. Pipeline Execution & Troubleshooting

### Running the Extraction
Ensure your imagery and `Data/All_img_data.csv` are in place, then execute:
```R
source("habcam_measurement_pipeline.R")
```

### Running Downstream Modeling (`fit_habcam_bias_gamm.R`)
Once `Data/calibrated_segments_clean.rds` is downloaded from **[Release `v1.0-data`](https://github.com/ncalabrese1/PEMAD-ESB-HabCam_Pitch_Roll/releases/tag/v1.0-data)** (or generated via Stage 1), execute:
```R
source("fit_habcam_bias_gamm.R")
```

### Resume Capability & Memory Management
The extraction script processes imagery in chunks of 2,500 frames to prevent RAM saturation, writing intermediate caches to `Extraction_Checkpoints/`.
- **To resume an interrupted run:** Re-run `source("habcam_measurement_pipeline.R")`. The script reads `frame_processing_manifest.rds`, skips completed chunks, and resumes automatically.
- **To re-run from scratch after updating calibration parameters:** Delete the contents of `Extraction_Checkpoints/` and the output `.rds` files in `Data/` before running.

---

## 5. Mathematical Context & GAMM Bridge

### The Operational Baseline (`FOV()`)
The extraction pipeline replicates the legacy HabCam operational projection (16 mm nominal lens $\times$ 1.33 underwater refractive index = 0.02128 m focal length, idealized 3D Tait-Bryan rotations, and a uniform scalar `MMPIX` resolution). Near level flight ($\vert{}\theta_{\text{pitch}}\vert{} \le 4^\circ$, $\vert{}\theta_{\text{roll,adj}}\vert{} \le 4^\circ$, central sensor region), empirical bias is negligible (+0.52% ± 1.07% SD), whereas across the full $\pm 35^\circ$ attitude envelope uncorrected bias averages -0.88% ± 4.33% SD (Port) and -1.13% ± 4.50% SD (Starboard).

### The GAMM Correction Architecture (`fit_habcam_bias_gamm.R`)
Downstream modeling fits a Generalized Additive Mixed Model via `mgcv::bam()` to remove attitude-induced scaling distortion:

```math
\text{bias\textunderscore pct} \sim \beta_0 + \beta_1(\text{camera}) + te(\theta_{\text{pitch}}, \theta_{\text{roll,adj}}, Z_{\text{alt}}) + s(\theta, \text{bs} = \text{"cc"}) + s(u, v) + b_{\text{image}}
```

- $\beta_1(\text{camera})$: Parametric offset capturing the baseline mounting/calibration disparity between Port and Starboard pods.
- $te(\theta_{\text{pitch}}, \theta_{\text{roll,adj}}, Z_{\text{alt}})$: Trivariate tensor-product smooth modeling nonlinear compound pitch–roll–altitude scaling distortion.
- $s(\theta)$: Cyclic cubic spline accounting for residual target orientation effects across the folded $0^\circ\text{--}90^\circ$ domain.
- $s(u, v)$: 2D spatial thin-plate spline (`pos_x, pos_y`) capturing residual radial lens distortion across the sensor field.
- $b_{\text{image}}$: Frame-level random effect (`s(img_key, bs = "re")`) isolating transient acoustic or frame-specific variance.

---

## 6. Output Table & Figure Guide

### Tables (`Tables/`)
1. **`Table_1_Data_Cleaning_Summary`**: Step-by-step frame, camera-detection, and segment attrition accounting from 42,429 raw stereo pairs down to 33,386 clean frames.
2. **`Table_2_Cleaned_Data_by_Atitude`**: Summary of the operational sampling envelope stratified by the four target altitude tiers.
3. **`Table_3_Model_Fit_Summary`**: Parametric coefficients and smooth-term significance for the fitted GAMM models.
4. **`Table_4_OOS_Validation_Performance`**: Out-of-sample RMSE and bias improvements before vs. after correction.
5. **`Table_5_Operational_Regime_Breakdown`**: Error statistics binned by varying degrees of compound vehicle tilt.
6. **`Table_6_Size_Stratified_Millimeter_Accuracy`**: Absolute millimeter error distributions converted from percentage bias, stratified by physical scallop shell size classes.

### Figures (`Figures/`)
- **Figures 1–5 (Data Analysis)**: Highlight raw uncorrected bias. `Figure_1` maps the sampling envelope; `Figure_2` and `Figure_3` display the severe nonlinear divergence of bias at steep pitch/roll boundaries; `Figure_4` demonstrates that target orientation ($0^\circ\text{--}90^\circ$) has minimal directional impact (trendlines remain flat); `Figure_5` shows the overall raw error distributions.
- **Figures 6–8 (Model Terms)**: Visualize the fitted GAMM partial effects, including the primary 3D $te()$ attitude surface, the 2D $s(u,v)$ sensor-plane spatial distortion, and the cyclic $s(\theta)$ orientation effect.
- **Figures 9–13 (Validation)**: Demonstrate model performance on out-of-sample data, showing corrected error distributions centered tightly at zero (`Figure_9`, `Figure_10`) and summarizing absolute millimeter accuracy improvements across target size classes (`Figure_12`, `Figure_13`).
