# ==============================================================================
# Script: habcam_measurement_pipeline.R
# Project: HabCam Attitude Bias Quantification Extraction Pipeline
#
# Purpose:
#   Stage 1 production pipeline (Extraction -> Cleaning -> Figures):
#     1. Split stereo composites (3872x1216) into Port and Starboard channels.
#     2. Reject blurry frames via Laplacian variance.
#     3. Detect rigid interior checkerboards via CLAHE + OpenCV subpixel matching.
#     4. Checkpoint every 250 frames with live visual QAQC previews.
#     5. Join vehicle telemetry (pitch, roll, altitude) and compute FOV scale (From Hans).
#     6. Drop frames with missing telemetry or heights outside the underwater tank range (1.35–3.80 m).
#     7. Drop frames within 5 seconds of out-of-water or missing altimeter pings (surface splash buffer).
#     8. Drop frames with sudden depth bounces (>0.12 m off local median / >0.35 m/s vertical speed).
#     9. Drop frames where pitch or adjusted roll exceeds 35 degrees.
#    10. Drop frames with <100 extracted segments (edge-clipped checkerboards).
#    11. Filter out video glitches (scanline tears) and warped grids by verifying that all checkerboard
#        squares look uniform in size and that their diagonals match true geometric proportions.
#    12. Export the analysis-ready dataset (Data/calibrated_segments_clean.rds).
#    13. Run measurement accuracy audits and generate figures strictly from clean data.
# Environment:
#   Managed via renv. Requires Python virtual environment 'habcam_env'.
# ==============================================================================

suppressPackageStartupMessages({
  library(reticulate)  # Python-R interface
  library(dplyr)       # Data manipulation
  library(tidyr)       # Data reshaping
  library(readr)       # Fast I/O
  library(tibble)      # Enhanced data frames
  library(tools)       # File path utilities
  library(ggplot2)     # Visualization
  library(scales)      # Plot scaling
  library(png)         # Live preview display
})

use_virtualenv("habcam_env", required = TRUE)

# ==============================================================================
# SECTION 1: Configuration & QC Thresholds
# ==============================================================================

# Input Paths
IMAGE_DIR          <- "//155.206.139.12/data/habcam/proc/Images/2026/202610"
TELEMETRY_CSV      <- file.path("Data", "All_img_data.csv")
if (!file.exists(TELEMETRY_CSV)) TELEMETRY_CSV <- "All_img_data.csv"
MANUAL_EXCLUSIONS  <- file.path("Data", "Manual_Frame_Exclusions.csv") # Optional

# Output Paths
DATA_DIR           <- "Data"
FIGURE_DIR         <- "Figures"
CHECKPOINT_DIR     <- "Extraction_Checkpoints"
PREVIEW_DIR        <- file.path(CHECKPOINT_DIR, "Previews")

dir.create(DATA_DIR, showWarnings = FALSE)
dir.create(FIGURE_DIR, showWarnings = FALSE)
dir.create(CHECKPOINT_DIR, showWarnings = FALSE)
dir.create(PREVIEW_DIR, showWarnings = FALSE, recursive = TRUE)

RAW_SEGMENTS_RDS   <- file.path(DATA_DIR, "calibrated_segments_raw.rds")
CLEAN_SEGMENTS_RDS <- file.path(DATA_DIR, "calibrated_segments_clean.rds")
FINAL_MANIFEST_RDS <- file.path(DATA_DIR, "frame_processing_manifest.rds")
QC_LEDGER_CSV      <- file.path(DATA_DIR, "qc_audit_ledger.csv")

# Physical Constants & Extraction Parameters
SQUARE_SIZE_MM     <- 80.0   # Physical checkerboard square size (mm)
ROLL_NEUTRAL_DEG   <- 137.0  # Neutral roll trim baseline (deg)
BLUR_THRESHOLD     <- 45.0   # Laplacian variance sharpness cutoff
CHUNK_SIZE         <- 250    # Frames per checkpoint (~30-45 sec per batch)
NUM_WORKERS        <- min(8, max(1, parallel::detectCores() - 1))

# Objective Physical, Altimeter & Geometric QC Thresholds (Applied in Sections 3 & 6)
ALT_BOUNDS_M       <- c(1.35, 3.80) # Submerged tank altitude bounds (blocks 1.20m air locks)
ALT_MEDIAN_TOL_M   <- 0.12          # Max deviation from local 5-frame rolling median (m)
ALT_MAX_SLEW_MPS   <- 0.35          # Max frame-to-frame vertical altimeter slew rate (m/s)
HOIST_BUFFER_SEC   <- 5.0           # Splash-zone buffer around out-of-water/NA pings (s)
MAX_TILT_DEG       <- 35.0          # Maximum |pitch| and |roll_adj| envelope (deg)
MIN_SEGS_DETECTION <- 100           # Rejects edge-clipped small subgrids
MAX_1SQ_RATIO      <- 1.35          # Rejects fiber-optic scanline tears
MAX_1SQ_CV         <- 0.12          # Rejects non-uniform grid spacing
DIAG_RATIO_TOL     <- 0.12          # Tolerance around sqrt(2) = 1.4142

# ==============================================================================
# SECTION 2: Field of View (FOV) Function (From Han)
# ==============================================================================
# Pinhole projection footprint in seawater (f = 16 mm * 1.33, pixel = 5.86 um).
FOV <- function(x) {
  altitude <- x[1]
  roll     <- x[2]
  pitch    <- x[3]
  
  DTOR        <- 0.017453
  focalLength <- 16 * 0.00133
  PIXEL_SIZE  <- 0.00000586
  
  sP <- sin(pitch * DTOR); cP <- cos(pitch * DTOR)
  sR <- sin(roll * DTOR);  cR <- cos(roll * DTOR)
  
  altitude <- altitude * cP * cR
  
  m <- matrix(0, nrow = 3, ncol = 3)
  m[1,1] <- cP;        m[1,2] <- 0.0; m[1,3] <- -sP
  m[2,1] <- sP * sR;   m[2,2] <- cR;  m[2,3] <- cP * sR
  m[3,1] <- sP * cR;   m[3,2] <- -sR; m[3,3] <- cP * cR
  
  ulX <- -1936 / 2 * PIXEL_SIZE; ulY <-  1216 / 2 * PIXEL_SIZE
  urX <- -ulX;                   urY <-  ulY
  llX <-  ulX;                   llY <- -ulY
  lrX <- -ulX;                   lrY <- -ulY
  
  pts <- list(
    c(ulX, ulY, -focalLength),
    c(llX, llY, -focalLength),
    c(lrX, lrY, -focalLength),
    c(urX, urY, -focalLength)
  )
  
  X <- numeric(4); Y <- numeric(4); Z <- numeric(4)
  for (i in 1:4) {
    p <- pts[[i]]
    res <- m %*% p
    X[i] <- res[1] * (altitude / res[3])
    Y[i] <- res[2] * (altitude / res[3])
    Z[i] <- res[3]
  }
  
  area <- 0; j <- 4
  for (i in 1:4) {
    area <- area + (X[j] + X[i]) * (Y[j] - Y[i])
    j <- i
  }
  return(-area / 2)
}

MMPIX <- function(xpix = 1936, ypix = 1216, fov) {
  sqrt(fov / (xpix * ypix)) * 1000
}

# Helper: Parse exact fractional-second UTC timestamp from HabCam imagename
parse_habcam_ts_sec <- function(img_keys) {
  m <- regexpr("202[0-9]{5}\\.[0-9]{6,9}", img_keys)
  out <- rep(NA_real_, length(img_keys))
  valid <- m != -1
  if (any(valid)) {
    ts_str   <- regmatches(img_keys, m)
    ymd      <- substr(ts_str, 1, 8)
    hms      <- substr(ts_str, 10, 15)
    frac_str <- substr(ts_str, 16, nchar(ts_str))
    base_sec <- as.numeric(as.POSIXct(paste0(ymd, " ", hms), format = "%Y%m%d %H%M%S", tz = "UTC"))
    frac_sec <- ifelse(nzchar(frac_str), as.numeric(paste0("0.", frac_str)), 0)
    out[valid] <- base_sec + frac_sec
  }
  return(out)
}

# ==============================================================================
# SECTION 3: Telemetry Ingestion, Time-Series Altimeter QC & Scale Lookup
#
# Overview: Builds a custom millimeter ruler for every photo and screens the
# chronological altimeter time series before segment filtering:
#   1. Parses millisecond timestamps and flags frames within +/-5 s of any
#      out-of-water (<1.35 m), >3.80 m, or dropped (NA) altimeter ping.
#   2. Groups continuous runs (gaps <= 10 s) and flags transient acoustic
#      multipath dips or spikes (>0.12 m off 5-frame rolling median or >0.35 m/s
#      vertical slew rate).
#   3. Zero-references roll (subtracting 137 deg) and computes pinhole FOV scale.
# ==============================================================================
message(">>> Ingesting vehicle telemetry & screening altimeter time series...")

# 1. Load vehicle telemetry (uses in-memory data frame if already loaded)
if (exists("All_img_data") && is.data.frame(All_img_data)) {
  telem_raw <- All_img_data
} else if (file.exists(TELEMETRY_CSV)) {
  telem_raw <- read_csv(TELEMETRY_CSV, show_col_types = FALSE)
} else {
  stop(sprintf("Telemetry CSV file not found at: '%s'.", TELEMETRY_CSV))
}

# 2. Chronological time-series screening (hoist transition buffer + multipath spikes)
telem_ts <- telem_raw %>%
  mutate(
    img_key = file_path_sans_ext(basename(as.character(imagename))),
    ts_sec  = parse_habcam_ts_sec(img_key)
  ) %>%
  arrange(ts_sec, img_key)

# Identify timestamps of out-of-water / out-of-bounds / dropped altimeter events
bad_alt_mask <- is.na(telem_ts$alt1) |
  telem_ts$alt1 < ALT_BOUNDS_M[1] |
  telem_ts$alt1 > ALT_BOUNDS_M[2]
bad_ts <- sort(unique(telem_ts$ts_sec[!is.na(telem_ts$ts_sec) & bad_alt_mask]))

if (length(bad_ts) > 0) {
  idx_left  <- findInterval(telem_ts$ts_sec, bad_ts)
  idx_right <- pmin(idx_left + 1L, length(bad_ts))
  d_left    <- ifelse(idx_left >= 1L, abs(telem_ts$ts_sec - bad_ts[pmax(idx_left, 1L)]), Inf)
  d_right   <- abs(bad_ts[idx_right] - telem_ts$ts_sec)
  telem_ts$hoist_buffer_flag <- !is.na(telem_ts$ts_sec) & (pmin(d_left, d_right) <= HOIST_BUFFER_SEC)
} else {
  telem_ts$hoist_buffer_flag <- FALSE
}

# Drop incomplete sensor rows, assign continuous deployment runs, and detect multipath spikes
telem_clean <- telem_ts %>%
  filter(!is.na(pitch), !is.na(roll), !is.na(alt1)) %>%
  mutate(
    dt_prev  = ts_sec - lag(ts_sec),
    run_id   = cumsum(is.na(dt_prev) | dt_prev > 10.0),
    roll_raw = roll,
    roll_adj = roll - ROLL_NEUTRAL_DEG
  ) %>%
  group_by(run_id) %>%
  mutate(
    n_run     = n(),
    alt1_med  = if (n_run[1] >= 5L) {
      as.numeric(runmed(alt1, k = 5L, endrule = "median"))
    } else if (n_run[1] >= 3L) {
      as.numeric(runmed(alt1, k = 3L, endrule = "median"))
    } else {
      alt1
    },
    dt_p      = ts_sec - lag(ts_sec),
    dt_n      = lead(ts_sec) - ts_sec,
    slew_prev = ifelse(!is.na(dt_p) & dt_p > 0 & dt_p <= 10.0, abs(alt1 - lag(alt1)) / dt_p, 0),
    slew_next = ifelse(!is.na(dt_n) & dt_n > 0 & dt_n <= 10.0, abs(lead(alt1) - alt1) / dt_n, 0),
    dev_med   = abs(alt1 - alt1_med),
    alt_spike_flag = (dev_med > ALT_MEDIAN_TOL_M & (slew_prev > ALT_MAX_SLEW_MPS | slew_next > ALT_MAX_SLEW_MPS)) |
      (dev_med > (2.0 * ALT_MEDIAN_TOL_M)) |
      (slew_prev > ALT_MAX_SLEW_MPS & slew_next > ALT_MAX_SLEW_MPS)
  ) %>%
  ungroup()

message(sprintf(">>> Telemetry QC flags pre-computed: %d hoist-buffer frames, %d altimeter-spike frames.",
                sum(telem_clean$hoist_buffer_flag, na.rm = TRUE),
                sum(telem_clean$alt_spike_flag, na.rm = TRUE)))

# 3. Compute operational seafloor footprint (m^2) and pixel scale (mm/pixel)
#    using the standard HabCam pinhole seawater-refraction model (1936 x 1216 px)
fov_matrix <- as.matrix(telem_clean[, c("alt1", "roll_adj", "pitch")])
telem_clean$fov_m2       <- apply(fov_matrix, 1, FOV)
telem_clean$mm_per_pixel <- MMPIX(1936, 1216, telem_clean$fov_m2)

# 4. Retain essential telemetry, scaling, and QC flag columns for fast joins
telem_lookup <- telem_clean %>%
  group_by(img_key) %>%
  slice(1) %>%
  ungroup() %>%
  select(img_key, alt1, pitch, roll_raw, roll_adj, fov_m2, mm_per_pixel,
         hoist_buffer_flag, alt_spike_flag)

telem_qc_flags <- telem_lookup %>%
  select(img_key, hoist_buffer_flag, alt_spike_flag)

# ==============================================================================
# SECTION 4: Embedded Python Extraction Engine + Visual QC Renderers
#
# Overview: Runs OpenCV computer vision inside R across multiple CPU threads to
# process stereo images in parallel. Each wide stereo frame is split into Port
# (left) and Starboard (right) views, screened for motion blur, and contrast-
# balanced (CLAHE). It then locates interior checkerboard corners (7x6 down to
# 4x3), discards frames distorted by fiber-optic scanline tears, measures 14
# types of grid lines at least 20 pixels away from image borders, and draws
# corner-mesh overlays for live visual quality control.
# ==============================================================================
py_run_string('
import os, cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def detect_rigid_grid(channel_img):
    if channel_img.dtype != np.uint8:
        channel_img = channel_img.astype(np.uint8)
        
    # Local contrast equalization (CLAHE) + 0.5x anti-aliased view to suppress square grain
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_full = clahe.apply(channel_img)
    norm_half = cv2.resize(norm_full, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    flags_large = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER
    flags_exact = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    criteria    = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    
    # 1. Fast 0.5x search with CALIB_CB_LARGER for 8-column / edge-clipped boards
    for seed in [(6, 5), (5, 6), (5, 5), (5, 4), (4, 5)]:
        ret, c, m = cv2.findChessboardCornersSBWithMeta(norm_half, seed, flags=flags_large)
        if ret and m is not None:
            rows, cols = m.shape
            grid = c.reshape((rows, cols, 2)) * 2.0
            
            # Trim outer rows/cols containing unverified or frame-clipped corners (meta == 0)
            while m.shape[0] >= 3 and m.shape[1] >= 3 and not np.all(m > 0):
                r_zeros = np.sum(m == 0, axis=1)
                c_zeros = np.sum(m == 0, axis=0)
                if r_zeros.max() >= c_zeros.max():
                    drop_r = int(np.argmax(r_zeros))
                    m = np.delete(m, drop_r, axis=0)
                    grid = np.delete(grid, drop_r, axis=0)
                else:
                    drop_c = int(np.argmax(c_zeros))
                    m = np.delete(m, drop_c, axis=1)
                    grid = np.delete(grid, drop_c, axis=1)
                    
            if m.shape[0] >= 3 and m.shape[1] >= 3 and (m.shape[0] * m.shape[1] >= 12) and np.all(m > 0):
                pts = grid.reshape(-1, 1, 2).astype(np.float32)
                pts = cv2.cornerSubPix(norm_full, pts, (9, 9), (-1, -1), criteria)
                return True, pts, m
                
    # 2. Fallback to full-resolution exact subgrid ladder
    ladder = [
        (7, 6), (6, 7),
        (7, 5), (5, 7),
        (6, 5), (5, 6),
        (6, 4), (4, 6),
        (5, 5),
        (5, 4), (4, 5),
        (6, 3), (3, 6),
        (5, 3), (3, 5),
        (4, 4),
        (4, 3), (3, 4)
    ]
    for pat in ladder:
        ret, c, m = cv2.findChessboardCornersSBWithMeta(norm_full, pat, flags=flags_exact)
        if ret and m is not None and np.all(m > 0):
            rows, cols = m.shape
            if c.shape[0] == rows * cols:
                return True, c, m
                
    return False, None, None

def extract_from_channel(channel_img, cam_label, base_name, square_size_mm, blur_thresh):
    h, w = channel_img.shape[:2]
    
    qc = {
        "imagename": base_name,
        "camera": cam_label,
        "status": "NO_BOARD_IN_FOV",
        "blur_score": 0.0,
        "subgrid_shape": "None",
        "corners_found": 0,
        "segments_extracted": 0,
        "ratio_1sq_px": np.nan,
        "cv_1sq_px": np.nan
    }
    
    # 1. Laplacian sharpness check
    lap_var = float(cv2.Laplacian(channel_img, cv2.CV_64F).var())
    qc["blur_score"] = round(lap_var, 2)
    if lap_var < float(blur_thresh):
        qc["status"] = "REJECTED_BLUR"
        return [], qc
        
    # 2. Rigid interior chessboard detection
    found, best_corners, meta = detect_rigid_grid(channel_img)
    if not found:
        return [], qc
        
    pts = best_corners.reshape(-1, 2)
    rows, cols = meta.shape
    grid = pts.reshape((rows, cols, 2))
    
    qc["subgrid_shape"] = f"{cols}x{rows}"
    qc["corners_found"] = cols * rows
    
    # 3. Physical 1-square rigidity check (blocks fiber-optic scanline tears)
    d_horiz = np.hypot(np.diff(grid[:, :, 0], axis=1), np.diff(grid[:, :, 1], axis=1))
    d_vert  = np.hypot(np.diff(grid[:, :, 0], axis=0), np.diff(grid[:, :, 1], axis=0))
    all_1sq = np.concatenate([d_horiz.ravel(), d_vert.ravel()])
    
    ratio_1sq = float(all_1sq.max() / all_1sq.min())
    cv_1sq    = float(all_1sq.std() / all_1sq.mean())
    qc["ratio_1sq_px"] = round(ratio_1sq, 3)
    qc["cv_1sq_px"]    = round(cv_1sq, 3)
    
    if ratio_1sq > 1.35 or cv_1sq > 0.12:
        qc["status"] = "REJECTED_FIBER_TEAR"
        return [], qc
        
    # 4. Directional segment extraction
    offsets = [
        (0, 1, "horiz_1sq"), (1, 0, "vert_1sq"),
        (1, 1, "diag_1sq_dr"), (1, -1, "diag_1sq_dl"),
        (1, 2, "knight_1x2_dr"), (1, -2, "knight_1x2_dl"),
        (2, 1, "knight_2x1_dr"), (2, -1, "knight_2x1_dl"),
        (0, 2, "horiz_2sq"), (2, 0, "vert_2sq"),
        (2, 2, "diag_2sq_dr"), (2, -2, "diag_2sq_dl"),
        (0, 3, "horiz_3sq"), (3, 0, "vert_3sq")
    ]
    
    margin = 20
    out_segments = []
    
    for r in range(rows):
        for c in range(cols):
            p1 = grid[r, c]
            if not (margin <= p1[0] <= (w - margin) and margin <= p1[1] <= (h - margin)):
                continue
                
            for dr, dc, lbl in offsets:
                tr, tc = r + dr, c + dc
                if 0 <= tr < rows and 0 <= tc < cols:
                    p2 = grid[tr, tc]
                    if not (margin <= p2[0] <= (w - margin) and margin <= p2[1] <= (h - margin)):
                        continue
                        
                    dx = float(p2[0] - p1[0])
                    dy = float(p2[1] - p1[1])
                    pixel_len = float(np.hypot(dx, dy))
                    true_len = float(float(square_size_mm) * np.hypot(dr, dc))
                    theta = float(np.degrees(np.arctan2(abs(dy), abs(dx)))) % 180.0
                    cx = float(((p1[0] + p2[0]) / 2.0) / float(w))
                    cy = float(((p1[1] + p2[1]) / 2.0) / float(h))
                    
                    out_segments.append({
                        "imagename": base_name,
                        "camera": cam_label,
                        "segment_type": lbl,
                        "pos_x": cx,
                        "pos_y": cy,
                        "theta": theta,
                        "true_len_mm": true_len,
                        "pixel_len": pixel_len,
                        "subgrid": f"{cols}x{rows}"
                    })
                    
    qc["status"] = "SUCCESS"
    qc["segments_extracted"] = len(out_segments)
    return out_segments, qc

def process_single_frame(args):
    img_path, square_size_mm, blur_thresh = args
    base_name = os.path.basename(img_path)
    
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return [], [{"imagename": base_name, "camera": "unknown", "status": "READ_ERROR",
                         "blur_score": 0.0, "subgrid_shape": "None", "corners_found": 0,
                         "segments_extracted": 0, "ratio_1sq_px": np.nan, "cv_1sq_px": np.nan}]
                         
        h, w = img.shape[:2]
        if w >= 3800:
            half_w = w // 2
            segs_p, qc_p = extract_from_channel(img[:, :half_w], "port", base_name, square_size_mm, blur_thresh)
            segs_s, qc_s = extract_from_channel(img[:, half_w:], "starboard", base_name, square_size_mm, blur_thresh)
            return segs_p + segs_s, [qc_p, qc_s]
        else:
            segs, qc = extract_from_channel(img, "port", base_name, square_size_mm, blur_thresh)
            return segs, [qc]
            
    except Exception:
        return [], [{"imagename": base_name, "camera": "unknown", "status": "ERROR_EXCEPTION",
                     "blur_score": 0.0, "subgrid_shape": "None", "corners_found": 0,
                     "segments_extracted": 0, "ratio_1sq_px": np.nan, "cv_1sq_px": np.nan}]

def extract_chunk_python(paths, square_size, blur_thresh, workers):
    tasks = [(p, square_size, blur_thresh) for p in paths]
    all_segments = []
    all_qc = []
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        for segs, qcs in ex.map(process_single_frame, tasks):
            if segs:
                all_segments.extend(segs)
            all_qc.extend(qcs)
    return pd.DataFrame(all_segments), pd.DataFrame(all_qc)

def render_chunk_preview_python(img_path, cam_label, out_png_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return False
    h, w = img_bgr.shape[:2]
    if w >= 3800:
        half_w = w // 2
        img_bgr = img_bgr[:, :half_w] if cam_label == "port" else img_bgr[:, half_w:]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    found, best_corners, meta = detect_rigid_grid(gray)
    if found:
        rows, cols = meta.shape
        cv2.drawChessboardCorners(img_bgr, (cols, rows), best_corners, True)
    cv2.imwrite(out_png_path, img_bgr)
    return True

def draw_vector_proof_python(img_path, cam_label, mm_per_px, out_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    if w >= 3800:
        img = img[:, :w//2] if cam_label == "port" else img[:, w//2:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, c, m = detect_rigid_grid(gray)
    if found:
        rows, cols = m.shape
        grid = c.reshape((rows, cols, 2))
        examples = [
            ((0, 0), (0, 1), (0, 255, 0),   "1-sq (80.0mm)"),
            ((1, 0), (2, 1), (255, 0, 255), "Diag (113.1mm)"),
            ((0, 1), (1, 3), (0, 215, 255), "Knight (178.9mm)"),
            ((3, 0), (3, 3), (255, 140, 0), "3-sq (240.0mm)")
        ]
        for (r1, c1), (r2, c2), color, lbl in examples:
            if r2 < rows and c2 < cols:
                p1 = tuple(np.round(grid[r1, c1]).astype(int))
                p2 = tuple(np.round(grid[r2, c2]).astype(int))
                px_d = np.hypot(grid[r2, c2, 0] - grid[r1, c1, 0], grid[r2, c2, 1] - grid[r1, c1, 1])
                mm_d = px_d * float(mm_per_px)
                cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)
                cv2.circle(img, p1, 5, (0, 0, 255), -1)
                cv2.circle(img, p2, 5, (0, 0, 255), -1)
                mid = ((p1[0] + p2[0]) // 2 + 8, (p1[1] + p2[1]) // 2)
                cv2.putText(img, f"{lbl} -> Meas: {mm_d:.1f}mm", mid,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.imwrite(out_path, img)
        return True
    return False
')

# ==============================================================================
# SECTION 5: Checkpoint-Managed Batch Extraction with Live Visual QC
#
# Overview: Manages the image extraction assembly line. It scans the network
# drive for all stereo photos, skips frames already saved in checkpoints, and
# processes any remaining images in auto-saved batches.
# ==============================================================================
message(">>> Enumerating network directory via Windows kernel (~30-60 seconds)...")

win_unc_path  <- gsub("/", "\\\\", IMAGE_DIR)
scan_cmd      <- sprintf('cmd /c (dir "%s\\*.png" /s /b & dir "%s\\*.tif" /s /b) 2>nul', win_unc_path, win_unc_path)
t_scan        <- Sys.time()
all_img_paths <- system(scan_cmd, intern = TRUE)

all_img_paths <- all_img_paths[grepl("\\.(png|tif|tiff)$", all_img_paths, ignore.case = TRUE)]
img_keys_all  <- file_path_sans_ext(basename(all_img_paths))
path_by_base  <- setNames(all_img_paths, basename(all_img_paths))

cat(sprintf(">>> Discovered %d images across network in %.1f seconds.\n", 
            length(all_img_paths), as.numeric(difftime(Sys.time(), t_scan, units = "secs"))))

checkpoint_files <- list.files(CHECKPOINT_DIR, pattern = "^manifest_chunk_.*\\.rds$", full.names = TRUE)
processed_keys   <- character(0)

if (length(checkpoint_files) > 0) {
  for (chk in checkpoint_files) {
    m_df <- readRDS(chk)
    if ("img_key" %in% names(m_df)) {
      processed_keys <- union(processed_keys, m_df$img_key)
    }
  }
  message(sprintf(">>> Resuming from %d existing checkpoints (%d frames already processed).",
                  length(checkpoint_files), length(processed_keys)))
}

remaining_idx    <- which(!img_keys_all %in% processed_keys)
paths_to_process <- all_img_paths[remaining_idx]
vector_proof_png <- file.path(FIGURE_DIR, "fig_measurement_vector_proof.png")

if (length(paths_to_process) > 0) {
  total_chunks <- ceiling(length(paths_to_process) / CHUNK_SIZE)
  
  for (i in seq_len(total_chunks)) {
    start_idx   <- ((i - 1) * CHUNK_SIZE) + 1
    end_idx     <- min(i * CHUNK_SIZE, length(paths_to_process))
    chunk_paths <- paths_to_process[start_idx:end_idx]
    
    chunk_num <- length(checkpoint_files) + i
    seg_file  <- file.path(CHECKPOINT_DIR, sprintf("segments_chunk_%04d.rds", chunk_num))
    man_file  <- file.path(CHECKPOINT_DIR, sprintf("manifest_chunk_%04d.rds", chunk_num))
    
    cat(sprintf("\n--- Processing Chunk %d/%d (Checkpoint #%04d | %d frames) ---\n", 
                i, total_chunks, chunk_num, length(chunk_paths)))
    t0 <- Sys.time()
    
    py_res <- py$extract_chunk_python(
      paths       = chunk_paths,
      square_size = SQUARE_SIZE_MM,
      blur_thresh = BLUR_THRESHOLD,
      workers     = NUM_WORKERS
    )
    
    chunk_segs <- as_tibble(py_res[[1]])
    chunk_man  <- as_tibble(py_res[[2]]) %>%
      mutate(img_key = file_path_sans_ext(basename(as.character(imagename))))
    
    saveRDS(chunk_man, man_file)
    
    if (nrow(chunk_segs) > 0) {
      chunk_segs <- chunk_segs %>%
        mutate(img_key = file_path_sans_ext(basename(as.character(imagename)))) %>%
        inner_join(telem_lookup, by = "img_key") %>%
        mutate(
          est_len_mm = pixel_len * mm_per_pixel,
          bias_pct   = ((est_len_mm - true_len_mm) / true_len_mm) * 100
        )
      saveRDS(chunk_segs, seg_file)
    } else {
      saveRDS(tibble(), seg_file)
    }
    
    elapsed  <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    n_succ   <- sum(chunk_man$status == "SUCCESS")
    n_tear   <- sum(chunk_man$status == "REJECTED_FIBER_TEAR")
    chunk_r2 <- if (nrow(chunk_segs) > 5) {
      chunk_segs %>%
        group_by(imagename, camera) %>%
        summarize(r2 = cor(pixel_len, true_len_mm)^2, .groups = "drop") %>%
        pull(r2) %>% median(na.rm = TRUE)
    } else NA_real_
    
    cat(sprintf("Saved %s | Valid: %d | Fiber Tears Cut: %d | Segs: %d | Med R^2: %.5f | Time: %.1fs (%.1f img/s)\n",
                basename(seg_file), n_succ, n_tear, nrow(chunk_segs), chunk_r2, elapsed, length(chunk_paths) / elapsed))
    
    if (!file.exists(vector_proof_png) && nrow(chunk_segs) > 0) {
      full_cand <- chunk_segs %>%
        group_by(imagename, camera) %>%
        summarize(n_segs = n(), mm_px = first(mm_per_pixel), .groups = "drop") %>%
        filter(n_segs >= 300) %>%
        slice(1)
      
      if (nrow(full_cand) == 1) {
        vp_bname <- basename(as.character(full_cand$imagename))
        vp_path  <- path_by_base[vp_bname]
        if (!is.na(vp_path) && file.exists(vp_path)) {
          py$draw_vector_proof_python(vp_path, as.character(full_cand$camera), full_cand$mm_px, vector_proof_png)
          message(sprintf("Generated vector measurement proof: %s", vector_proof_png))
        }
      }
    }
    
    succ_rows <- chunk_man %>% filter(status == "SUCCESS")
    if (nrow(succ_rows) > 0) {
      sample_row   <- succ_rows[1, ]
      sample_bname <- basename(as.character(sample_row$imagename))
      sample_path  <- path_by_base[sample_bname]
      preview_png  <- file.path(PREVIEW_DIR, sprintf("preview_chunk_%04d.png", chunk_num))
      
      if (!is.na(sample_path) && file.exists(sample_path)) {
        ok <- py$render_chunk_preview_python(sample_path, as.character(sample_row$camera), preview_png)
        if (isTRUE(ok) && file.exists(preview_png)) {
          tryCatch({
            prev_img <- readPNG(preview_png)
            h_p <- dim(prev_img)[1]; w_p <- dim(prev_img)[2]
            mean_b <- if (nrow(chunk_segs) > 0) {
              mean(abs(chunk_segs$bias_pct[chunk_segs$imagename == sample_bname & chunk_segs$camera == sample_row$camera]), na.rm = TRUE)
            } else NA_real_
            
            par(mar = c(0.5, 0.5, 2.8, 0.5))
            plot(c(0, w_p), c(h_p, 0), type = "n", xlab = "", ylab = "", axes = FALSE, asp = 1)
            rasterImage(as.raster(prev_img), 0, h_p, w_p, 0)
            title(main = sprintf("Chunk %d/%d Live QC [%s] %s\nGrid: %s | 1sq Ratio: %.2f | Mean |Bias|: %.2f%%",
                                 i, total_chunks, toupper(sample_row$camera), sample_bname,
                                 sample_row$subgrid_shape, sample_row$ratio_1sq_px, mean_b),
                  cex.main = 0.95)
          }, error = function(e) NULL)
        }
      }
    }
    
    gc()
  }
}

# ==============================================================================
# SECTION 6: Aggregate Checkpoints & Run Integrated Telemetry + Geometric Cleaning
#
# Overview: Merges all saved batch checkpoints into a single raw dataset and
# audits every checkerboard detection against physical, time-series altimeter,
# and Euclidean geometry rules BEFORE exporting `calibrated_segments_clean.rds`
# and generating any diagnostic figures:
#   - Manual blacklist check
#   - Submerged altitude bounds [1.35 m, 3.80 m] (blocks 1.20 m out-of-water locks)
#   - Hoist / splash-zone transition buffer (+/- 5 s around dry/NA pings)
#   - Altimeter multipath / jump filter (rolling median & vertical slew rate)
#   - Maximum tilt envelope (|pitch| <= 35 deg, |roll_adj| <= 35 deg)
#   - Minimum segment count (>= 100), 1-sq rigidity, and diagonal proportion
# ==============================================================================
message("\n>>> Aggregating checkpoints and running integrated telemetry + geometric cleaning...")

all_man_files <- list.files(CHECKPOINT_DIR, pattern = "^manifest_chunk_.*\\.rds$", full.names = TRUE)
manifest_full <- bind_rows(lapply(all_man_files, readRDS))
saveRDS(manifest_full, FINAL_MANIFEST_RDS)

all_seg_files <- list.files(CHECKPOINT_DIR, pattern = "^segments_chunk_.*\\.rds$", full.names = TRUE)
segments_raw  <- bind_rows(lapply(all_seg_files, function(f) {
  d <- readRDS(f)
  if (nrow(d) > 0 && "bias_pct" %in% names(d)) d else NULL
}))
saveRDS(segments_raw, RAW_SEGMENTS_RDS)

# Optional manual exclusion blacklist
manual_bad_keys <- character(0)
if (file.exists(MANUAL_EXCLUSIONS)) {
  manual_df <- read_csv(MANUAL_EXCLUSIONS, show_col_types = FALSE)
  if ("imagename" %in% names(manual_df)) {
    manual_bad_keys <- file_path_sans_ext(basename(as.character(manual_df$imagename)))
  }
}

# Audit every (imagename, camera) detection against physical, altimeter & geometric rules
qc_audit_ledger <- segments_raw %>%
  mutate(img_key = file_path_sans_ext(basename(as.character(imagename)))) %>%
  group_by(img_key, imagename, camera) %>%
  summarize(
    alt1         = first(alt1),
    pitch        = first(pitch),
    roll_adj     = first(roll_adj),
    n_segs       = n(),
    mean_1sq_px  = mean(pixel_len[segment_type %in% c("horiz_1sq", "vert_1sq")], na.rm = TRUE),
    sd_1sq_px    = sd(pixel_len[segment_type %in% c("horiz_1sq", "vert_1sq")], na.rm = TRUE),
    min_1sq_px   = min(pixel_len[segment_type %in% c("horiz_1sq", "vert_1sq")], na.rm = TRUE),
    max_1sq_px   = max(pixel_len[segment_type %in% c("horiz_1sq", "vert_1sq")], na.rm = TRUE),
    mean_diag_px = mean(pixel_len[grepl("^diag_1sq", segment_type)], na.rm = TRUE),
    .groups      = "drop"
  ) %>%
  left_join(telem_qc_flags, by = "img_key") %>%
  mutate(
    cv_1sq_px    = sd_1sq_px / mean_1sq_px,
    ratio_1sq_px = max_1sq_px / min_1sq_px,
    diag_ratio   = mean_diag_px / mean_1sq_px,
    qc_status    = case_when(
      img_key %in% manual_bad_keys                                                              ~ "Rejected: Manual Blacklist",
      is.na(alt1) | alt1 < ALT_BOUNDS_M[1] | alt1 > ALT_BOUNDS_M[2]                             ~ "Rejected: Altitude Bounds (<1.35m or >3.80m)",
      coalesce(hoist_buffer_flag, FALSE)                                                        ~ "Rejected: Hoist Transition Buffer (+/-5s)",
      coalesce(alt_spike_flag, FALSE)                                                           ~ "Rejected: Altimeter Spike / Multipath",
      is.na(pitch) | is.na(roll_adj) | abs(pitch) > MAX_TILT_DEG | abs(roll_adj) > MAX_TILT_DEG ~ "Rejected: Attitude Envelope",
      n_segs < MIN_SEGS_DETECTION                                                               ~ "Rejected: Edge-Clipped Subgrid (<100 segs)",
      is.na(ratio_1sq_px) | ratio_1sq_px > MAX_1SQ_RATIO | cv_1sq_px > MAX_1SQ_CV               ~ "Rejected: Fiber Tear / Non-Rigid Grid",
      is.na(diag_ratio) | abs(diag_ratio - sqrt(2)) > DIAG_RATIO_TOL                            ~ "Rejected: Euclidean Ratio Distortion",
      TRUE                                                                                      ~ "VALID"
    )
  )

write_csv(qc_audit_ledger, QC_LEDGER_CSV)

valid_detections <- qc_audit_ledger %>%
  filter(qc_status == "VALID") %>%
  select(imagename, camera)

# Build analysis-ready clean dataset BEFORE any figure generation
segments_clean <- segments_raw %>%
  inner_join(valid_detections, by = c("imagename", "camera")) %>%
  filter(
    !is.na(bias_pct), !is.na(pitch), !is.na(roll_adj),
    !is.na(alt1), !is.na(theta), !is.na(pos_x), !is.na(pos_y)
  ) %>%
  mutate(
    img_key   = file_path_sans_ext(basename(as.character(imagename))),
    camera    = factor(as.character(camera)),
    imagename = factor(as.character(imagename))
  )

saveRDS(segments_clean, CLEAN_SEGMENTS_RDS)

# ==============================================================================
# SECTION 7: Clean Dataset Diagnostic & Measurement Accuracy Report
#
# Overview: Prints a four-part quality report card to the console strictly on
# the filtered dataset before figures or Stage 2 GAMM modeling begin.
# ==============================================================================
cat("\n========================================================================\n")
cat("             HABCAM PIPELINE & CLEANED DATA AUDIT REPORT                \n")
cat("========================================================================\n\n")

cat("--- 1. GEOMETRIC, ALTIMETER & HOIST QC SUMMARY (DETECTION LEVEL) ---\n")
qc_audit_ledger %>%
  count(qc_status) %>%
  mutate(pct = round(n / sum(n) * 100, 2)) %>%
  print()

cat(sprintf("\nAnalysis-Ready Clean Dataset: %d segments across %d frames (%d camera detections)\n",
            nrow(segments_clean), n_distinct(segments_clean$img_key), nrow(valid_detections)))

cat("\n--- 2. EUCLIDEAN PROPORTIONALITY CHECK (CLEANED DATA ONLY) ---\n")
base_1sq <- mean(segments_clean$pixel_len[segments_clean$segment_type %in% c("horiz_1sq", "vert_1sq")])
diag_px  <- mean(segments_clean$pixel_len[grepl("^diag_1sq", segments_clean$segment_type)])
kngt_px  <- mean(segments_clean$pixel_len[grepl("^knight", segments_clean$segment_type)])
twos_px  <- mean(segments_clean$pixel_len[grepl("2sq$", segments_clean$segment_type)])

cat("1-Square Baseline:            ", round(base_1sq, 2), "pixels\n")
cat("Diagonal / 1-Sq (Target 1.4142):", round(diag_px / base_1sq, 4), 
    ifelse(abs(diag_px / base_1sq - sqrt(2)) < 0.02, "[PASS]", "[FAIL]"), "\n")
cat("Knight   / 1-Sq (Target 2.2361):", round(kngt_px / base_1sq, 4), 
    ifelse(abs(kngt_px / base_1sq - sqrt(5)) < 0.02, "[PASS]", "[FAIL]"), "\n")
cat("2-Square / 1-Sq (Target 2.0000):", round(twos_px / base_1sq, 4), 
    ifelse(abs(twos_px / base_1sq - 2.0) < 0.02, "[PASS]", "[FAIL]"), "\n")

cat("\n--- 3. WITHIN-FRAME LINEARITY & ORTHOGONALITY (CLEANED DATA ONLY) ---\n")
linearity_audit <- segments_clean %>%
  group_by(imagename, camera) %>%
  summarize(
    r_sq       = cor(pixel_len, true_len_mm)^2,
    theta_diff = abs(mean(theta[segment_type == "vert_1sq"]) - mean(theta[segment_type == "horiz_1sq"])),
    .groups    = "drop"
  )
cat(sprintf("Median Within-Frame R^2 (Target > 0.998):      %.5f\n", median(linearity_audit$r_sq, na.rm = TRUE)))
cat(sprintf("Detections Passing R^2 > 0.995:                %.2f%%\n", mean(linearity_audit$r_sq > 0.995, na.rm = TRUE) * 100))
cat(sprintf("Median Horiz-vs-Vert Angle Diff (Target 90°):  %.2f deg\n", median(linearity_audit$theta_diff, na.rm = TRUE)))

cat("\n--- 4. FLAT-TRIM PHYSICAL LENGTH CHECK (|Pitch|<=4°, |Roll|<=4°, Center FOV) ---\n")
nadir_segs <- segments_clean %>%
  filter(abs(pitch) <= 4.0, abs(roll_adj) <= 4.0,
         pos_x >= 0.30, pos_x <= 0.70, pos_y >= 0.30, pos_y <= 0.70)

if (nrow(nadir_segs) > 0) {
  nadir_segs %>%
    group_by(segment_type, true_len_mm) %>%
    summarize(
      n_obs         = n(),
      mean_est_mm   = round(mean(est_len_mm), 2),
      sd_est_mm     = round(sd(est_len_mm), 2),
      mean_bias_pct = round(mean(bias_pct), 2),
      .groups       = "drop"
    ) %>%
    arrange(true_len_mm) %>%
    print(n = 15)
}

# ==============================================================================
# SECTION 8: Raw Data Summary (Strictly on Cleaned / Included Frames)
# ==============================================================================
message("\n>>> Generating publication summary tables in Tables/...")

TABLES_DIR <- "Tables"
dir.create(TABLES_DIR, showWarnings = FALSE)

alt_min_m <- if (exists("ALT_BOUNDS_M")) ALT_BOUNDS_M[1] else 1.35
alt_max_m <- if (exists("ALT_BOUNDS_M")) ALT_BOUNDS_M[2] else 3.80

# Reuse in-memory objects from Section 6 (or load from Data/ if run standalone)
man_tbl <- (if (exists("manifest_full")) manifest_full else readRDS(FINAL_MANIFEST_RDS)) %>%
  mutate(
    img_key   = file_path_sans_ext(basename(as.character(imagename))),
    imagename = as.character(imagename),
    camera    = as.character(camera)
  )

raw_tbl <- (if (exists("segments_raw")) segments_raw else readRDS(RAW_SEGMENTS_RDS)) %>%
  mutate(
    img_key   = file_path_sans_ext(basename(as.character(imagename))),
    imagename = as.character(imagename),
    camera    = as.character(camera)
  )

qc_tbl <- (if (exists("qc_audit_ledger")) qc_audit_ledger else read_csv(QC_LEDGER_CSV, show_col_types = FALSE)) %>%
  mutate(
    img_key   = file_path_sans_ext(basename(as.character(imagename))),
    imagename = as.character(imagename),
    camera    = as.character(camera)
  )

clean_tbl <- (if (exists("segments_clean")) segments_clean else readRDS(CLEAN_SEGMENTS_RDS)) %>%
  mutate(
    img_key   = file_path_sans_ext(basename(as.character(imagename))),
    imagename = as.character(imagename),
    camera    = as.character(camera)
  )

# ==============================================================================
# Table 1: Summary of Rejected Frames and Final Clean Dataset
# Description: Data-attrition ledger tracking total stereo frames, individual
#   camera detections (Port and Starboard), and measured checkerboard segments
#   from raw archive ingestion through computer-vision screening, telemetry and
#   altimeter quality-control filters, and geometric rigidity checks to the
#   final analysis-ready dataset.
# ==============================================================================
cv_stage_order <- c(
  "REJECTED_FIBER_TEAR" = "Rejected: Fiber-Optic Issue",
  "REJECTED_BLUR"       = "Rejected: Motion Blur (Laplacian < 45)",
  "NO_BOARD_IN_FOV"     = "Rejected: No Checkerboard in FOV",
  "READ_ERROR"          = "Rejected: Image Read / I/O Error",
  "ERROR_EXCEPTION"     = "Rejected: Processing Exception"
)

cv_rejections <- man_tbl %>%
  filter(status != "SUCCESS") %>%
  mutate(
    Stage_Step = recode(status, !!!cv_stage_order, .default = paste("Rejected:", status)),
    step_order = match(Stage_Step, unname(cv_stage_order))
  ) %>%
  group_by(Stage_Step, step_order) %>%
  summarize(
    Stereo_Frames     = n_distinct(img_key),
    Camera_Detections = n(),
    Segments          = 0L,
    .groups           = "drop"
  ) %>%
  arrange(step_order, Stage_Step) %>%
  select(-step_order)

cv_success           <- man_tbl %>% filter(status == "SUCCESS")
raw_joined_dets      <- raw_tbl %>% distinct(img_key, imagename, camera)
n_missing_telem_dets <- nrow(cv_success) - nrow(raw_joined_dets)

seg_counts_by_det <- raw_tbl %>%
  group_by(imagename, camera) %>%
  summarize(n_segs_raw = n(), .groups = "drop")

qc_detailed <- qc_tbl %>%
  left_join(seg_counts_by_det, by = c("imagename", "camera")) %>%
  mutate(n_segs_raw = coalesce(n_segs_raw, as.integer(n_segs)))

qc_rule_labels <- c(
  "Rejected: Manual Blacklist"                   = "Rejected: Manual Exclusion Blacklist",
  "Rejected: Altitude Bounds (<1.35m or >3.80m)" = "Rejected: Altitude Bounds (< 1.35 m or > 3.80 m)",
  "Rejected: Hoist Transition Buffer (+/-5s)"    = "Rejected: Out of Water (±5 s)",
  "Rejected: Altimeter Spike / Multipath"        = "Rejected: Altimeter Spike (> 0.35 m/s)",
  "Rejected: Attitude Envelope"                  = "Rejected: Attitude Envelope (|Pitch| or |Roll_adj| > 35°)",
  "Rejected: Edge-Clipped Subgrid (<100 segs)"   = "Rejected: Small Partial Board (< 100 Segments)",
  "Rejected: Fiber Tear / Non-Rigid Grid"        = "Rejected: Non-Rigid Grid / Scanline Tear (Ratio > 1.35 or CV > 0.12)",
  "Rejected: Euclidean Ratio Distortion"         = "Rejected: Euclidean Diagonal Distortion (|Diag/Side - √2| > 0.12)"
)

qc_rejections <- qc_detailed %>%
  filter(qc_status != "VALID") %>%
  mutate(
    Stage_Step = recode(qc_status, !!!qc_rule_labels, .default = qc_status),
    step_order = match(Stage_Step, unname(qc_rule_labels))
  ) %>%
  group_by(Stage_Step, step_order) %>%
  summarize(
    Stereo_Frames     = n_distinct(img_key),
    Camera_Detections = n(),
    Segments          = sum(n_segs_raw),
    .groups           = "drop"
  ) %>%
  arrange(step_order, Stage_Step) %>%
  select(-step_order)

table1_raw <- bind_rows(
  tibble(
    Stage_Step        = "Total Stereo Images Taken",
    Stereo_Frames     = n_distinct(man_tbl$img_key),
    Camera_Detections = nrow(man_tbl),
    Segments          = NA_integer_
  ),
  cv_rejections,
  tibble(
    Stage_Step        = "Total Stereo Images Extracted",
    Stereo_Frames     = n_distinct(cv_success$img_key),
    Camera_Detections = nrow(cv_success),
    Segments          = nrow(raw_tbl)
  ),
  if (n_missing_telem_dets > 0) {
    tibble(
      Stage_Step        = "Rejected: Missing / Unmatched Sensor Log",
      Stereo_Frames     = n_distinct(setdiff(cv_success$img_key, raw_joined_dets$img_key)),
      Camera_Detections = n_missing_telem_dets,
      Segments          = NA_integer_
    )
  },
  qc_rejections,
  tibble(
    Stage_Step        = sprintf("Final Clean Data Set (Port: %s | Starboard: %s)",
                                format(n_distinct(clean_tbl$imagename[clean_tbl$camera == "Port"]), big.mark = ",", trim = TRUE),
                                format(n_distinct(clean_tbl$imagename[clean_tbl$camera == "Starboard"]), big.mark = ",", trim = TRUE)),
    Stereo_Frames     = n_distinct(clean_tbl$img_key),
    Camera_Detections = nrow(distinct(clean_tbl, imagename, camera)),
    Segments          = nrow(clean_tbl)
  )
)

table1_pub <- table1_raw %>%
  mutate(
    `% of Extracted Segments` = ifelse(is.na(Segments) | Segments == 0, "—",
                                       sprintf("%.2f%%", (Segments / nrow(raw_tbl)) * 100)),
    `Stereo Frames`           = ifelse(is.na(Stereo_Frames), "—", format(Stereo_Frames, big.mark = ",", trim = TRUE)),
    `Camera Detections`       = ifelse(is.na(Camera_Detections), "—", format(Camera_Detections, big.mark = ",", trim = TRUE)),
    `Segments`                = ifelse(is.na(Segments), "—", format(Segments, big.mark = ",", trim = TRUE))
  ) %>%
  select(
    ` ` = Stage_Step,
    `Stereo Frames`,
    `Camera Detections`,
    `Segments`,
    `% of Extracted Segments`
  )

print(as.data.frame(table1_pub), row.names = FALSE, right = FALSE)
write_excel_csv(table1_pub, file.path(TABLES_DIR, "Table_1_Data_Cleaning_Summary.csv"))

# ==============================================================================
# Figure 1: Attitude coverage by altitude bin
# Description: Joint pitch-by-roll sampling domain of all cleaned stereo frames
#   across four altitude tiers (1.5 m, 2.0 m, 2.5 m, and 3.0 m). Blue points
#   represent individual frame attitudes (zero-referenced roll vs. pitch)
#   overlaid with 2D kernel density contours highlighting primary test clusters.
# ==============================================================================
message("\n>>> Computing attitude coverage envelope from cleaned frames only...")

used_keys <- unique(segments_clean$img_key)

used_telem <- telem_clean %>%
  filter(img_key %in% used_keys) %>%
  mutate(
    altitude_tier = cut(
      alt1,
      breaks = c(ALT_BOUNDS_M[1], 1.75, 2.25, 2.75, ALT_BOUNDS_M[2]),
      labels = c("1.5 m", "2.0 m", "2.5 m", "3.0 m"),
      include.lowest = TRUE
    ),
    pitch_bin    = round(pitch / 5) * 5,
    roll_adj_bin = round(roll_adj / 5) * 5
  ) %>%
  filter(!is.na(altitude_tier))

used_attitude_summary <- used_telem %>%
  group_by(altitude_tier) %>%
  summarize(
    cleaned_frames    = n(),
    alt1_mean_m       = round(mean(alt1), 2),
    alt1_sd_m         = round(sd(alt1), 2),
    pitch_min_deg     = round(min(pitch), 1),
    pitch_max_deg     = round(max(pitch), 1),
    pitch_mean_deg    = round(mean(pitch), 1),
    pitch_sd_deg      = round(sd(pitch), 1),
    roll_adj_min_deg  = round(min(roll_adj), 1),
    roll_adj_max_deg  = round(max(roll_adj), 1),
    roll_adj_mean_deg = round(mean(roll_adj), 1),
    roll_adj_sd_deg   = round(sd(roll_adj), 1),
    .groups           = "drop"
  )

print(used_attitude_summary)
write_csv(used_attitude_summary, file.path(DATA_DIR, "output_attitude_tier_summary_clean_frames.csv"))

p_coverage_used <- ggplot(used_telem, aes(x = roll_adj, y = pitch)) +
  geom_point(alpha = 0.35, size = 1.0, color = "#2b83ba") +
  geom_density_2d(color = "black", linewidth = 0.35) +
  facet_wrap(~ altitude_tier, nrow = 1) +
  scale_x_continuous(breaks = seq(-30, 30, by = 15), limits = c(-35, 35)) +
  scale_y_continuous(breaks = seq(-30, 30, by = 15), limits = c(-35, 35)) +
  coord_fixed() +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = expression(paste("Roll Angle (", theta[roll], ", degrees)")),
    y = expression(paste("Pitch Angle (", theta[pitch], ", degrees)"))
  )

ggsave(
  file.path(FIGURE_DIR, "Figure_1_Atiitude_Coverage_by_Altitude.png"),
  p_coverage_used,
  width = 12,
  height = 3.4,
  dpi = 300,
  create.dir = TRUE
)

# ==============================================================================
# Empirical Bias Figures (Strictly on Cleaned / Included Data)
# ==============================================================================
message("\n>>> Generating empirical bias figures from cleaned data...")

plot_data <- segments_clean %>%
  mutate(
    camera = recode(trimws(tolower(as.character(camera))),
                    "port" = "Port", "starboard" = "Starboard"),
    altitude_tier = cut(
      alt1,
      breaks = c(ALT_BOUNDS_M[1], 1.75, 2.25, 2.75, ALT_BOUNDS_M[2]),
      labels = c("1.5 m", "2.0 m", "2.5 m", "3.0 m"),
      include.lowest = TRUE
    ),
    orientation_class = case_when(
      (theta >= 65 & theta <= 115) ~ "Transverse / Across-Track (65°–90°)",
      (theta <= 25 | theta >= 155) ~ "Longitudinal / Along-Track (0°–25°)",
      TRUE ~ "Diagonal / Oblique (25°–65°)"
    )
  ) %>%
  filter(!is.na(altitude_tier))

# ==============================================================================
# Figure 2: 2D Bias Phase-Space Surface (Pitch vs Roll, Split by Camera & Altitude)
# Description: Binned 2D phase-space heatmaps (2.5° x 2.5° attitude cells) of
#   mean operational length measurement percent bias across the pitch-roll
#   envelope, faceted by camera pod (Port vs. Starboard) and altitude tier.
# ==============================================================================
p_surface <- ggplot(plot_data, aes(x = roll_adj, y = pitch, z = bias_pct)) +
  stat_summary_2d(fun = mean, binwidth = c(2.5, 2.5)) +
  facet_grid(camera ~ altitude_tier, drop = FALSE) +
  scale_fill_gradient2(
    low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
    midpoint = 0, limits = c(-25, 25), oob = squish,
    name = "Mean % Bias"
  ) +
  scale_x_continuous(breaks = seq(-30, 30, by = 15), limits = c(-35, 35)) +
  scale_y_continuous(breaks = seq(-30, 30, by = 15), limits = c(-35, 35)) +
  coord_fixed() +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = expression(paste("Roll Angle (", theta[roll], ", degrees)")),
    y = expression(paste("Pitch Angle (", theta[pitch], ", degrees)"))
  )

ggsave(
  file.path(FIGURE_DIR, "Figure_2_Bias_Surface.png"),
  p_surface,
  width = 11,
  height = 6.5,
  dpi = 300,
  create.dir = TRUE
)

# ==============================================================================
# Figure 3: Single-Axis Transects
# Description: Isolated single-axis measurement percent bias responses along
#   near-neutral pitch sweeps (|Roll| <= 5°) and roll sweeps (|Pitch| <= 5°)
#   across altitude tiers. Points show detection-level mean bias and curves
#   show segment-weighted cubic spline GAM smooths with 95% confidence bands
#   comparing Port and Starboard cameras.
# ==============================================================================
tol <- 5.0

det_summary <- plot_data %>%
  group_by(across(any_of(c("imagename", "img_key", "camera", "altitude_tier", "pitch", "roll_adj")))) %>%
  summarize(
    bias_pct = mean(bias_pct, na.rm = TRUE),
    n_segs   = n(),
    .groups  = "drop"
  )

transect_data <- bind_rows(
  det_summary %>%
    filter(abs(roll_adj) <= tol) %>%
    mutate(axis = paste0("Pitch Sweep (|Roll| <= ", tol, "°)"), angle = pitch),
  det_summary %>%
    filter(abs(pitch) <= tol) %>%
    mutate(axis = paste0("Roll Sweep (|Pitch| <= ", tol, "°)"), angle = roll_adj)
)

if (nrow(transect_data) > 0) {
  p_transect <- ggplot(transect_data, aes(x = angle, y = bias_pct, color = camera)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
    geom_point(alpha = 0.25, size = 0.8) +
    geom_smooth(
      aes(weight = n_segs),
      method = "gam",
      formula = y ~ s(x, bs = "cs", k = 4),
      se = TRUE,
      linewidth = 0.8
    ) +
    facet_grid(axis ~ altitude_tier, drop = TRUE) +
    scale_color_manual(values = c("Port" = "#1b9e77", "Starboard" = "#d95f02"), name = "Camera") +
    guides(color = guide_legend(override.aes = list(size = 3, alpha = 1))) +
    coord_cartesian(ylim = c(-25, 25)) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.background = element_rect(fill = "grey92", color = "grey40"),
      strip.text = element_text(face = "bold")
    ) +
    labs(
      x = "Attitude Excursion Angle",
      y = " % Bias"
    )
  
  ggsave(
    file.path(FIGURE_DIR, "Figure_3_Single_Axis_Transects.png"),
    p_transect,
    width = 11,
    height = 6.5,
    dpi = 300,
    create.dir = TRUE
  )
}

# ==============================================================================
# Figure 4: Target Orientation (theta) Bias
# Description: Operational measurement percent bias across folded target segment
#   orientation angles (0°–90° in image space), faceted by camera pod and
#   altitude tier. Points (subsampled for display) are colored by alignment
#   class (longitudinal, diagonal, or transverse), overlaid with a segment-
#   weighted cyclic GAM smooth (black line) across the orientation range.
# ==============================================================================
theta_bins <- plot_data %>%
  mutate(theta_bin = pmin(89.5, floor(theta) + 0.5)) %>%
  group_by(camera, altitude_tier, orientation_class, theta_bin) %>%
  summarize(
    bias_pct = mean(bias_pct, na.rm = TRUE),
    n_segs   = n(),
    .groups  = "drop"
  )

set.seed(42)
pts_sample <- plot_data %>%
  slice_sample(n = min(30000L, nrow(plot_data)))

p_orientation <- ggplot(pts_sample, aes(x = theta, y = bias_pct, color = orientation_class)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  geom_point(alpha = 0.25, size = 0.8) +
  geom_smooth(
    data = theta_bins,
    aes(x = theta_bin, y = bias_pct, weight = n_segs, group = 1),
    method = "gam",
    formula = y ~ s(x, bs = "cc", k = 10),
    color = "black",
    se = TRUE,
    linewidth = 0.85
  ) +
  facet_grid(camera ~ altitude_tier, drop = TRUE) +
  scale_color_manual(
    values = c(
      "Longitudinal / Along-Track (0°–25°)" = "#d95f02",
      "Transverse / Across-Track (65°–90°)" = "#1b9e77",
      "Diagonal / Oblique (25°–65°)"        = "#7570b3"
    ),
    name = "Target Alignment"
  ) +
  guides(color = guide_legend(override.aes = list(size = 4, alpha = 1))) +
  scale_x_continuous(breaks = seq(0, 90, by = 15), limits = c(0, 90)) +
  coord_cartesian(ylim = c(-25, 25)) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = expression(paste("Segment Orientation Angle (", theta, ", degrees)")),
    y = " % Bias"
  )

ggsave(
  file.path(FIGURE_DIR, "Figure_4_Orientation_Bias.png"),
  p_orientation,
  width = 11,
  height = 6.5,
  dpi = 300,
  create.dir = TRUE
)

# ==============================================================================
# Figure 5: Bias Histograms (All Cleaned Data)
# Description: Frequency distribution of operational length measurement percent
#   bias (0.5% bin width) across 100% of cleaned segments (full ±35° attitude
#   envelope), faceted by camera pod and altitude tier. Dashed vertical lines
#   mark zero bias, solid black lines mark tier means, and panel annotations
#   report exact mean, standard deviation, and total segment count (n).
# ==============================================================================
message("\n>>> Generating Figure 5 bias histograms across all cleaned segments...")

bin_w <- 0.5  # 0.5% bias bin width

hist_stats <- plot_data %>%
  group_by(camera, altitude_tier) %>%
  summarize(
    n_segs      = n(),
    mean_bias   = mean(bias_pct),
    sd_bias     = sd(bias_pct),
    median_bias = median(bias_pct),
    iqr_bias    = IQR(bias_pct),
    stat_label  = sprintf("Mean: %+.2f%%\nSD: %.2f%%\nn = %s",
                          mean(bias_pct), sd(bias_pct), format(n(), big.mark = ",")),
    .groups     = "drop"
  )

print(as.data.frame(hist_stats %>% select(-stat_label)), row.names = FALSE)

hist_bins <- plot_data %>%
  mutate(bias_bin = (floor(bias_pct / bin_w) + 0.5) * bin_w) %>%
  group_by(camera, altitude_tier, bias_bin) %>%
  summarize(n = n(), .groups = "drop_last") %>%
  mutate(pct_segs = (n / sum(n)) * 100) %>%
  ungroup() %>%
  filter(abs(bias_bin) <= 20)

p_hist <- ggplot(hist_bins, aes(x = bias_bin, y = pct_segs, fill = camera)) +
  geom_col(width = bin_w * 0.92, alpha = 0.85, color = NA) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey25", linewidth = 0.6) +
  geom_vline(
    data = hist_stats,
    aes(xintercept = mean_bias),
    linetype = "solid",
    color = "black",
    linewidth = 0.65
  ) +
  geom_text(
    data = hist_stats,
    aes(x = -19, y = Inf, label = stat_label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1.2,
    size = 3.1,
    color = "grey15"
  ) +
  facet_grid(camera ~ altitude_tier, drop = TRUE) +
  scale_fill_manual(
    values = c("Port" = "#1b9e77", "Starboard" = "#d95f02"),
    name = "Camera"
  ) +
  scale_x_continuous(breaks = seq(-20, 20, by = 5), limits = c(-20, 20)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  theme_bw(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    x = "% Bias",
    y = "Percentage of Segments (%)"
  )

ggsave(
  file.path(FIGURE_DIR, "Figure_5_Bias_Histograms.png"),
  p_hist,
  width = 11,
  height = 6.5,
  dpi = 300,
  create.dir = TRUE
)