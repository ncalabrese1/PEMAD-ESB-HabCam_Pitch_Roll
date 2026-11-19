# ==============================================================================
# Script: 02_fit_habcam_bias_gamm.R
# Project: HabCam Attitude Bias Quantification (WHOI Sundance Test Tank)
#
# Complete Stage 2 Pipeline (RE-GAMMs + Fixed-Effects GAMs + Tables + Figures):
#   - Section 1: Load QC-verified segment dataset & partition Port / Starboard.
#   - Section 2A: Fit 10,000-frame Random-Effects GAMMs (s(imagename, bs = "re"))
#                 for Port and Starboard (exact specification run overnight).
#   - Section 2B: Fit 100%-data Fixed-Effects GAMs (no random frame intercept,
#                 frame-equalizing weights) for Port and Starboard:
#                   (1) Additive Fixed-Effects GAM
#                   (2) Coupled Tilt x Sensor Interaction GAM (ti() terms)
#   - Section 3: Head-to-head out-of-sample validation comparison tables
#                (Uncorrected vs. RE-GAMM vs. Fixed GAM vs. Interaction GAM).
#   - Section 4: Generate all publication & comparison figures (Figures 1–6).
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(mgcv)
  library(parallel)
})

# ==============================================================================
# Configuration (Applied Identically to Port and Starboard)
# ==============================================================================
RE_GAMM_MAX_FRAMES  <- 10000L  # Exact stratified frame budget used for RE-GAMMs
EQUAL_FRAME_WEIGHTS <- TRUE    # Weight Fixed GAM segments by 1 / n_segs_in_frame (mean = 1)

DATA_DIR     <- if (dir.exists("Data")) "Data" else "data"
FIGURE_DIR   <- if (dir.exists("Figures")) "Figures" else "figures"
ALT_BOUNDS_M <- c(1.35, 3.80)

dir.create(DATA_DIR, showWarnings = FALSE)
dir.create(FIGURE_DIR, showWarnings = FALSE)

# ==============================================================================
# SECTION 1: Load Prepped Dataset & Partition Cameras to Keep RAM Minimal
# ==============================================================================
CLEAN_DATA_PATH <- file.path(DATA_DIR, "calibrated_segments_clean.rds")
if (!file.exists(CLEAN_DATA_PATH)) {
  CLEAN_DATA_PATH <- file.path(DATA_DIR, "sundance_calibrated_segments_clean.rds")
}
if (!file.exists(CLEAN_DATA_PATH)) {
  stop("Prepped dataset not found in ", DATA_DIR, ". Run Stage 1 extraction first.")
}

message(">>> Loading prepped, QC-verified segment dataset...")
clean_data <- readRDS(CLEAN_DATA_PATH) %>%
  select(
    camera, imagename, bias_pct,
    pitch, roll_adj, alt1, theta, pos_x, pos_y,
    est_len_mm, true_len_mm
  ) %>%
  mutate(
    camera    = trimws(tolower(as.character(camera))),
    imagename = as.character(imagename)
  )

total_cores <- detectCores(logical = TRUE)
if (is.na(total_cores) || total_cores < 1L) total_cores <- 4L
bam_threads <- min(4L, total_cores)

cat(sprintf("Loaded %s clean segments across %s frames | bam() threads: %d\n",
            format(nrow(clean_data), big.mark = ","),
            format(n_distinct(clean_data$imagename), big.mark = ","),
            bam_threads))

port_tmp_rds <- tempfile(pattern = "habcam_port_segs_", fileext = ".rds")
stbd_tmp_rds <- tempfile(pattern = "habcam_stbd_segs_", fileext = ".rds")
saveRDS(clean_data %>% filter(camera == "port") %>% select(-camera), port_tmp_rds)
saveRDS(clean_data %>% filter(camera == "starboard") %>% select(-camera), stbd_tmp_rds)

rm(clean_data)
gc()

# Helper to build the exact stratified 80% train / 20% holdout frame split
get_holdout_ids <- function(cam_df, seed = 42) {
  frame_meta <- cam_df %>%
    group_by(imagename) %>%
    summarize(
      alt1     = first(alt1),
      pitch    = first(pitch),
      roll_adj = first(roll_adj),
      .groups  = "drop"
    ) %>%
    mutate(
      alt_tier = cut(alt1, breaks = c(0, 1.75, 2.25, 2.75, 5.0), labels = FALSE),
      att_bin  = paste0(round(pitch / 10) * 10, "_", round(roll_adj / 10) * 10)
    )
  set.seed(seed)
  frame_meta %>%
    group_by(alt_tier, att_bin) %>%
    slice_sample(prop = 0.20) %>%
    pull(imagename)
}

# Helper for chunked out-of-sample prediction (keeps peak RAM < 1.5 GB)
predict_chunked <- function(model_obj, newdata_df, exclude_terms = NULL, batch_size = 250000L) {
  n_rows <- nrow(newdata_df)
  preds  <- numeric(n_rows)
  if (!is.null(exclude_terms) && "imagename" %in% names(model_obj$var.summary)) {
    re_lvls <- levels(model_obj$var.summary$imagename)
  }
  for (start_idx in seq(1L, n_rows, by = batch_size)) {
    end_idx <- min(start_idx + batch_size - 1L, n_rows)
    idx_seq <- start_idx:end_idx
    chunk   <- newdata_df[idx_seq, ]
    if (!is.null(exclude_terms)) {
      if ("imagename" %in% names(model_obj$var.summary)) {
        chunk$imagename <- factor(re_lvls[1], levels = re_lvls)
      }
      preds[idx_seq] <- as.numeric(
        predict(model_obj, newdata = chunk, exclude = exclude_terms, discrete = FALSE)
      )
    } else {
      preds[idx_seq] <- as.numeric(
        predict(model_obj, newdata = chunk, discrete = FALSE)
      )
    }
  }
  preds
}

# ==============================================================================
# SECTION 2: Fit Camera-Specific Random-Effects GAMMs
#   - Fits separate Gaussian GAMMs (mgcv::bam, fREML) for Port and Starboard
#     across a stratified sample of up to 10,000 training frames per camera.
#   - Predicts segment measurement bias (%) from:
#       1. 3D attitude-altitude tensor smooth: te(pitch, roll_adj, alt1, k = c(5,5,4))
#       2. Folded target orientation spline:   s(theta, bs = "cr", k = 10)
#       3. 2D focal-plane sensor tensor:       te(pos_x, pos_y, k = c(8,8))
#       4. Frame-level random intercept:       s(imagename, bs = "re")
#   - Evaluates out-of-sample population-level predictions (excluding frame REs)
#     on the 20% unseen holdout frames.
# ==============================================================================
fit_and_validate_re_gamm <- function(cam_name, cam_rds_path, n_threads,
                                     max_train_frames = 10000L, seed = 42) {
  cat("\n========================================================================\n")
  cat(sprintf("   FITTING & VALIDATING CAMERA MODEL: %s (%d THREADS)\n",
              toupper(cam_name), n_threads))
  cat("========================================================================\n")
  
  cam_df  <- readRDS(cam_rds_path)
  val_ids <- get_holdout_ids(cam_df, seed = seed)
  
  val_df   <- cam_df %>% filter(imagename %in% val_ids)
  train_df <- cam_df %>% filter(!imagename %in% val_ids)
  
  # Stratified training frame budget to keep s(imagename, bs = "re") within 16 GB RAM
  if (is.finite(max_train_frames)) {
    train_meta <- train_df %>%
      group_by(imagename) %>%
      summarize(
        alt1     = first(alt1),
        pitch    = first(pitch),
        roll_adj = first(roll_adj),
        .groups  = "drop"
      ) %>%
      mutate(
        alt_tier = cut(alt1, breaks = c(0, 1.75, 2.25, 2.75, 5.0), labels = FALSE),
        att_bin  = paste0(round(pitch / 10) * 10, "_", round(roll_adj / 10) * 10)
      )
    
    if (nrow(train_meta) > max_train_frames) {
      keep_prop <- max_train_frames / nrow(train_meta)
      set.seed(seed + 1L)
      keep_ids <- train_meta %>%
        group_by(alt_tier, att_bin) %>%
        slice_sample(prop = keep_prop) %>%
        pull(imagename)
      train_df <- train_df %>% filter(imagename %in% keep_ids)
    }
    rm(train_meta)
  }
  
  # Park unseen validation set on disk during fitting
  val_tmp_rds <- tempfile(pattern = "regamm_val_", fileext = ".rds")
  saveRDS(val_df, val_tmp_rds)
  rm(cam_df, val_df)
  gc()
  
  train_df$imagename <- factor(train_df$imagename)
  
  cat(sprintf("[%s] Training set: %s segments (%s frames) | Holdout set: %s unseen frames\n",
              toupper(cam_name),
              format(nrow(train_df), big.mark = ","),
              format(nlevels(train_df$imagename), big.mark = ","),
              format(length(val_ids), big.mark = ",")))
  
  gamm_formula <- bias_pct ~
    te(pitch, roll_adj, alt1, k = c(5, 5, 4), d = c(1, 1, 1)) +
    s(theta, bs = "cr", k = 10) +
    te(pos_x, pos_y, k = c(8, 8)) +
    s(imagename, bs = "re")
  
  t0 <- Sys.time()
  re_gamm_model <- bam(
    formula    = gamm_formula,
    data       = train_df,
    method     = "fREML",
    discrete   = TRUE,
    nthreads   = n_threads,
    chunk.size = 65536,
    gc.level   = 2,
    control    = gam.control(trace = TRUE, maxit = 25)
  )
  
  fit_mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  cat(sprintf("\n[%s] MODEL CONVERGED in %.2f minutes!\n", toupper(cam_name), fit_mins))
  
  model_out_path <- file.path(DATA_DIR, sprintf("habcam_bias_gamm_%s.rds", cam_name))
  saveRDS(re_gamm_model, model_out_path)
  cat(sprintf("[%s] Saved model object to: %s\n", toupper(cam_name), model_out_path))
  
  print(summary(re_gamm_model))
  
  # Smooth term effect magnitudes (SD in % excluding frame random intercept)
  sub_idx    <- seq(1L, nrow(train_df), by = 10L)
  term_preds <- predict(re_gamm_model, newdata = train_df[sub_idx, ],
                        type = "terms", exclude = "s(imagename)", discrete = FALSE)
  term_sds   <- round(apply(term_preds, 2, sd), 3)
  
  rm(train_df, term_preds)
  gc()
  
  # Chunked Out-of-Sample Prediction on 100% of Unseen Validation Frames
  val_df <- readRDS(val_tmp_rds)
  unlink(val_tmp_rds)
  
  cat(sprintf("[%s] Predicting onto %s unseen validation segments (%s frames)...\n",
              toupper(cam_name),
              format(nrow(val_df), big.mark = ","),
              format(length(val_ids), big.mark = ",")))
  
  pred_re <- predict_chunked(re_gamm_model, val_df, exclude_terms = "s(imagename)")
  rm(re_gamm_model)
  gc()
  
  cat(sprintf("\n[%s] Smooth Term Effect Magnitudes (SD in %%):\n", toupper(cam_name)))
  print(term_sds)
  
  val_evaluated <- val_df %>%
    mutate(
      camera             = cam_name,
      pred_bias          = pred_re,
      corrected_est_mm   = est_len_mm / (1 + pred_bias / 100),
      corrected_bias_pct = ((corrected_est_mm - true_len_mm) / true_len_mm) * 100,
      raw_error_mm       = est_len_mm - true_len_mm,
      corrected_error_mm = corrected_est_mm - true_len_mm
    )
  
  cat(sprintf("[%s] Out-of-Sample RMSE: %.2f%% -> %.2f%% | MdAE: %.2f%% -> %.2f%%\n",
              toupper(cam_name),
              sqrt(mean(val_evaluated$bias_pct^2)),
              sqrt(mean(val_evaluated$corrected_bias_pct^2)),
              median(abs(val_evaluated$bias_pct)),
              median(abs(val_evaluated$corrected_bias_pct))))
  
  rm(val_df)
  gc()
  
  return(val_evaluated)
}

re_val_port <- fit_and_validate_re_gamm("port", port_tmp_rds, bam_threads,
                                        max_train_frames = RE_GAMM_MAX_FRAMES, seed = 42)
re_val_stbd <- fit_and_validate_re_gamm("starboard", stbd_tmp_rds, bam_threads,
                                        max_train_frames = RE_GAMM_MAX_FRAMES, seed = 42)

re_val_all <- bind_rows(re_val_port, re_val_stbd)
saveRDS(re_val_all, file.path(DATA_DIR, "habcam_validation_evaluated_clean.rds"))

summarize_regamm <- function(df, label) {
  df %>%
    summarize(
      Camera          = label,
      Val_Frames      = n_distinct(imagename),
      Val_Segments    = n(),
      Raw_RMSE_pct    = round(sqrt(mean(bias_pct^2)), 2),
      GAMM_RMSE_pct   = round(sqrt(mean(corrected_bias_pct^2)), 2),
      RMSE_Drop_pct   = round((1 - sqrt(mean(corrected_bias_pct^2)) / sqrt(mean(bias_pct^2))) * 100, 1),
      Raw_MAE_pct     = round(mean(abs(bias_pct)), 2),
      GAMM_MAE_pct    = round(mean(abs(corrected_bias_pct)), 2),
      Raw_MdAE_pct    = round(median(abs(bias_pct)), 2),
      GAMM_MdAE_pct   = round(median(abs(corrected_bias_pct)), 2),
      Within_3pct_Raw = round(mean(abs(bias_pct) <= 3.0) * 100, 1),
      Within_3pct_New = round(mean(abs(corrected_bias_pct) <= 3.0) * 100, 1),
      Within_5pct_Raw = round(mean(abs(bias_pct) <= 5.0) * 100, 1),
      Within_5pct_New = round(mean(abs(corrected_bias_pct) <= 5.0) * 100, 1)
    )
}

regamm_summary_tbl <- bind_rows(
  summarize_regamm(re_val_port, "Port (Left)"),
  summarize_regamm(re_val_stbd, "Starboard (Right)"),
  summarize_regamm(re_val_all,  "Combined Stereo")
)
write_csv(regamm_summary_tbl, file.path(DATA_DIR, "output_gamm_validation_summary_by_camera.csv"))

rm(re_val_port, re_val_stbd)
gc()

# ==============================================================================
# SECTION 3: Fit Camera-Specific Additive Fixed-Effects GAMs
#   - Fits separate Gaussian GAMs (mgcv::bam, fREML, no frame random intercept)
#     for Port and Starboard across 100% of training segments, using
#     frame-equalizing weights (1 / n_segs_in_frame, normalized to mean = 1).
#   - Models measurement bias (%) as an additive sum of three main effects:
#       1. 3D attitude-altitude tensor smooth: te(pitch, roll_adj, alt1, k = c(6,6,4))
#       2. Folded target orientation spline:   s(theta, bs = "cr", k = 10)
#       3. 2D focal-plane sensor tensor:       te(pos_x, pos_y, k = c(10,10))
#   - Evaluates out-of-sample predictions on the 20% unseen holdout frames.
# ==============================================================================
t_stage2b <- Sys.time()

fit_additive_gam_camera <- function(cam_name, cam_rds_path, n_threads,
                                    equal_frame_weights = TRUE, seed = 42) {
  cat("\n========================================================================\n")
  cat(sprintf("   FITTING ADDITIVE FIXED-EFFECTS GAM (100%% DATA): %s (%d THREADS)\n",
              toupper(cam_name), n_threads))
  cat("========================================================================\n")
  
  cam_df  <- readRDS(cam_rds_path)
  val_ids <- get_holdout_ids(cam_df, seed = seed)
  
  val_df   <- cam_df %>% filter(imagename %in% val_ids)
  train_df <- cam_df %>% filter(!imagename %in% val_ids)
  rm(cam_df); gc()
  
  # Frame-equalizing weights so every stereo photo contributes equally (mean = 1)
  if (equal_frame_weights) {
    train_df <- train_df %>%
      group_by(imagename) %>%
      mutate(obs_weight = 1 / n()) %>%
      ungroup() %>%
      mutate(obs_weight = obs_weight / mean(obs_weight))
  } else {
    train_df$obs_weight <- 1.0
  }
  
  cat(sprintf("[%s] Training Additive GAM on 100%% of train data: %s segments (%s frames)\n",
              toupper(cam_name),
              format(nrow(train_df), big.mark = ","),
              format(n_distinct(train_df$imagename), big.mark = ",")))
  
  t0 <- Sys.time()
  fixed_gam <- bam(
    formula = bias_pct ~
      te(pitch, roll_adj, alt1, k = c(6, 6, 4)) +
      s(theta, bs = "cr", k = 10) +
      te(pos_x, pos_y, k = c(10, 10)),
    data       = train_df,
    weights    = obs_weight,
    method     = "fREML",
    discrete   = TRUE,
    nthreads   = n_threads,
    chunk.size = 65536,
    gc.level   = 2,
    control    = gam.control(trace = TRUE, maxit = 25)
  )
  cat(sprintf("\n[%s] Additive Fixed GAM converged in %.1f seconds.\n",
              toupper(cam_name), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  print(summary(fixed_gam))
  
  sub_idx  <- seq(1L, nrow(train_df), by = 10L)
  term_sds <- round(apply(predict(fixed_gam, newdata = train_df[sub_idx, ],
                                  type = "terms", discrete = FALSE), 2, sd), 3)
  cat(sprintf("[%s] Additive Fixed GAM Smooth Term SDs (%%):\n", toupper(cam_name)))
  print(term_sds)
  
  saveRDS(fixed_gam, file.path(DATA_DIR, sprintf("habcam_bias_fixed_gam_%s.rds", cam_name)))
  rm(train_df); gc()
  
  # Predict onto 100% of unseen validation frames
  pred_fgam <- predict_chunked(fixed_gam, val_df)
  rm(fixed_gam); gc()
  
  val_df %>%
    mutate(
      camera         = cam_name,
      fgam_pred_bias = pred_fgam,
      fgam_est_mm    = est_len_mm / (1 + fgam_pred_bias / 100),
      fgam_bias_pct  = ((fgam_est_mm - true_len_mm) / true_len_mm) * 100
    )
}

fgam_val_port <- fit_additive_gam_camera("port", port_tmp_rds, bam_threads,
                                         equal_frame_weights = EQUAL_FRAME_WEIGHTS, seed = 42)
fgam_val_stbd <- fit_additive_gam_camera("starboard", stbd_tmp_rds, bam_threads,
                                         equal_frame_weights = EQUAL_FRAME_WEIGHTS, seed = 42)

fgam_val_all <- bind_rows(fgam_val_port, fgam_val_stbd)
rm(fgam_val_port, fgam_val_stbd); gc()

# ==============================================================================
# SECTION 4: Fit Camera-Specific Coupled Tilt x Sensor Interaction GAMs
#   - Fits separate Gaussian GAMs (mgcv::bam, fREML, no frame random intercept)
#     for Port and Starboard across 100% of training segments, using
#     frame-equalizing weights (1 / n_segs_in_frame, normalized to mean = 1).
#   - Extends the additive fixed-effects GAM with functional ANOVA tensor
#     interactions to capture cross-frame perspective foreshortening gradients
#     induced when the camera sled pitches or rolls:
#       1. 3D attitude-altitude tensor smooth: te(pitch, roll_adj, alt1, k = c(6,6,4))
#       2. 2D focal-plane sensor tensor:       te(pos_x, pos_y, k = c(10,10))
#       3. Pitch x focal-plane interaction:    ti(pitch, pos_x, pos_y, k = c(4,5,5))
#       4. Roll x focal-plane interaction:     ti(roll_adj, pos_x, pos_y, k = c(4,5,5))
#       5. Folded target orientation spline:   s(theta, bs = "cr", k = 10)
#   - Evaluates out-of-sample predictions on the 20% unseen holdout frames and
#     combines results with Section 2B for head-to-head model comparison.
# ==============================================================================
fit_interaction_gam_camera <- function(cam_name, cam_rds_path, n_threads,
                                       equal_frame_weights = TRUE, seed = 42) {
  cat("\n========================================================================\n")
  cat(sprintf("   FITTING TILT x SENSOR INTERACTION GAM (100%% DATA): %s (%d THREADS)\n",
              toupper(cam_name), n_threads))
  cat("========================================================================\n")
  
  cam_df  <- readRDS(cam_rds_path)
  val_ids <- get_holdout_ids(cam_df, seed = seed)
  
  val_df   <- cam_df %>% filter(imagename %in% val_ids)
  train_df <- cam_df %>% filter(!imagename %in% val_ids)
  rm(cam_df); gc()
  
  # Frame-equalizing weights so every stereo photo contributes equally (mean = 1)
  if (equal_frame_weights) {
    train_df <- train_df %>%
      group_by(imagename) %>%
      mutate(obs_weight = 1 / n()) %>%
      ungroup() %>%
      mutate(obs_weight = obs_weight / mean(obs_weight))
  } else {
    train_df$obs_weight <- 1.0
  }
  
  cat(sprintf("[%s] Training Interaction GAM on 100%% of train data: %s segments (%s frames)\n",
              toupper(cam_name),
              format(nrow(train_df), big.mark = ","),
              format(n_distinct(train_df$imagename), big.mark = ",")))
  
  t1 <- Sys.time()
  int_gam <- bam(
    formula = bias_pct ~
      te(pitch, roll_adj, alt1, k = c(6, 6, 4)) +
      te(pos_x, pos_y, k = c(10, 10)) +
      ti(pitch, pos_x, pos_y, k = c(4, 5, 5)) +
      ti(roll_adj, pos_x, pos_y, k = c(4, 5, 5)) +
      s(theta, bs = "cr", k = 10),
    data       = train_df,
    weights    = obs_weight,
    method     = "fREML",
    discrete   = TRUE,
    nthreads   = n_threads,
    chunk.size = 65536,
    gc.level   = 2,
    control    = gam.control(trace = TRUE, maxit = 25)
  )
  cat(sprintf("\n[%s] Tilt x Sensor Interaction GAM converged in %.1f seconds.\n",
              toupper(cam_name), as.numeric(difftime(Sys.time(), t1, units = "secs"))))
  print(summary(int_gam))
  
  sub_idx  <- seq(1L, nrow(train_df), by = 10L)
  term_sds <- round(apply(predict(int_gam, newdata = train_df[sub_idx, ],
                                  type = "terms", discrete = FALSE), 2, sd), 3)
  cat(sprintf("[%s] Tilt x Sensor Interaction GAM Smooth Term SDs (%%):\n", toupper(cam_name)))
  print(term_sds)
  
  saveRDS(int_gam, file.path(DATA_DIR, sprintf("habcam_bias_interaction_gam_%s.rds", cam_name)))
  rm(train_df); gc()
  
  # Predict onto 100% of unseen validation frames
  pred_intgam <- predict_chunked(int_gam, val_df)
  rm(int_gam); gc()
  
  val_df %>%
    transmute(
      camera           = cam_name,
      imagename        = imagename,
      intgam_pred_bias = pred_intgam,
      intgam_est_mm    = est_len_mm / (1 + intgam_pred_bias / 100),
      intgam_bias_pct  = ((intgam_est_mm - true_len_mm) / true_len_mm) * 100
    )
}

intgam_val_port <- fit_interaction_gam_camera("port", port_tmp_rds, bam_threads,
                                              equal_frame_weights = EQUAL_FRAME_WEIGHTS, seed = 42)
unlink(port_tmp_rds)

intgam_val_stbd <- fit_interaction_gam_camera("starboard", stbd_tmp_rds, bam_threads,
                                              equal_frame_weights = EQUAL_FRAME_WEIGHTS, seed = 42)
unlink(stbd_tmp_rds)

intgam_val_all <- bind_rows(intgam_val_port, intgam_val_stbd)
rm(intgam_val_port, intgam_val_stbd); gc()

stopifnot(
  nrow(fgam_val_all) == nrow(intgam_val_all),
  identical(fgam_val_all$imagename, intgam_val_all$imagename)
)

val_eval_all <- fgam_val_all %>%
  mutate(
    intgam_pred_bias = intgam_val_all$intgam_pred_bias,
    intgam_est_mm    = intgam_val_all$intgam_est_mm,
    intgam_bias_pct  = intgam_val_all$intgam_bias_pct
  )

rm(fgam_val_all, intgam_val_all); gc()

# ==============================================================================
# SECTION 4: Head-to-Head Comparison (RE-GAMM vs. Additive GAM vs. Interaction GAM)
# ==============================================================================
if (!exists("re_val_all")) {
  re_saved_path <- file.path(DATA_DIR, "habcam_validation_evaluated_clean.rds")
  if (!file.exists(re_saved_path)) {
    stop("Saved RE-GAMM validation file not found: ", re_saved_path)
  }
  message(">>> Loading saved 10,000-frame RE-GAMM validation predictions from disk...")
  re_val_all <- readRDS(re_saved_path) %>%
    mutate(camera = trimws(tolower(as.character(camera))))
}

re_val_aligned <- bind_rows(
  re_val_all %>% filter(camera == "port"),
  re_val_all %>% filter(camera == "starboard")
)
stopifnot(nrow(val_eval_all) == nrow(re_val_aligned))

val_eval_all$re_pred_bias <- re_val_aligned$pred_bias
val_eval_all$re_bias_pct  <- re_val_aligned$corrected_bias_pct

rm(re_val_all, re_val_aligned); gc()
saveRDS(val_eval_all, file.path(DATA_DIR, "habcam_validation_all_models.rds"))

summarize_models <- function(df, label) {
  frame_df <- df %>%
    group_by(imagename) %>%
    summarize(
      raw_f    = mean(bias_pct),
      re_f     = mean(re_bias_pct),
      fgam_f   = mean(fgam_bias_pct),
      intgam_f = mean(intgam_bias_pct),
      .groups  = "drop"
    )
  
  tibble(
    Camera                = label,
    Val_Frames            = n_distinct(df$imagename),
    Val_Segments          = nrow(df),
    Raw_Seg_RMSE          = round(sqrt(mean(df$bias_pct^2)), 3),
    RE_GAMM_Seg_RMSE      = round(sqrt(mean(df$re_bias_pct^2)), 3),
    Fixed_GAM_Seg_RMSE    = round(sqrt(mean(df$fgam_bias_pct^2)), 3),
    Int_GAM_Seg_RMSE      = round(sqrt(mean(df$intgam_bias_pct^2)), 3),
    Raw_Frame_RMSE        = round(sqrt(mean(frame_df$raw_f^2)), 3),
    RE_GAMM_Frame_RMSE    = round(sqrt(mean(frame_df$re_f^2)), 3),
    Fixed_GAM_Frame_RMSE  = round(sqrt(mean(frame_df$fgam_f^2)), 3),
    Int_GAM_Frame_RMSE    = round(sqrt(mean(frame_df$intgam_f^2)), 3),
    RE_GAMM_MdAE          = round(median(abs(df$re_bias_pct)), 3),
    Fixed_GAM_MdAE        = round(median(abs(df$fgam_bias_pct)), 3),
    Int_GAM_MdAE          = round(median(abs(df$intgam_bias_pct)), 3),
    Within_5pct_Raw       = round(mean(abs(df$bias_pct) <= 5.0) * 100, 1),
    Within_5pct_RE_GAMM   = round(mean(abs(df$re_bias_pct) <= 5.0) * 100, 1),
    Within_5pct_Fixed_GAM = round(mean(abs(df$fgam_bias_pct) <= 5.0) * 100, 1),
    Within_5pct_Int_GAM   = round(mean(abs(df$intgam_bias_pct) <= 5.0) * 100, 1)
  )
}

model_comp_table <- bind_rows(
  summarize_models(val_eval_all %>% filter(camera == "port"),      "Port (Left)"),
  summarize_models(val_eval_all %>% filter(camera == "starboard"), "Starboard (Right)"),
  summarize_models(val_eval_all,                                   "Combined Stereo")
)

cat("\n========================================================================\n")
cat("   OUT-OF-SAMPLE COMPARISON: RE-GAMM vs. FIXED GAM vs. INTERACTION GAM  \n")
cat("========================================================================\n")
print(model_comp_table, width = Inf)
write_csv(model_comp_table, file.path(DATA_DIR, "output_model_comparison_gamm_vs_gams.csv"))

# Altitude-Tier Validation Breakdown (Fixed-Effects GAM)
summary_by_tier <- val_eval_all %>%
  mutate(
    Camera = recode(camera, "port" = "Port", "starboard" = "Starboard"),
    Altitude_Tier = cut(
      alt1,
      breaks = c(ALT_BOUNDS_M[1], 1.75, 2.25, 2.75, ALT_BOUNDS_M[2]),
      labels = c("1.5 m", "2.0 m", "2.5 m", "3.0 m"),
      include.lowest = TRUE
    )
  ) %>%
  filter(!is.na(Altitude_Tier)) %>%
  group_by(Camera, Altitude_Tier) %>%
  summarize(
    Val_Frames      = n_distinct(imagename),
    Val_Segments    = n(),
    Raw_Mean_pct    = round(mean(bias_pct), 2),
    FGAM_Mean_pct   = round(mean(fgam_bias_pct), 2),
    IntGAM_Mean_pct = round(mean(intgam_bias_pct), 2),
    Raw_RMSE_pct    = round(sqrt(mean(bias_pct^2)), 2),
    RE_RMSE_pct     = round(sqrt(mean(re_bias_pct^2)), 2),
    FGAM_RMSE_pct   = round(sqrt(mean(fgam_bias_pct^2)), 2),
    IntGAM_RMSE_pct = round(sqrt(mean(intgam_bias_pct^2)), 2),
    .groups         = "drop"
  )

cat("\n========================================================================\n")
cat("          OUT-OF-SAMPLE VALIDATION BY CAMERA & ALTITUDE TIER            \n")
cat("========================================================================\n")
print(summary_by_tier, width = Inf)
write_csv(summary_by_tier, file.path(DATA_DIR, "output_validation_summary_by_tier.csv"))

# ==============================================================================
# SECTION 4: Generate Comparison Tables
# ==============================================================================

message("\n>>> Generating publication-ready summary tables (Tables 3–6)...")

rm(val_plot_df); gc()

TABLE_DIR <- "Tables"
dir.create(TABLE_DIR, showWarnings = FALSE)

export_pub_table <- function(df, base_name) {
  write_csv(df, file.path(TABLE_DIR, paste0(base_name, ".csv")))
}

# ------------------------------------------------------------------------------
# Table 3: In-Sample Model Fit & Smooth Term Complexity
# Description: Summarizes in-sample fit metrics (Adjusted R2, deviance explained,
#   residual variance), standardized level-flight baseline bias (0 deg tilt,
#   2.0 m altitude, sensor center [0.5, 0.5], 45 deg orientation), and smooth-term
#   effective degrees of freedom (EDF) across Model 1 (Random-Effects GAMM),
#   Model 2 (Additive Fixed GAM), and Model 3 (Tilt x Sensor Interaction GAM).
# ------------------------------------------------------------------------------
extract_fit_row <- function(rds_file, cam_label, model_label, is_re = FALSE) {
  m <- readRDS(file.path(DATA_DIR, rds_file))
  
  n_obs    <- length(m$y)
  edf_sum  <- sum(m$edf)
  w        <- as.numeric(m$prior.weights)
  mean_y   <- sum(w * m$y) / sum(w)
  rss      <- sum(w * (m$y - m$fitted.values)^2)
  tss      <- sum(w * (m$y - mean_y)^2)
  adj_r2   <- 1 - (rss / tss) * ((n_obs - 1) / (n_obs - edf_sum))
  dev_expl <- (m$null.deviance - m$deviance) / m$null.deviance
  
  # Evaluate physical baseline bias (%) at level flight (0 deg tilt, 2.0 m alt, sensor center)
  ref_pt <- data.frame(
    pitch    = 0,
    roll_adj = 0,
    alt1     = 2.0,
    theta    = 45,
    pos_x    = 0.5,
    pos_y    = 0.5
  )
  if (is_re) {
    re_lvls <- levels(m$var.summary$imagename)
    ref_pt$imagename <- factor(re_lvls[1], levels = re_lvls)
    pr0 <- predict(m, newdata = ref_pt, exclude = "s(imagename)", se.fit = TRUE, discrete = FALSE)
  } else {
    pr0 <- predict(m, newdata = ref_pt, se.fit = TRUE, discrete = FALSE)
  }
  
  edf_vec <- sapply(m$smooth, function(sm) sum(m$edf[sm$first.para:sm$last.para]))
  names(edf_vec) <- sapply(m$smooth, `[[`, "label")
  
  get_edf <- function(pattern) {
    idx <- grep(pattern, names(edf_vec), fixed = TRUE)
    if (length(idx) == 0) return("-")
    paste(sprintf("%.2f", edf_vec[idx]), collapse = " / ")
  }
  
  out <- tibble(
    `Camera`                    = cam_label,
    `Model Variant`             = model_label,
    `Train Segments`            = format(n_obs, big.mark = ","),
    `Train Frames`              = if (is_re) format(nlevels(m$var.summary$imagename), big.mark = ",") else "All",
    `Adj R2`                    = sprintf("%.3f", adj_r2),
    `Deviance Explained (%)`    = sprintf("%.1f%%", dev_expl * 100),
    `Residual Variance (%2)`    = sprintf("%.2f", m$sig2),
    `Level Flight Baseline (%)` = sprintf("%+.2f (%.2f)", as.numeric(pr0$fit), as.numeric(pr0$se.fit)),
    `Attitude EDF`              = get_edf("te(pitch"),
    `Sensor Plane EDF`          = get_edf("te(pos_x"),
    `Tilt / Sensor EDF`         = get_edf("ti("),
    `Orientation EDF`           = get_edf("s(theta)"),
    `Total EDF`                 = sprintf("%.1f", edf_sum)
  )
  rm(m); gc()
  out
}

table3 <- bind_rows(
  extract_fit_row("habcam_bias_gamm_port.rds",                "Port",      "1. Random-Effects GAMM",           is_re = TRUE),
  extract_fit_row("habcam_bias_fixed_gam_port.rds",           "Port",      "2. Additive Fixed-Effects GAM",    is_re = FALSE),
  extract_fit_row("habcam_bias_interaction_gam_port.rds",     "Port",      "3. Tilt x Sensor Interaction GAM", is_re = FALSE),
  extract_fit_row("habcam_bias_gamm_starboard.rds",           "Starboard", "1. Random-Effects GAMM",           is_re = TRUE),
  extract_fit_row("habcam_bias_fixed_gam_starboard.rds",      "Starboard", "2. Additive Fixed-Effects GAM",    is_re = FALSE),
  extract_fit_row("habcam_bias_interaction_gam_starboard.rds", "Starboard", "3. Tilt x Sensor Interaction GAM", is_re = FALSE)
)

print(table3, width = Inf)
export_pub_table(table3, "Table_3_Model_Fit_Summary")

# ------------------------------------------------------------------------------
# Table 4: Out-of-Sample Accuracy (% Bias & Millimeter Length)
# Description: Evaluates out-of-sample relative (%) and absolute (mm) accuracy
#   across Port, Starboard, and Combined validation datasets, comparing True
#   Values, Raw Measurements, RE-GAMM (Model 1), Additive Fixed GAM (Model 2),
#   and Tilt x Sensor Interaction GAM (Model 3).
# ------------------------------------------------------------------------------
val_df <- readRDS(file.path(DATA_DIR, "habcam_validation_all_models.rds")) %>%
  mutate(
    Camera    = recode(trimws(tolower(as.character(camera))),
                       "port" = "Port", "starboard" = "Starboard"),
    re_est_mm = est_len_mm / (1 + re_pred_bias / 100)
  )

eval_state_metrics <- function(df, cam_lbl, state_lbl, pct_col, mm_col) {
  pct_err <- df[[pct_col]]
  mm_est  <- df[[mm_col]]
  mm_err  <- mm_est - df$true_len_mm
  
  frame_rmse <- df %>%
    group_by(imagename) %>%
    summarize(fb = mean(.data[[pct_col]]), .groups = "drop") %>%
    summarize(r = sqrt(mean(fb^2))) %>%
    pull(r)
  
  tibble(
    `Camera`              = cam_lbl,
    `Calibration`         = state_lbl,
    `Validation Frames`   = format(n_distinct(df$imagename), big.mark = ","),
    `Validation Segments` = format(nrow(df), big.mark = ","),
    `Mean Bias (%)`       = sprintf("%+.2f%%", mean(pct_err)),
    `Segment RMSE (%)`    = sprintf("%.2f%%", sqrt(mean(pct_err^2))),
    `Frame RMSE (%)`      = sprintf("%.2f%%", frame_rmse),
    `MdAE (%)`            = sprintf("%.2f%%", median(abs(pct_err))),
    `Segments <3% Bias`   = sprintf("%.1f%%", mean(abs(pct_err) <= 3.0) * 100),
    `Segments <5% Bias`   = sprintf("%.1f%%", mean(abs(pct_err) <= 5.0) * 100),
    `Mean Length (mm)`    = sprintf("%.1f", mean(mm_est)),
    `Mean Diff (mm)`      = sprintf("%+.2f", mean(mm_err)),
    `Length RMSE (mm)`    = sprintf("%.2f", sqrt(mean(mm_err^2))),
    `Length MAE (mm)`     = sprintf("%.2f", mean(abs(mm_err)))
  )
}

build_cam_block <- function(df, cam_lbl) {
  df <- df %>% mutate(zero_pct = 0.0)
  bind_rows(
    eval_state_metrics(df, cam_lbl, "True Values",            "zero_pct",        "true_len_mm"),
    eval_state_metrics(df, cam_lbl, "Raw Measurements",       "bias_pct",        "est_len_mm"),
    eval_state_metrics(df, cam_lbl, "1. Random-Effects GAMM", "re_bias_pct",     "re_est_mm"),
    eval_state_metrics(df, cam_lbl, "2. Additive Fixed GAM",  "fgam_bias_pct",   "fgam_est_mm"),
    eval_state_metrics(df, cam_lbl, "3. Tilt x Sensor GAM",   "intgam_bias_pct", "intgam_est_mm")
  )
}

table4 <- bind_rows(
  build_cam_block(val_df %>% filter(Camera == "Port"),      "Port"),
  build_cam_block(val_df %>% filter(Camera == "Starboard"), "Starboard"),
  build_cam_block(val_df,                                   "Combined")
)

print(table4, width = Inf)
export_pub_table(table4, "Table_4_OOS_Validation_Performance")

# ------------------------------------------------------------------------------
# Table 5: Operational Performance by Altitude Tier & Vehicle Tilt Regime
# Description: Stratifies out-of-sample Mean Bias (%), Segment RMSE (%), and
#   Length RMSE (mm) across four operational altitude tiers (1.5, 2.0, 2.5, 3.0 m)
#   and two attitude regimes (Near-Level <10 deg vs. High-Tilt >=10 deg) for
#   Raw, GAMM, FGAM, and IntGAM estimates.
# ------------------------------------------------------------------------------
fmt_pct <- function(x) sprintf("%.2f%%", round(x, 2) + 0)

summarize_stratum <- function(df, stratum_type) {
  df %>%
    group_by(Camera, Stratum) %>%
    summarize(
      `Stratum Type`     = stratum_type,
      `Frames`           = format(n_distinct(imagename), big.mark = ","),
      `Segments`         = format(n(), big.mark = ","),
      `Raw Bias (%)`     = fmt_pct(mean(bias_pct)),
      `Raw RMSE (%)`     = fmt_pct(sqrt(mean(bias_pct^2))),
      `Raw RMSE (mm)`    = round(sqrt(mean((est_len_mm - true_len_mm)^2)), 2),
      `GAMM Bias (%)`    = fmt_pct(mean(re_bias_pct)),
      `GAMM RMSE (%)`    = fmt_pct(sqrt(mean(re_bias_pct^2))),
      `GAMM RMSE (mm)`   = round(sqrt(mean((re_est_mm - true_len_mm)^2)), 2),
      `FGAM Bias (%)`    = fmt_pct(mean(fgam_bias_pct)),
      `FGAM RMSE (%)`    = fmt_pct(sqrt(mean(fgam_bias_pct^2))),
      `FGAM RMSE (mm)`   = round(sqrt(mean((fgam_est_mm - true_len_mm)^2)), 2),
      `IntGAM Bias (%)`  = fmt_pct(mean(intgam_bias_pct)),
      `IntGAM RMSE (%)`  = fmt_pct(sqrt(mean(intgam_bias_pct^2))),
      `IntGAM RMSE (mm)` = round(sqrt(mean((intgam_est_mm - true_len_mm)^2)), 2),
      .groups            = "drop"
    ) %>%
    select(`Camera`, `Stratum Type`, `Stratum`, everything())
}

tier_df <- val_df %>%
  mutate(Stratum = as.character(cut(
    alt1,
    breaks = c(ALT_BOUNDS_M[1], 1.75, 2.25, 2.75, ALT_BOUNDS_M[2]),
    labels = c("1.5 m Tier", "2.0 m Tier", "2.5 m Tier", "3.0 m Tier"),
    include.lowest = TRUE
  ))) %>%
  filter(!is.na(Stratum))

tilt_df <- val_df %>%
  mutate(Stratum = ifelse(abs(pitch) < 10 & abs(roll_adj) < 10,
                          "Near-Level (|Pitch| & |Roll| < 10°)",
                          "High-Tilt (|Pitch| or |Roll| >= 10°)"))

table5 <- bind_rows(
  summarize_stratum(tier_df, "Altitude"),
  summarize_stratum(tilt_df, "Attitude")
)

print(table5, width = Inf)

# Write two-row grouped header CSV matching the layout in Excel
t5_path  <- file.path(TABLE_DIR, "Table_5_Operational_Regime_Breakdown.csv")
top_hdr  <- c("", "", "", "", "", "Raw", "", "", "GAMM", "", "", "FGAM", "", "", "IntGAM", "", "")
sub_hdr  <- c("Camera", "Stratum Type", "Stratum", "Frames", "Segments",
              rep(c("Bias (%)", "RMSE (%)", "RMSE (mm)"), 4))

writeLines(
  c(paste(top_hdr, collapse = ","), paste(sub_hdr, collapse = ",")),
  con = t5_path
)
write_csv(table5, t5_path, append = TRUE, col_names = FALSE)

# ------------------------------------------------------------------------------
# Table 6: Size-Stratified Millimeter Accuracy Across True Board Grid Lengths
# Description: Evaluates out-of-sample Mean Length (mm), Mean Diff (mm), and
#   Length RMSE (mm) across the six primary calibration checkerboard segment
#   lengths (80.0, 113.1, 160.0, 178.9, 226.3, and 240.0 mm; n > 1,000) for
#   Raw, GAMM, FGAM, and IntGAM estimates.
# ------------------------------------------------------------------------------
keep_targets <- val_df %>%
  mutate(target_mm = round(true_len_mm, 1)) %>%
  count(target_mm) %>%
  filter(n > 1000) %>%
  pull(target_mm)

table6 <- val_df %>%
  mutate(target_mm = round(true_len_mm, 1)) %>%
  filter(target_mm %in% keep_targets) %>%
  group_by(Camera, target_mm) %>%
  summarize(
    `Segments`         = format(n(), big.mark = ","),
    `Raw Mean (mm)`    = round(mean(est_len_mm), 1),
    `Raw Diff (mm)`    = round(mean(est_len_mm - true_len_mm), 2) + 0,
    `Raw RMSE (mm)`    = round(sqrt(mean((est_len_mm - true_len_mm)^2)), 2),
    `GAMM Mean (mm)`   = round(mean(re_est_mm), 1),
    `GAMM Diff (mm)`   = round(mean(re_est_mm - true_len_mm), 2) + 0,
    `GAMM RMSE (mm)`   = round(sqrt(mean((re_est_mm - true_len_mm)^2)), 2),
    `FGAM Mean (mm)`   = round(mean(fgam_est_mm), 1),
    `FGAM Diff (mm)`   = round(mean(fgam_est_mm - true_len_mm), 2) + 0,
    `FGAM RMSE (mm)`   = round(sqrt(mean((fgam_est_mm - true_len_mm)^2)), 2),
    `IntGAM Mean (mm)` = round(mean(intgam_est_mm), 1),
    `IntGAM Diff (mm)` = round(mean(intgam_est_mm - true_len_mm), 2) + 0,
    `IntGAM RMSE (mm)` = round(sqrt(mean((intgam_est_mm - true_len_mm)^2)), 2),
    .groups            = "drop"
  ) %>%
  rename(`True (mm)` = target_mm)

print(table6, width = Inf)

# Write two-row grouped header CSV matching Table 5's layout in Excel
t6_path  <- file.path(TABLE_DIR, "Table_6_Size_Stratified_Millimeter_Accuracy.csv")
top_hdr6 <- c("", "", "", "Raw", "", "", "GAMM", "", "", "FGAM", "", "", "IntGAM", "", "")
sub_hdr6 <- c("Camera", "True (mm)", "Segments",
              rep(c("Mean (mm)", "Diff (mm)", "RMSE (mm)"), 4))

writeLines(
  c(paste(top_hdr6, collapse = ","), paste(sub_hdr6, collapse = ",")),
  con = t6_path
)
write_csv(table6, t6_path, append = TRUE, col_names = FALSE)

rm(val_df, tier_df, tilt_df); gc()


# ==============================================================================
# SECTION 5: Generate All Figures (Figures 6–13)
# ==============================================================================
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

if (!exists("DATA_DIR"))     DATA_DIR     <- if (dir.exists("Data")) "Data" else "data"
if (!exists("FIGURE_DIR"))   FIGURE_DIR   <- if (dir.exists("Figures")) "Figures" else "figures"
if (!exists("ALT_BOUNDS_M")) ALT_BOUNDS_M <- c(1.35, 3.80)
dir.create(FIGURE_DIR, showWarnings = FALSE)

message("\n>>> Generating all publication and validation figures (Figures 6–13)...")

if (!exists("val_eval_all")) {
  val_eval_all <- readRDS(file.path(DATA_DIR, "habcam_validation_all_models.rds"))
}

val_plot_df <- val_eval_all %>%
  mutate(
    camera_label = recode(trimws(tolower(as.character(camera))),
                          "port" = "Port", "starboard" = "Starboard"),
    altitude_tier = cut(
      alt1,
      breaks = c(ALT_BOUNDS_M[1], 1.75, 2.25, 2.75, ALT_BOUNDS_M[2]),
      labels = c("1.5 m", "2.0 m", "2.5 m", "3.0 m"),
      include.lowest = TRUE
    )
  ) %>%
  filter(!is.na(altitude_tier))

rm(val_eval_all); gc()

# ------------------------------------------------------------------------------
# Figure 6: Fitted Attitude-Altitude Surfaces Across Tiers (Interaction GAM)
# Description: Displays predicted percentage bias surfaces across roll and pitch
#   (±30 deg) at four discrete altitudes (1.5, 2.0, 2.5, 3.0 m) for Port and
#   Starboard cameras from Model 3 (evaluated at optical center [0.5, 0.5] and
#   45 deg orientation), masked to empirical flight trajectories (dist = 0.035).
# ------------------------------------------------------------------------------
ig_port <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_port.rds"))
ig_stbd <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_starboard.rds"))

att_grid <- expand.grid(
  pitch    = seq(-30, 30, length.out = 101),
  roll_adj = seq(-30, 30, length.out = 101),
  alt1     = c(1.5, 2.0, 2.5, 3.0),
  theta    = 45,
  pos_x    = 0.5,
  pos_y    = 0.5
) %>%
  mutate(
    altitude_tier = factor(sprintf("%.1f m", alt1),
                           levels = c("1.5 m", "2.0 m", "2.5 m", "3.0 m"))
  )

surf_port <- att_grid %>%
  mutate(camera_label = "Port",
         pred_bias = as.numeric(predict(ig_port, newdata = att_grid, discrete = FALSE)))

surf_stbd <- att_grid %>%
  mutate(camera_label = "Starboard",
         pred_bias = as.numeric(predict(ig_stbd, newdata = att_grid, discrete = FALSE)))

# Deduplicate empirical pitch-roll footprint per camera & tier to mask extrapolation
obs_footprint <- val_plot_df %>%
  mutate(
    r_round = round(roll_adj * 4) / 4,
    p_round = round(pitch * 4) / 4
  ) %>%
  distinct(camera_label, altitude_tier, r_round, p_round)

fig6_surf <- bind_rows(surf_port, surf_stbd) %>%
  group_by(camera_label, altitude_tier) %>%
  group_modify(~ {
    fp <- obs_footprint %>%
      filter(camera_label == .y$camera_label,
             altitude_tier == .y$altitude_tier)
    too_far <- mgcv::exclude.too.far(
      g1   = .x$roll_adj,
      g2   = .x$pitch,
      d1   = fp$r_round,
      d2   = fp$p_round,
      dist = 0.035
    )
    .x %>% mutate(pred_bias = ifelse(too_far, NA_real_, pred_bias))
  }) %>%
  ungroup()

p_fig6 <- ggplot(fig6_surf, aes(x = roll_adj, y = pitch)) +
  geom_tile(aes(fill = pred_bias)) +
  geom_contour(aes(z = pred_bias), color = "black", alpha = 0.40,
               breaks = seq(-15, 15, by = 2.5), na.rm = TRUE) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey35", linewidth = 0.35) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey35", linewidth = 0.35) +
  facet_grid(camera_label ~ altitude_tier) +
  scale_fill_gradient2(
    low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
    midpoint = 0, limits = c(-15, 15), oob = scales::squish,
    na.value = "transparent",
    name = "Predicted\nBias (%)"
  ) +
  coord_fixed(xlim = c(-30, 30), ylim = c(-30, 30)) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid       = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text       = element_text(face = "bold")
  ) +
  labs(
    x = expression(paste("Roll Angle (", theta[roll], ", degrees)")),
    y = expression(paste("Pitch Angle (", theta[pitch], ", degrees)"))
  )

ggsave(file.path(FIGURE_DIR, "Figure_6_Fitted_Attitude_Altitude_Surfaces.png"),
       p_fig6, width = 11.5, height = 6.2, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 7: Dynamic Focal-Plane Distortion Across Level, Pitched & Rolled States
# Description: Shows spatial bias surfaces across normalized sensor coordinates
#   (X, Y in [0, 1]^2; origin bottom-left) from Model 3 at 2.0 m altitude and
#   45 deg orientation under Level (0/0 deg), Pitched (+15/0 deg), and Rolled
#   (0/+15 deg) flight states, masked to the stereo-rectified overlap footprint.
# ------------------------------------------------------------------------------
if (!exists("ig_port")) ig_port <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_port.rds"))
if (!exists("ig_stbd")) ig_stbd <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_starboard.rds"))

att_states <- tibble(
  attitude_state = factor(
    c("Level Flight (0° Pitch, 0° Roll)",
      "Pitched (+15° Pitch, 0° Roll)",
      "Rolled (0° Pitch, +15° Roll)"),
    levels = c("Level Flight (0° Pitch, 0° Roll)",
               "Pitched (+15° Pitch, 0° Roll)",
               "Rolled (0° Pitch, +15° Roll)")
  ),
  pitch    = c(0, 15, 0),
  roll_adj = c(0, 0, 15)
)

xy_grid <- tidyr::crossing(
  pos_x = seq(0.01, 0.99, length.out = 101),
  pos_y = seq(0.01, 0.99, length.out = 101),
  att_states
) %>%
  mutate(alt1 = 2.0, theta = 45)

xy_port <- xy_grid %>%
  mutate(camera_label = "Port",
         pred_bias = as.numeric(predict(ig_port, newdata = xy_grid, discrete = FALSE)))

xy_stbd <- xy_grid %>%
  mutate(camera_label = "Starboard",
         pred_bias = as.numeric(predict(ig_stbd, newdata = xy_grid, discrete = FALSE)))

# Mask extreme sensor corners where stereo overlap never occurs
xy_footprint <- val_plot_df %>%
  mutate(
    x_round = round(pos_x * 50) / 50,
    y_round = round(pos_y * 50) / 50
  ) %>%
  distinct(camera_label, x_round, y_round)

fig7_surf <- bind_rows(xy_port, xy_stbd) %>%
  group_by(camera_label) %>%
  group_modify(~ {
    fp <- xy_footprint %>% filter(camera_label == .y$camera_label)
    too_far <- mgcv::exclude.too.far(
      g1   = .x$pos_x,
      g2   = .x$pos_y,
      d1   = fp$x_round,
      d2   = fp$y_round,
      dist = 0.04
    )
    .x %>% mutate(pred_bias = ifelse(too_far, NA_real_, pred_bias))
  }) %>%
  ungroup()

p_fig7 <- ggplot(fig7_surf, aes(x = pos_x, y = pos_y)) +
  geom_tile(aes(fill = pred_bias)) +
  geom_contour(aes(z = pred_bias), color = "black", alpha = 0.40,
               breaks = seq(-6, 6, by = 1), na.rm = TRUE) +
  facet_grid(camera_label ~ attitude_state) +
  scale_fill_gradient2(
    low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
    midpoint = 0, limits = c(-6, 6), oob = scales::squish,
    na.value = "transparent",
    name = "Focal-Plane\nBias (%)"
  ) +
  coord_fixed(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid       = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text       = element_text(face = "bold")
  ) +
  labs(
    x = "Normalized Horizontal Sensor Coordinate (X)",
    y = "Normalized Vertical Sensor Coordinate (Y)"
  )

ggsave(file.path(FIGURE_DIR, "Figure_7_Dynamic_Sensor_Plane_Distortion.png"),
       p_fig7, width = 11.2, height = 6.2, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 8: Target Orientation Effect (0°–90° Folded Domain, Interaction GAM)
# Description: Plots conditional predicted bias (%) and 95% confidence ribbons
#   across folded segment orientation (0-90 deg) from Model 3 at 2.0 m altitude,
#   0 deg pitch/roll, and sensor center (0.5, 0.5), with dashed vertical lines and
#   points marking discrete checkerboard target angles.
# ------------------------------------------------------------------------------
if (!exists("DATA_DIR"))   DATA_DIR   <- if (dir.exists("Data")) "Data" else "data"
if (!exists("FIGURE_DIR")) FIGURE_DIR <- if (dir.exists("Figures")) "Figures" else "figures"
if (!exists("ig_port"))    ig_port    <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_port.rds"))
if (!exists("ig_stbd"))    ig_stbd    <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_starboard.rds"))
if (!exists("val_plot_df")) {
  val_plot_df <- readRDS(file.path(DATA_DIR, "habcam_validation_all_models.rds"))
}

# Extract the discrete measurement angles (or tight empirical angle clusters)
theta_rounded <- round(val_plot_df$theta, 1)
theta_counts  <- sort(table(theta_rounded), decreasing = TRUE)

if (length(theta_counts) <= 25) {
  meas_angles <- sort(as.numeric(names(theta_counts)))
} else {
  theta_bins <- val_plot_df %>%
    mutate(t_bin = round(theta)) %>%
    count(t_bin) %>%
    mutate(pct = n / sum(n)) %>%
    filter(pct >= 0.01) %>%
    arrange(desc(n))
  
  meas_angles <- numeric(0)
  for (ang in theta_bins$t_bin) {
    if (all(abs(ang - meas_angles) >= 2.5)) {
      meas_angles <- c(meas_angles, ang)
    }
  }
  meas_angles <- sort(meas_angles)
}

theta_rug_df <- tibble(theta = sort(unique(round(val_plot_df$theta, 2))))

# Smooth prediction curve across 0°–90°
theta_grid <- expand.grid(
  theta    = seq(0, 90, length.out = 181),
  pitch    = 0,
  roll_adj = 0,
  alt1     = 2.0,
  pos_x    = 0.5,
  pos_y    = 0.5
)

pr_p <- predict(ig_port, newdata = theta_grid, se.fit = TRUE, discrete = FALSE)
pr_s <- predict(ig_stbd, newdata = theta_grid, se.fit = TRUE, discrete = FALSE)

theta_df <- bind_rows(
  theta_grid %>% mutate(camera_label = "Port",
                        fit = as.numeric(pr_p$fit), se = as.numeric(pr_p$se.fit)),
  theta_grid %>% mutate(camera_label = "Starboard",
                        fit = as.numeric(pr_s$fit), se = as.numeric(pr_s$se.fit))
)

# Exact predictions at the discrete measurement angles
pts_grid <- expand.grid(
  theta    = meas_angles,
  pitch    = 0,
  roll_adj = 0,
  alt1     = 2.0,
  pos_x    = 0.5,
  pos_y    = 0.5
)

theta_pts_df <- bind_rows(
  pts_grid %>% mutate(camera_label = "Port",
                      fit = as.numeric(predict(ig_port, newdata = pts_grid, discrete = FALSE))),
  pts_grid %>% mutate(camera_label = "Starboard",
                      fit = as.numeric(predict(ig_stbd, newdata = pts_grid, discrete = FALSE)))
)

p_fig8 <- ggplot(theta_df, aes(x = theta, y = fit, color = camera_label, fill = camera_label)) +
  geom_vline(xintercept = meas_angles, linetype = "dashed", color = "grey55",
             linewidth = 0.45, alpha = 0.75) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey30") +
  geom_ribbon(aes(ymin = fit - 1.96 * se, ymax = fit + 1.96 * se), alpha = 0.20, color = NA) +
  geom_line(linewidth = 1.0) +
  geom_point(data = theta_pts_df, size = 2.3) +
  geom_rug(data = theta_rug_df, aes(x = theta), sides = "b", color = "grey35",
           alpha = 0.65, length = unit(0.025, "npc"), inherit.aes = FALSE) +
  scale_color_manual(values = c("Port" = "#1b9e77", "Starboard" = "#d95f02"), name = "Camera") +
  scale_fill_manual(values  = c("Port" = "#1b9e77", "Starboard" = "#d95f02"), name = "Camera") +
  scale_x_continuous(breaks = seq(0, 90, by = 15), limits = c(0, 90)) +
  scale_y_continuous(expand = expansion(mult = c(0.08, 0.10))) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  ) +
  labs(
    x = expression(paste("Target Segment Orientation (", theta, ", degrees, 0°–90° folded)")),
    y = "Predicted Measurement Bias (%)"
  )

if (length(meas_angles) <= 12) {
  p_fig8 <- p_fig8 +
    geom_text(
      data = tibble(theta = meas_angles, label = sprintf("%.1f°", meas_angles)),
      aes(x = theta, y = Inf, label = label),
      inherit.aes = FALSE,
      vjust = 1.4, size = 2.7, color = "grey30"
    )
}

ggsave(file.path(FIGURE_DIR, "Figure_8_Target_Orientation_Effect.png"),
       p_fig8, width = 7.5, height = 4.5, dpi = 300)

# ------------------------------------------------------------------------------
# Figure 9: Out-of-Sample Segment-Level Bias Distributions (Raw vs. FGAM vs. IntGAM)
# Description: Compares out-of-sample segment-level percentage bias distributions
#   (0.5% bins, ±20%) across four operational altitude tiers (1.5-3.0 m) for
#   Uncorrected Raw (orange bars), Additive Fixed GAM (green step curve), and
#   Tilt x Sensor Interaction GAM (blue bars), annotated with Mean Bias and RMSE.
# ------------------------------------------------------------------------------
bin_w <- 0.5
x_lim <- 20

oos_seg_long <- val_plot_df %>%
  select(
    camera_label, altitude_tier,
    `Uncorrected`        = bias_pct,
    `Additive Fixed GAM` = fgam_bias_pct,
    `Tilt x Sensor GAM`  = intgam_bias_pct
  ) %>%
  pivot_longer(
    cols      = c(`Uncorrected`, `Additive Fixed GAM`, `Tilt x Sensor GAM`),
    names_to  = "State",
    values_to = "bias_val"
  ) %>%
  mutate(State = factor(State, levels = c("Uncorrected", "Additive Fixed GAM", "Tilt x Sensor GAM")))

stats_tbl <- oos_seg_long %>%
  group_by(camera_label, altitude_tier, State) %>%
  summarize(
    n_obs    = n(),
    mean_val = mean(bias_val),
    rmse_val = sqrt(mean(bias_val^2)),
    .groups  = "drop"
  )

annot_tbl <- stats_tbl %>%
  pivot_wider(
    id_cols     = c(camera_label, altitude_tier, n_obs),
    names_from  = State,
    values_from = c(mean_val, rmse_val)
  ) %>%
  mutate(
    label = sprintf(
      "n = %s segments\nRaw: %+.2f%% (RMSE %.2f%%)\nFGAM: %+.2f%% (RMSE %.2f%%)\nIntGAM: %+.2f%% (RMSE %.2f%%)",
      format(n_obs, big.mark = ","),
      `mean_val_Uncorrected`, `rmse_val_Uncorrected`,
      `mean_val_Additive Fixed GAM`, `rmse_val_Additive Fixed GAM`,
      `mean_val_Tilt x Sensor GAM`, `rmse_val_Tilt x Sensor GAM`
    )
  )

bins_tbl <- oos_seg_long %>%
  mutate(bias_bin = (floor(bias_val / bin_w) + 0.5) * bin_w) %>%
  group_by(camera_label, altitude_tier, State, bias_bin) %>%
  summarize(n = n(), .groups = "drop_last") %>%
  mutate(pct = (n / sum(n)) * 100) %>%
  ungroup() %>%
  filter(abs(bias_bin) <= x_lim)

bars_tbl <- bins_tbl %>% filter(State %in% c("Uncorrected", "Tilt x Sensor GAM"))

step_tbl <- bins_tbl %>%
  filter(State == "Additive Fixed GAM") %>%
  arrange(camera_label, altitude_tier, bias_bin) %>%
  group_by(camera_label, altitude_tier) %>%
  group_modify(~ bind_rows(.x, tail(.x, 1) %>% mutate(bias_bin = bias_bin + bin_w))) %>%
  ungroup()

p_fig9 <- ggplot(bars_tbl, aes(x = bias_bin, y = pct, fill = State)) +
  geom_col(position = "identity", width = bin_w * 0.92, alpha = 0.48, color = NA) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey25", linewidth = 0.6) +
  geom_vline(
    data = stats_tbl,
    aes(xintercept = mean_val, color = State, linetype = State),
    linewidth = 0.70
  ) +
  geom_step(
    data = step_tbl,
    aes(x = bias_bin - bin_w / 2, y = pct, color = State),
    direction = "hv",
    linewidth = 0.75,
    inherit.aes = FALSE
  ) +
  geom_label(
    data = annot_tbl,
    aes(x = -x_lim + 0.6, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1.10,
    size = 2.65,
    color = "grey15",
    fill = scales::alpha("white", 0.90)
  ) +
  facet_grid(camera_label ~ altitude_tier, drop = TRUE) +
  scale_fill_manual(
    values = c("Uncorrected" = "#D55E00", "Tilt x Sensor GAM" = "#0072B2"),
    name   = "Histogram Fill"
  ) +
  scale_color_manual(
    values = c("Uncorrected" = "#9E3D00", "Additive Fixed GAM" = "#009E73", "Tilt x Sensor GAM" = "#004C7A"),
    name   = "Mean / Step Curve"
  ) +
  scale_linetype_manual(
    values = c("Uncorrected" = "dashed", "Additive Fixed GAM" = "dotdash", "Tilt x Sensor GAM" = "solid"),
    name   = "Mean / Step Curve"
  ) +
  scale_x_continuous(breaks = seq(-x_lim, x_lim, by = 5), limits = c(-x_lim, x_lim)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.28))) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text       = element_text(face = "bold")
  ) +
  labs(
    x = "Measurement Bias (%)",
    y = "Percentage of Out-of-Sample Segments (%)"
  )

ggsave(file.path(FIGURE_DIR, "Figure_9_OOS_Segment_Bias_Combo_Histogram.png"),
       p_fig9, width = 11.5, height = 6.8, dpi = 300)

rm(oos_seg_long, stats_tbl, annot_tbl, bins_tbl, bars_tbl, step_tbl); gc()

# ------------------------------------------------------------------------------
# Figure 10: Out-of-Sample Segment-Level Bias Across Pitch & Roll Strata
# Description: Stratifies out-of-sample segment-level percentage bias (0.5% bins,
#   ±20%) across five mutually exclusive vehicle attitude regimes: Level Flight
#   (<5 deg), Moderate Tilt (5-10 deg), Pitch-Dominated (|Pitch| >= 10 deg),
#   Roll-Dominated (|Roll| >= 10 deg), and Compound Tilt (both >= 10 deg).
# ------------------------------------------------------------------------------
bin_w <- 0.5
x_lim <- 20

att_levels <- c(
  "Level Flight\n(|Pitch| & |Roll| < 5°)",
  "Moderate Tilt\n(5° ≤ |Tilt| < 10°)",
  "Pitch-Dominated\n(|Pitch| ≥ 10°, |Roll| < 10°)",
  "Roll-Dominated\n(|Pitch| < 10°, |Roll| ≥ 10°)",
  "Compound Tilt\n(|Pitch| & |Roll| ≥ 10°)"
)

oos_att_long <- val_plot_df %>%
  mutate(
    attitude_stratum = factor(
      case_when(
        abs(pitch) < 5   & abs(roll_adj) < 5   ~ att_levels[1],
        abs(pitch) < 10  & abs(roll_adj) < 10  ~ att_levels[2],
        abs(pitch) >= 10 & abs(roll_adj) < 10  ~ att_levels[3],
        abs(pitch) < 10  & abs(roll_adj) >= 10 ~ att_levels[4],
        TRUE                                   ~ att_levels[5]
      ),
      levels = att_levels
    )
  ) %>%
  select(
    camera_label, attitude_stratum,
    `Uncorrected`        = bias_pct,
    `Additive Fixed GAM` = fgam_bias_pct,
    `Tilt x Sensor GAM`  = intgam_bias_pct
  ) %>%
  pivot_longer(
    cols      = c(`Uncorrected`, `Additive Fixed GAM`, `Tilt x Sensor GAM`),
    names_to  = "State",
    values_to = "bias_val"
  ) %>%
  mutate(State = factor(State, levels = c("Uncorrected", "Additive Fixed GAM", "Tilt x Sensor GAM")))

stats_att_tbl <- oos_att_long %>%
  group_by(camera_label, attitude_stratum, State) %>%
  summarize(
    n_obs    = n(),
    mean_val = mean(bias_val),
    rmse_val = sqrt(mean(bias_val^2)),
    .groups  = "drop"
  )

annot_att_tbl <- stats_att_tbl %>%
  pivot_wider(
    id_cols     = c(camera_label, attitude_stratum, n_obs),
    names_from  = State,
    values_from = c(mean_val, rmse_val)
  ) %>%
  mutate(
    label = sprintf(
      "n = %s segs\nRaw: %+.2f%% (RMSE %.2f%%)\nFGAM: %+.2f%% (RMSE %.2f%%)\nIntGAM: %+.2f%% (RMSE %.2f%%)",
      format(n_obs, big.mark = ","),
      `mean_val_Uncorrected`, `rmse_val_Uncorrected`,
      `mean_val_Additive Fixed GAM`, `rmse_val_Additive Fixed GAM`,
      `mean_val_Tilt x Sensor GAM`, `rmse_val_Tilt x Sensor GAM`
    )
  )

bins_att_tbl <- oos_att_long %>%
  mutate(bias_bin = (floor(bias_val / bin_w) + 0.5) * bin_w) %>%
  group_by(camera_label, attitude_stratum, State, bias_bin) %>%
  summarize(n = n(), .groups = "drop_last") %>%
  mutate(pct = (n / sum(n)) * 100) %>%
  ungroup() %>%
  filter(abs(bias_bin) <= x_lim)

bars_att_tbl <- bins_att_tbl %>% filter(State %in% c("Uncorrected", "Tilt x Sensor GAM"))

step_att_tbl <- bins_att_tbl %>%
  filter(State == "Additive Fixed GAM") %>%
  arrange(camera_label, attitude_stratum, bias_bin) %>%
  group_by(camera_label, attitude_stratum) %>%
  group_modify(~ bind_rows(.x, tail(.x, 1) %>% mutate(bias_bin = bias_bin + bin_w))) %>%
  ungroup()

p_fig10 <- ggplot(bars_att_tbl, aes(x = bias_bin, y = pct, fill = State)) +
  geom_col(position = "identity", width = bin_w * 0.92, alpha = 0.48, color = NA) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey25", linewidth = 0.6) +
  geom_vline(
    data = stats_att_tbl,
    aes(xintercept = mean_val, color = State, linetype = State),
    linewidth = 0.70
  ) +
  geom_step(
    data = step_att_tbl,
    aes(x = bias_bin - bin_w / 2, y = pct, color = State),
    direction = "hv",
    linewidth = 0.75,
    inherit.aes = FALSE
  ) +
  geom_label(
    data = annot_att_tbl,
    aes(x = -x_lim + 0.6, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1.10,
    size = 2.40,
    color = "grey15",
    fill = scales::alpha("white", 0.90)
  ) +
  facet_grid(camera_label ~ attitude_stratum, drop = TRUE) +
  scale_fill_manual(
    values = c("Uncorrected" = "#D55E00", "Tilt x Sensor GAM" = "#0072B2"),
    name   = "Histogram Fill"
  ) +
  scale_color_manual(
    values = c("Uncorrected" = "#9E3D00", "Additive Fixed GAM" = "#009E73", "Tilt x Sensor GAM" = "#004C7A"),
    name   = "Mean / Step Curve"
  ) +
  scale_linetype_manual(
    values = c("Uncorrected" = "dashed", "Additive Fixed GAM" = "dotdash", "Tilt x Sensor GAM" = "solid"),
    name   = "Mean / Step Curve"
  ) +
  scale_x_continuous(breaks = seq(-x_lim, x_lim, by = 5), limits = c(-x_lim, x_lim)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.32))) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text       = element_text(face = "bold", size = 9.0)
  ) +
  labs(
    x = "Measurement Bias (%)",
    y = "Percentage of Out-of-Sample Segments (%)"
  )

ggsave(file.path(FIGURE_DIR, "Figure_10_OOS_Segment_Bias_Attitude_Strata.png"),
       p_fig10, width = 13.5, height = 6.8, dpi = 300)

rm(oos_att_long, stats_att_tbl, annot_att_tbl, bins_att_tbl, bars_att_tbl, step_att_tbl); gc()

# ------------------------------------------------------------------------------
# Figure 11: Side-by-Side 2.0 m Surface Comparison (Landscape: Camera x Model)
# Description: Compares fitted 2.0 m roll-pitch bias surfaces (±30 deg, at sensor
#   center [0.5, 0.5] and 45 deg orientation) across all three candidate models
#   (1. Random-Effects GAMM, 2. Additive Fixed-Effects GAM, and 3. Tilt x Sensor
#   Interaction GAM) for Port and Starboard cameras, demonstrating oversmoothing
#   in the RE-GAMM vs. consistent central surfaces in both fixed-effects GAMs.
# ------------------------------------------------------------------------------
if (file.exists(file.path(DATA_DIR, "habcam_bias_gamm_port.rds")) &&
    file.exists(file.path(DATA_DIR, "habcam_bias_gamm_starboard.rds"))) {
  
  grid_2m <- expand.grid(
    pitch    = seq(-30, 30, length.out = 101),
    roll_adj = seq(-30, 30, length.out = 101),
    alt1     = 2.0,
    theta    = 45,
    pos_x    = 0.5,
    pos_y    = 0.5
  )
  
  fg_port <- readRDS(file.path(DATA_DIR, "habcam_bias_fixed_gam_port.rds"))
  fg_stbd <- readRDS(file.path(DATA_DIR, "habcam_bias_fixed_gam_starboard.rds"))
  s_fg_p  <- grid_2m %>% mutate(Camera = "Port", Model = "2. Additive Fixed-Effects GAM",
                                pred_bias = as.numeric(predict(fg_port, newdata = grid_2m, discrete = FALSE)))
  s_fg_s  <- grid_2m %>% mutate(Camera = "Starboard", Model = "2. Additive Fixed-Effects GAM",
                                pred_bias = as.numeric(predict(fg_stbd, newdata = grid_2m, discrete = FALSE)))
  rm(fg_port, fg_stbd); gc()
  
  ig_port <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_port.rds"))
  ig_stbd <- readRDS(file.path(DATA_DIR, "habcam_bias_interaction_gam_starboard.rds"))
  s_ig_p  <- grid_2m %>% mutate(Camera = "Port", Model = "3. Tilt x Sensor Interaction GAM",
                                pred_bias = as.numeric(predict(ig_port, newdata = grid_2m, discrete = FALSE)))
  s_ig_s  <- grid_2m %>% mutate(Camera = "Starboard", Model = "3. Tilt x Sensor Interaction GAM",
                                pred_bias = as.numeric(predict(ig_stbd, newdata = grid_2m, discrete = FALSE)))
  rm(ig_port, ig_stbd); gc()
  
  re_port <- readRDS(file.path(DATA_DIR, "habcam_bias_gamm_port.rds"))
  lv_p    <- levels(re_port$var.summary$imagename)
  g2_re_p <- grid_2m %>% mutate(imagename = factor(lv_p[1], levels = lv_p))
  s_re_p  <- grid_2m %>% mutate(Camera = "Port", Model = "1. Random-Effects GAMM",
                                pred_bias = as.numeric(predict(re_port, newdata = g2_re_p,
                                                               exclude = "s(imagename)",
                                                               block.size = 2500, discrete = FALSE)))
  rm(re_port, g2_re_p); gc()
  
  re_stbd <- readRDS(file.path(DATA_DIR, "habcam_bias_gamm_starboard.rds"))
  lv_s    <- levels(re_stbd$var.summary$imagename)
  g2_re_s <- grid_2m %>% mutate(imagename = factor(lv_s[1], levels = lv_s))
  s_re_s  <- grid_2m %>% mutate(Camera = "Starboard", Model = "1. Random-Effects GAMM",
                                pred_bias = as.numeric(predict(re_stbd, newdata = g2_re_s,
                                                               exclude = "s(imagename)",
                                                               block.size = 2500, discrete = FALSE)))
  rm(re_stbd, g2_re_s); gc()
  
  # Mask unvisited 2.0 m attitude corners identically across all three models
  obs_fp_2m <- val_plot_df %>%
    filter(altitude_tier == "2.0 m") %>%
    mutate(
      r_round = round(roll_adj * 4) / 4,
      p_round = round(pitch * 4) / 4
    ) %>%
    distinct(camera_label, r_round, p_round)
  
  model_lvls <- c("1. Random-Effects GAMM",
                  "2. Additive Fixed-Effects GAM",
                  "3. Tilt x Sensor Interaction GAM")
  
  fig11_surf <- bind_rows(s_re_p, s_re_s, s_fg_p, s_fg_s, s_ig_p, s_ig_s) %>%
    mutate(Model = factor(Model, levels = model_lvls)) %>%
    group_by(Camera, Model) %>%
    group_modify(~ {
      fp <- obs_fp_2m %>% filter(camera_label == .y$Camera)
      too_far <- mgcv::exclude.too.far(
        g1   = .x$roll_adj,
        g2   = .x$pitch,
        d1   = fp$r_round,
        d2   = fp$p_round,
        dist = 0.04
      )
      .x %>% mutate(pred_bias = ifelse(too_far, NA_real_, pred_bias))
    }) %>%
    ungroup()
  
  rm(s_re_p, s_re_s, s_fg_p, s_fg_s, s_ig_p, s_ig_s, obs_fp_2m); gc()
  
  p_fig11 <- ggplot(fig11_surf, aes(x = roll_adj, y = pitch)) +
    geom_raster(aes(fill = pred_bias)) +
    geom_contour(aes(z = pred_bias), color = "black", alpha = 0.40,
                 breaks = seq(-15, 15, by = 2.5), na.rm = TRUE) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey35", linewidth = 0.35) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey35", linewidth = 0.35) +
    facet_grid(Camera ~ Model) +
    scale_fill_gradient2(
      low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
      midpoint = 0, limits = c(-15, 15), oob = scales::squish,
      na.value = "transparent",
      name = "Predicted\nBias (%)"
    ) +
    coord_fixed(xlim = c(-30, 30), ylim = c(-30, 30)) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid       = element_blank(),
      strip.background = element_rect(fill = "grey92", color = "grey40"),
      strip.text       = element_text(face = "bold")
    ) +
    labs(
      x = expression(paste("Roll Angle (", theta[roll], ", degrees)")),
      y = expression(paste("Pitch Angle (", theta[pitch], ", degrees)"))
    )
  
  ggsave(file.path(FIGURE_DIR, "Figure_11_GAMM_vs_GAM_Surface_Comparison_2m.png"),
         p_fig11, width = 11.8, height = 6.8, dpi = 300)
  rm(fig11_surf, p_fig11); gc()
}

# ------------------------------------------------------------------------------
# Figure 12: Size-Stratified Millimeter Error Across Board Grid Targets (80–240 mm)
# Description: Displays out-of-sample mean millimeter length error (Estimated -
#   True, mm) and interquartile range (25th-75th percentile whiskers) across the
#   six primary checkerboard target sizes (80.0 to 240.0 mm) for Raw, FGAM, and
#   IntGAM estimates, illustrating removal of size-proportional underestimation.
# ------------------------------------------------------------------------------
keep_targets <- val_plot_df %>%
  mutate(target_mm = round(true_len_mm, 1)) %>%
  count(target_mm) %>%
  filter(n > 1000) %>%
  pull(target_mm)

size_err_df <- val_plot_df %>%
  mutate(target_mm = round(true_len_mm, 1)) %>%
  filter(target_mm %in% keep_targets) %>%
  transmute(
    camera_label,
    target_mm,
    `Raw Measurements`   = est_len_mm - true_len_mm,
    `Additive Fixed GAM` = fgam_est_mm - true_len_mm,
    `Tilt x Sensor GAM`  = intgam_est_mm - true_len_mm
  ) %>%
  pivot_longer(cols = c(`Raw Measurements`, `Additive Fixed GAM`, `Tilt x Sensor GAM`),
               names_to = "Model", values_to = "err_mm") %>%
  mutate(Model = factor(Model, levels = c("Raw Measurements", "Additive Fixed GAM", "Tilt x Sensor GAM"))) %>%
  group_by(camera_label, target_mm, Model) %>%
  summarize(
    mean_mm = mean(err_mm),
    q25_mm  = quantile(err_mm, 0.25),
    q75_mm  = quantile(err_mm, 0.75),
    .groups = "drop"
  )

pd <- position_dodge(width = 4.5)

p_fig12 <- ggplot(size_err_df, aes(x = target_mm, y = mean_mm, color = Model, shape = Model)) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "grey25", linewidth = 0.65) +
  geom_errorbar(aes(ymin = q25_mm, ymax = q75_mm), position = pd, width = 3.5, linewidth = 0.65, alpha = 0.75) +
  geom_line(position = pd, linewidth = 0.85) +
  geom_point(position = pd, size = 2.6) +
  facet_wrap(~ camera_label) +
  scale_color_manual(
    values = c("Raw Measurements" = "#D55E00", "Additive Fixed GAM" = "#009E73", "Tilt x Sensor GAM" = "#0072B2"),
    name   = NULL
  ) +
  scale_shape_manual(
    values = c("Raw Measurements" = 16, "Additive Fixed GAM" = 17, "Tilt x Sensor GAM" = 15),
    name   = NULL
  ) +
  scale_x_continuous(breaks = c(80, 113.1, 160, 178.9, 226.3, 240),
                     labels = c("80.0", "113.1", "160.0", "178.9", "226.3", "240.0")) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text       = element_text(face = "bold")
  ) +
  labs(
    x = "True Calibration Target Length (mm)",
    y = "Measurement Difference: Estimated - True (mm)  [Mean & IQR]"
  )

ggsave(file.path(FIGURE_DIR, "Figure_12_Size_Stratified_Millimeter_Error.png"),
       p_fig12, width = 10.5, height = 5.0, dpi = 300)
rm(size_err_df); gc()

# ------------------------------------------------------------------------------
# Figure 13: Millimeter Error Distributions in Near-Level (<10°) vs. High-Tilt (>=10°)
# Description: Compares out-of-sample millimeter length error distributions
#   (1 mm bins, ±15 mm) in Near-Level (|Pitch| & |Roll| < 10 deg) versus
#   High-Tilt (|Pitch| or |Roll| >= 10 deg) flight regimes across Raw (shaded
#   area), Additive Fixed GAM (green line), and Tilt x Sensor GAM (blue line).
# ------------------------------------------------------------------------------
regime_mm_df <- val_plot_df %>%
  transmute(
    camera_label,
    Regime = factor(
      ifelse(abs(pitch) < 10 & abs(roll_adj) < 10,
             "Near-Level (|Pitch| & |Roll| < 10°)",
             "High-Tilt (|Pitch| or |Roll| >= 10°)"),
      levels = c("Near-Level (|Pitch| & |Roll| < 10°)",
                 "High-Tilt (|Pitch| or |Roll| >= 10°)")
    ),
    `Raw Measurements`   = est_len_mm - true_len_mm,
    `Additive Fixed GAM` = fgam_est_mm - true_len_mm,
    `Tilt x Sensor GAM`  = intgam_est_mm - true_len_mm
  ) %>%
  pivot_longer(cols = c(`Raw Measurements`, `Additive Fixed GAM`, `Tilt x Sensor GAM`),
               names_to = "Model", values_to = "err_mm") %>%
  mutate(Model = factor(Model, levels = c("Raw Measurements", "Additive Fixed GAM", "Tilt x Sensor GAM")))

regime_stats <- regime_mm_df %>%
  group_by(camera_label, Regime, Model) %>%
  summarize(
    n_obs   = n(),
    mean_mm = mean(err_mm),
    rmse_mm = sqrt(mean(err_mm^2)),
    .groups = "drop"
  )

regime_annot <- regime_stats %>%
  pivot_wider(
    id_cols     = c(camera_label, Regime, n_obs),
    names_from  = Model,
    values_from = c(mean_mm, rmse_mm)
  ) %>%
  mutate(
    label = sprintf(
      "n = %s segs\nRaw: %+.2f mm (RMSE %.2f mm)\nFGAM: %+.2f mm (RMSE %.2f mm)\nIntGAM: %+.2f mm (RMSE %.2f mm)",
      format(n_obs, big.mark = ","),
      `mean_mm_Raw Measurements`, `rmse_mm_Raw Measurements`,
      `mean_mm_Additive Fixed GAM`, `rmse_mm_Additive Fixed GAM`,
      `mean_mm_Tilt x Sensor GAM`, `rmse_mm_Tilt x Sensor GAM`
    )
  )

regime_bins <- regime_mm_df %>%
  mutate(err_bin = floor(err_mm) + 0.5) %>%
  filter(abs(err_bin) <= 15) %>%
  group_by(camera_label, Regime, Model, err_bin) %>%
  summarize(n = n(), .groups = "drop_last") %>%
  mutate(pct = (n / sum(n)) * 100) %>%
  ungroup()

rm(regime_mm_df); gc()

p_fig13 <- ggplot(regime_bins, aes(x = err_bin, y = pct, color = Model, fill = Model)) +
  geom_area(data = filter(regime_bins, Model == "Raw Measurements"),
            alpha = 0.28, position = "identity", linewidth = 0.7) +
  geom_line(data = filter(regime_bins, Model != "Raw Measurements"),
            linewidth = 0.95) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "grey20", linewidth = 0.65) +
  geom_label(
    data = regime_annot,
    aes(x = -14.5, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1.08,
    size = 2.7,
    fill = scales::alpha("white", 0.90)
  ) +
  facet_grid(camera_label ~ Regime) +
  scale_fill_manual(values = c("Raw Measurements" = "#D55E00"), guide = "none") +
  scale_color_manual(
    values = c("Raw Measurements" = "#D55E00", "Additive Fixed GAM" = "#009E73", "Tilt x Sensor GAM" = "#0072B2"),
    name   = NULL
  ) +
  scale_x_continuous(breaks = seq(-15, 15, by = 5)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.28))) +
  theme_bw(base_size = 11) +
  theme(
    legend.position  = "bottom",
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey92", color = "grey40"),
    strip.text       = element_text(face = "bold")
  ) +
  labs(
    x = "Measurement Difference: Estimated - True (mm)",
    y = "Percentage of Segments per 1 mm Bin (%)"
  )

ggsave(file.path(FIGURE_DIR, "Figure_13_Attitude_Regime_Millimeter_Error.png"),
       p_fig13, width = 11.2, height = 6.8, dpi = 300)