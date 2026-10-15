#!/usr/bin/env Rscript

# Batch-convert .rda/.RData files to Parquet for packaging in era_py.
#
# Usage:
#   Rscript data-raw/convert_farr_data_to_parquet.R [source_data_dir] [dest_dir] [dataset ...]
#
# Defaults (from repo root):
#   source_data_dir = ../farr/data
#   dest_dir        = src/era_data/_data

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
})

has_haven <- requireNamespace("haven", quietly = TRUE)

args <- commandArgs(trailingOnly = TRUE)
source_dir <- if (length(args) >= 1) args[[1]] else "../farr/data"
dest_dir <- if (length(args) >= 2) args[[2]] else "src/era_data/_data"
datasets <- if (length(args) >= 3) args[3:length(args)] else character()

if (!dir.exists(source_dir)) {
  stop(sprintf("Source directory does not exist: %s", source_dir))
}

dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)

normalize_for_parquet <- function(df) {
  metadata <- list(
    factor_levels = structure(list(), names = character()),
    original_classes = list()
  )

  for (col in names(df)) {
    x <- df[[col]]
    metadata$original_classes[[col]] <- class(x)

    if (is.factor(x)) {
      metadata$factor_levels[[col]] <- levels(x)
      df[[col]] <- as.character(x)
      next
    }

    if (has_haven && haven::is.labelled(x)) {
      df[[col]] <- haven::zap_labels(x)
      next
    }

    if (inherits(x, "POSIXct")) {
      df[[col]] <- as.POSIXct(x, tz = "UTC")
      next
    }
  }

  list(data = df, metadata = metadata)
}

rdata_files <- list.files(
  source_dir,
  pattern = "\\.(rda|RData)$",
  full.names = TRUE
)

if (length(datasets)) {
  available <- tools::file_path_sans_ext(basename(rdata_files))
  missing <- setdiff(datasets, available)
  if (length(missing)) stop("Unknown datasets: ", paste(missing, collapse = ", "))
  rdata_files <- rdata_files[available %in% datasets]
}

if (length(rdata_files) == 0) {
  stop(sprintf("No .rda/.RData files found in %s", source_dir))
}

message(sprintf("Found %d data files.", length(rdata_files)))

written <- 0L
skipped <- 0L

for (file in rdata_files) {
  e <- new.env(parent = emptyenv())
  loaded_names <- load(file, envir = e)

  for (obj_name in loaded_names) {
    obj <- e[[obj_name]]

    if (!is.data.frame(obj)) {
      message(sprintf("Skipping non-data.frame object %s from %s", obj_name, basename(file)))
      skipped <- skipped + 1L
      next
    }

    normalized <- normalize_for_parquet(obj)
    out_base <- file.path(dest_dir, obj_name)
    parquet_path <- paste0(out_base, ".parquet")
    meta_path <- paste0(out_base, ".meta.json")

    # Retain documentation and other provenance when refreshing data values.
    metadata <- if (file.exists(meta_path)) {
      read_json(meta_path, simplifyVector = FALSE)
    } else {
      list()
    }
    metadata$factor_levels <- normalized$metadata$factor_levels
    metadata$original_classes <- normalized$metadata$original_classes

    write_parquet(
      normalized$data,
      parquet_path,
      compression = "zstd"
    )

    write_json(
      metadata,
      meta_path,
      auto_unbox = TRUE,
      pretty = TRUE
    )

    message(sprintf("Wrote %s", parquet_path))
    written <- written + 1L
  }
}

message(sprintf("Done. Wrote %d parquet datasets (%d skipped).", written, skipped))
