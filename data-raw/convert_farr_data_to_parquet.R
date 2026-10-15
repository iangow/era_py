#!/usr/bin/env Rscript

# Batch-convert .rda/.RData files to Parquet for packaging in era_py.
#
# Usage:
#   Rscript data-raw/convert_farr_data_to_parquet.R [source_data_dir] [dest_dir] [dataset ...]
# Add --check to verify existing files without writing them.
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
check_only <- "--check" %in% args
args <- args[args != "--check"]
source_dir <- if (length(args) >= 1) args[[1]] else "../farr/data"
dest_dir <- if (length(args) >= 2) args[[2]] else "src/era_data/_data"
datasets <- if (length(args) >= 3) args[3:length(args)] else character()

if (!dir.exists(source_dir)) {
  stop(sprintf("Source directory does not exist: %s", source_dir))
}

if (check_only) {
  if (!dir.exists(dest_dir)) stop("Destination directory does not exist: ", dest_dir)
} else {
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
}

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
    source <- list(file = basename(file), md5 = unname(tools::md5sum(file)))
    if (check_only) {
      if (!file.exists(parquet_path)) stop("Missing Parquet dataset: ", parquet_path)
      actual <- read_parquet(parquet_path, as_data_frame = TRUE)
      same <- all.equal(as.data.frame(actual), as.data.frame(normalized$data),
                        tolerance = 0)
      if (!isTRUE(same)) stop("Dataset differs from farr: ", obj_name, ": ",
                              paste(same, collapse = "; "))
      if (!identical(metadata$farr_source, source)) {
        stop("Source provenance differs from farr: ", obj_name, "; reconvert the dataset")
      }
      # Check type metadata too: loaders use it to restore R-originated types.
      expected_types <- fromJSON(toJSON(normalized$metadata, auto_unbox = TRUE),
                                simplifyVector = FALSE)
      if (!identical(metadata$original_classes, expected_types$original_classes) ||
          !identical(metadata$factor_levels, expected_types$factor_levels)) {
        stop("Type metadata differs from farr: ", obj_name)
      }
      message("Checked ", obj_name, ": values, types and source hash match farr")
      written <- written + 1L
      next
    }
    metadata$farr_source <- source
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

message(sprintf("Done. %s %d parquet datasets (%d skipped).",
                if (check_only) "Checked" else "Wrote", written, skipped))
