# Changelog

All notable changes to Micro Post will be documented in this file.

## [1.2.2] - 2026-04-27

### Changed

- **Ensemble-averaged MSD/MSAD now uses trajectory-length weighted averaging**:
  - Weight = (N_i - τ), the number of valid displacement pairs per trajectory at lag time τ
  - Longer trajectories with more reliable statistics contribute proportionally more to the ensemble mean and standard deviation
  - `Count` column in Summary now reports total valid displacement pairs instead of number of objects
  - Added `_weighted_mean_std()` static method in `MotionAnalyzer` for consistent weighted statistics

### Documentation

- Updated README.md and manual.html with new version (1.2.2) and release date (2026-04-27)
- Added weighted averaging methodology explanation to Changelog and user manual
- Added CHANGELOG.md

## [1.2.1] - 2026-01-13

### Fixed

- Optimized data processing algorithms
- Improved error handling in Excel export functionality

## [1.2.0] - 2026-01-13

### Fixed

- Fixed maximum displacement calculation to correctly compute trajectory bounding box dimensions (width and height) instead of displacement from origin

### Changed

- Enhanced Excel export formatting:
  - Unified Arial font (10pt) for all cells across all sheets
  - Bold formatting for header rows for better readability
  - Improved column width auto-adjustment (min: 10, max: 60, padding: 3)

## [1.1.0] - 2026-01-07

### Added

- Data loading and validation with parameter consistency checking
- Data filtering and merging by minimum tracking duration
- Interactive trajectory preview with zoom and pan capabilities
- Manual object exclusion by ID numbers
- Comprehensive motion analysis:
  - Displacement (X/Y and total)
  - Angular displacement with 180° wrapping
  - Instantaneous and lag-time averaged velocities
  - Mean Squared Displacement (MSD)
  - Mean Squared Angular Displacement (MSAD)
  - Trajectory ellipse fitting
  - Oscillation index (PCA-based)
- MSD/MSAD curve fitting:
  - Constant Velocity Drift Model
  - Active Diffusion Model
  - Journal-quality fitting result plots
- Summary statistics with mean and standard deviation
- Modern dark tech UI with card-based layout
- Comprehensive user manual in HTML format
- Excel output with auto-adjusted column widths
