# Streamlit dashboard

Run `streamlit run qls_testing/visualization/app_streamlit.py`.

- **Results:** separate `S`, `Xs`, `P`, and `Cs` tabs for enzyme/QSSA/NDME
  systems. Each graph overlays pipeline and reference and shows both absolute
  and a user-selected absolute, relative, or combined error panel. The axis and
  summary table rescale with that choice. Point selection reveals values and
  separately labelled staged diagnostics.
- **Error & Complexity:** time-resolved decomposition, final-source table,
  human-readable runtime drivers, and LaTeX scaling formulas.
- **Math:** only equations active in the selected linearization, integrator,
  solver, or NDME branch.
- **Advanced / Debug:** lifted-coordinate traces and raw solver metadata. These
  are intentionally not the default scientific view.

Plotly supplies hover, zoom, pan, autoscale, reset, and PNG controls. CSV, NPZ,
and HTML exports use the same result object as the CLI.
