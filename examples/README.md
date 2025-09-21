# Examples overview

## Gap analysis widgets
The `gap_analysis_widgets.py` helper and the `svami_gap_analysis_demo.ipynb` notebook now provide an interactive dashboard for the Svami orchestrator.

### Quick start (local)
1. Start the Svami orchestrator on `http://localhost:8000` and export `SVAMI_API_TOKEN` with a valid bearer token.
2. Launch Jupyter or Kaggle, open `examples/svami_gap_analysis_demo.ipynb`, and run the final cell.  
3. Enter a topic such as `FastAPI onboarding` and press **Run gap analysis** to fetch live results.

### Kaggle/offline mode
Kaggle forbids outbound network calls. Set `GAP_ANALYSIS_OFFLINE=1` in the notebook (or run `export GAP_ANALYSIS_OFFLINE=1` before launching) to use the bundled sample payload in `examples/data/sample_gap_analysis.json`. The widget still renders summaries, a heat map and a progress curve using Plotly, making it safe for demos without external requests.

The notebook produces:
- Markdown summary with severity, confidence and suggested fixes.
- Interactive Plotly heat map aggregating severity by artifact type and rule.
- Progress chart that tracks how many gaps remain as the most severe findings are addressed.

Refer to `examples/gap_analysis_widgets.py` for the underlying helper functions if you’d like to embed the visuals in other notebooks or scripts.
