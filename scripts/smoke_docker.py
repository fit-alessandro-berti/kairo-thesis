"""Run inside the built image: docker run --rm -i kairo-dashboard python - < scripts/smoke_docker.py."""

import base64
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pm4py
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_md5

import kairo


def check(app: AppTest) -> None:
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error, [error.value for error in app.error]


def click(app: AppTest, label: str) -> None:
    matches = [button for button in app.button if button.label == label]
    assert len(matches) == 1, (label, [button.label for button in app.button])
    matches[0].click().run()
    check(app)


def main() -> None:
    assert os.getuid() != 0, "The image should run as an unprivileged user"
    assert Path("kairo/kairo.png").is_file(), "Dashboard logo is missing"

    # Vary activities, resources, durations and daily volumes across two months.
    rows = []
    for day in range(60):
        for case in range(2 + day % 3):
            for event in range(4):
                rows.append({
                    "case:concept:name": f"case-{day}-{case}",
                    "concept:name": f"activity-{(event + day // 30) % 5}",
                    "time:timestamp": pd.Timestamp("2025-01-01", tz="UTC")
                    + pd.Timedelta(days=day, hours=case * 4 + event),
                    "org:resource": f"resource-{(case + event + day) % 4}",
                    "event_duration": float(1 + event + day % 7),
                })
    log = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as folder:
        csv = Path(folder) / "smoke.csv"
        xes = Path(folder) / "smoke.xes"
        log.to_csv(csv, index=False)
        pm4py.write_xes(log, str(xes))
        assert len(kairo.read_log(str(csv))) == len(log)
        assert len(kairo.read_log(str(xes))) == len(log)
        raw = pd.read_csv(csv)
    print("PASS: non-root runtime, assets, temporary files, CSV and XES loading")

    # Match the import path added by `streamlit run dashboard/app.py`.
    sys.path.insert(0, str(Path("dashboard").resolve()))
    app = AppTest.from_file("dashboard/app.py", default_timeout=60).run()
    check(app)
    assert app.title[0].value == "State-based process monitoring for concept-drift detection"
    app.switch_page("views/upload.py").run()
    check(app)

    # AppTest cannot drive the file uploader. Seed its parsed data and column
    # choices, then let the upload page perform its normal mapping and rendering.
    app.session_state["raw"] = raw
    app.session_state["file"] = "smoke.csv"
    app.session_state["picked"] = {column: column for column in log.columns}
    app.session_state["picked"].update(event_id=None, start_timestamp=None)
    app.session_state["case_attributes"] = {}
    app.run()
    check(app)
    assert len(app.session_state["log"]) == len(log)
    print("PASS: overview and upload page mapping")

    for perspective in ("intra", "resource", "inter"):
        for page, buttons in (
            ("features", ("Compute features",)),
            ("pca", ("Compute PCA", "Apply PCA")),
            ("states", ("Compute states", "Plot state colors", "Plot distances")),
            ("drift", ("Compute drift signal",)),
        ):
            app.switch_page(f"views/{perspective}/{page}.py")
            # AppTest.switch_page hashes the filename, but st.Page hashes the
            # custom url_path used by these otherwise identically named pages.
            app._page_hash = calc_md5(f"{perspective}_{page}")
            app.run()
            check(app)
            for label in buttons:
                click(app, label)
        scores = app.session_state[f"{perspective}_drift_signal"]["drift_score"].dropna()
        assert len(scores) > 1 and np.isfinite(scores).all()
        print(f"PASS: {perspective} features, PCA, SOM states, plots and drift signal")

    app.switch_page("views/copilot.py").run()
    check(app)
    messages = kairo.create_messages(
        "openai", None, ["Describe this drift signal."],
        [app.session_state["intra_plot_drift_signal"]],
    )
    url = messages[0]["content"][1]["image_url"]["url"]
    png = base64.b64decode(url.split(",", 1)[1])
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and len(png) > 1000
    print("PASS: Copilot page and Chromium/Kaleido plot attachment (no API call)")


if __name__ == "__main__":
    main()
