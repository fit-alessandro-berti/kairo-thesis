"""Load the complete, pinned logs and their published drift annotations."""

import re
from collections.abc import Iterator
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

CASE = "case:concept:name"
ACTIVITY = "concept:name"
TIME = "time:timestamp"
START = "start:timestamp"
RESOURCE = "org:resource"
EVENT_COLUMNS = (
    "event:id",
    "event:duration_min",
    "lifecycle:transition",
    "org:role",
    "start_timestamp",
    START,
    RESOURCE,
)
CASE_ATTRIBUTES = {
    "amount",
    "region",
    "requestedamount",
    "loanamount",
    "applicationtype",
    "loangoal",
}
PADELLA_RECORD = "https://zenodo.org/records/21005862"
PADELLA_DOWNLOAD = (
    "https://zenodo.org/api/records/21005862/files/drifted_logs.zip/content"
)
PADELLA_SHA256 = "e6fc4f90aa76c3b6231c8305df1ffd061600e3f7fd3535be7b7366347cd8131f"
DRIVE_THESIS = (
    "https://drive.google.com/drive/folders/1sie4iFTUNmHuyvMVoW8tktyvtoUzbmEp"
)
DRIVE_DOWNLOAD = (
    "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
)
DRIVE_VIEW = "https://drive.google.com/file/d/{}/view"
ARCHIVE_REPO = "https://github.com/cpitsch/cdrift-evaluation"
ARCHIVE_REVISION = "1ca37fa79eaf50c2f25ef7857ccf27065d15d57e"
ARCHIVE_RAW = (
    f"https://raw.githubusercontent.com/cpitsch/cdrift-evaluation/{ARCHIVE_REVISION}"
)
ARCHIVE_LABELS = (
    f"{ARCHIVE_REPO}/blob/{ARCHIVE_REVISION}/testAll_reproducibility.py#L463-L482"
)
CERAVOLO_DESIGN = "https://doi.org/10.21227/2kxd-m509"

# Keys use the thesis log numbers. Paths refer to the original Zenodo release.
PADELLA = {
    "padella_04": (
        "time_fast/gradual/loan_application_process_bpi12w",
        "loan_application_process_bpi12w_time_fast_x01_gradual.xes",
        "1xnnOfJcT3rrR5ALP8nuQPPmt9SCWND5X",
    ),
    "padella_05": (
        "cf/sudden/domestic_declarations_bpi20",
        "domestic_declarations_bpi20_cf_sudden.xes",
        "11z1iHdtXMtT16PdsPnwL8er0Q-4ffI6Z",
    ),
    "padella_06": (
        "time_fast/sudden/bank_account_closure",
        "bank_account_closure_time_fast_x05_sudden.xes",
        "1OwabGKdMwpgv40rNcvyHCHoZnv7FBfdf",
    ),
    "padella_07": (
        "baselines/request_for_payment_bpi20",
        "request_for_payment_bpi20_baseline.xes",
        "13Db9EdpnpXX-FAhE-9YOT27FcRNUHCQV",
    ),
    "padella_08": (
        "cf/sudden/request_for_payment_bpi20",
        "request_for_payment_bpi20_cf_sudden.xes",
        "1OtvgRH958gg8eo1dO6X5BF0ibS4EQdXl",
    ),
    "padella_10": (
        "time_slow/sudden/domestic_declarations_bpi20",
        "domestic_declarations_bpi20_time_slow_x4_sudden.xes",
        "1fvHcZbYiKHZocXIMd5jYBdvxAGoR7JHY",
    ),
    "padella_12": (
        "baselines/loan_application_process_bpi12w",
        "loan_application_process_bpi12w_baseline.xes",
        "1CKp5H9ATp-wRl49YNO3inIq952PB2NEv",
    ),
    "padella_15": (
        "cf/gradual/purchase_order_handling_bpi19",
        "purchase_order_handling_bpi19_cf_gradual.xes",
        "1YxYAQwGGuwzJywue0lzPpJPittNoMl-y",
    ),
}

# Drive file IDs and SHA-256 digests for each XES and generator metadata pair.
RHEON = {
    "rheon_single_03": (
        "17JtSXwV__B02hg8zivLi2v1dunFNV5Dj",
        "a69a93e3557be49a96ba5b785f23dd9d90f2bce81ab48bbc8cc662087c26070e",
        "1AfrD-bNz8lBHeYVo_-Oz9tpv2scuJIjd",
        "ad6eb2e2e668d10dfd7a7e85cbf347da36d37c81b8151497d812295493d46cdd",
    ),
    "rheon_single_11": (
        "1-G6ghKxyphGtzu0iv1esCofPvCbCdOeM",
        "73c588a89e5da0d0bcc45c3da713ead25303181674c3e8cf50f78a43bd1e96b1",
        "1IoujnXTQinvFW0qLKwCZYK4Wif3h0XNW",
        "ad8c94043e3b1c35a405e6274dfd0448443c89e43010cfa5e906f0ee592fc734",
    ),
    "rheon_single_15": (
        "1vFiT5135Lxkxu1Q83QsVSEK9dUVYmzLZ",
        "27a8f74f34276f60c033da1871a23e720dc3e6f3ac5801b2bcc83742f1a37cc0",
        "1-ec6pvYu-MRgapIXE5sstREVOwPJ30m9",
        "232195d04cc140d9210f58fee03179e134948dbfbf16083bd4a14ede0fe3d557",
    ),
    "rheon_multi_02": (
        "1xfViA65xBLZ8L4YT4Yluxqu8HkqhpxmM",
        "2805962c79632a270f7fda1a7b8349db53237802fb3448eeac3e1835fe9ad5b2",
        "1diQscVoJCzp70ZMmnZDNISRd_dZYY4ze",
        "d76b738e1ba444354efe4ac0bbbe8c82482700e2fc012f77cc1bfd6c999da90e",
    ),
    "rheon_multi_03": (
        "19Ly8i3eBP8I-YpnUerTCq1EAP0glIwyo",
        "3a1effa9c782d06e4b947606a30527a966ef7be245d2baab01974e478f887b26",
        "1eF6r45rI-jzsJmuPBRIS9zkhe7wmv6Ox",
        "9c19bc5531e32ddd05b358e26074e8c936a03752c607322a1225102612c4df4d",
    ),
    "rheon_multi_07": (
        "1nBwodkc5nzDX6M4qdh2nLTyKxoJ8l7rO",
        "f204a0c0169dfb8143b398392aa3aa0336779a7b4a75c99fe952d5ebe0a54eb1",
        "1iz22QHflLChXNpQfIe7iw9BkOWaT8JcD",
        "050367455a0ea3883046bd5481b5270235b353be964dd4287affbed948316525",
    ),
    "rheon_mixed_06": (
        "12me1Hjj-LzsI7Hd-xrTJ_56aypO4FiU4",
        "c8cc9f912998cfd527c728d2b7d7bfefbd3211c6a9e3e8ea7068280019fbcc5f",
        "1GpSNYYTOuNMVUoitR1u-_dVWo6qYl2wJ",
        "dd01d01b4c504cbc991754b7071d17f58a758e0d198eb50364d30164459a2e9d",
    ),
    "rheon_mixed_07": (
        "1eS4VDEuJAdktCURmbmboBovHRW2r3o1M",
        "a80834629ca449ead6128a4f59844ddd26072821cda327da8c264a59092d6aed",
        "1MaOBX9hanHAQaIFd3PTgaehB4nG-FfuY",
        "8d13af1aa7a82c476483795c8e127e707bbb6df56aacf718c1a21a7693bca6f6",
    ),
    "rheon_mixed_13": (
        "125l8R0FSQ2aYTgql8l6L42Ak7i2-1Hwl",
        "f41fe28c12de51e7f2166385b0914ce5f0720862baff9055a0e9e38ef1699d45",
        "1EkNK7nClBWTUQYNX4sr1XoG7xaMuGfB6",
        "130918b7f78bf73c7a8f042a9fc9247d347f51e0c745f948881acb6236e50daa",
    ),
}

# The archive evaluation uses indices of last pre-change traces (zero-based).
CDRIFT = {
    "cdrift_bose": (
        "Bose/bose_log",
        (1199, 2399, 3599, 4799),
        "3e7263885628d82188e9d178f6e22e1a799434c9b52827428b81b25663040680",
    ),
    "cdrift_ceravolo_rio": (
        "Ceravolo/sudden_trace_noise0_1000_RIO",
        (499,),
        "e5decfe85754b5b01dc7d70c3d010de7d387fb47d9df1f12985a6d178e2d7847",
    ),
    "cdrift_ceravolo_rio_noise15": (
        "Ceravolo/sudden_trace_noise15_1000_RIO",
        (499,),
        "a2257a87a5b723952373b8cd10a9d06a57ef6b6921b1d3349e6533a5c48cec1c",
    ),
    "cdrift_ceravolo_rp": (
        "Ceravolo/sudden_trace_noise0_1000_rp",
        (499,),
        "ec935f70efccf09265f9faace37385939bb5ac137ee0c033f48327dad083001e",
    ),
    "cdrift_ostovar_loop": (
        "Ostovar/Atomic_Loop_output_Loop",
        (999, 1999),
        "1aa0e8d14eea2fcaa9215ad9f772e2325537e9fc0d896ebebde8fdaf5102153a",
    ),
    "cdrift_ostovar_parallel_to_sequence": (
        "Ostovar/Atomic_ParallelToSequence_output_ParallelToSequence_2",
        (999, 1999),
        "59daf1018f6ff835db58870a9756c758f34776afb2b5076a0db1bbccf737fc01",
    ),
}


def default_ids() -> list[str]:
    return [*CDRIFT, *RHEON, *PADELLA]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _cache_copy(path: Path, source, expected_digest: str) -> Path:
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            copyfileobj(source, temporary)
            temporary.flush()
            digest = _sha256(temporary_path)
            if digest != expected_digest:
                raise ValueError(f"SHA-256 mismatch for {path.name}: {digest}")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def _download(path: Path, url: str, expected_digest: str) -> Path:
    if path.is_file() and _sha256(path) == expected_digest:
        return path
    with urlopen(url, timeout=90) as response:
        return _cache_copy(path, response, expected_digest)


def _read_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    from pm4py.objects.log.importer.xes import importer

    log = importer.apply(
        path.as_posix(),
        variant=importer.Variants.ITERPARSE,
        parameters={"show_progress_bar": False},
    )
    source_ids = [trace.attributes["concept:name"] for trace in log]
    duplicates = len(source_ids) - len(set(source_ids))
    if any(not trace for trace in log):
        raise ValueError(f"Unexpected empty trace in {path.name}")
    rows = []
    case_ids = []
    for index, (trace, source_id) in enumerate(zip(log, source_ids)):
        # The Bose archive repeats its literal IDs in each of five segments.
        case_id = f"xestrace_{index:06d}" if duplicates else source_id
        case_ids.append(case_id)
        attributes = {
            f"case:{key}": value
            for key, value in trace.attributes.items()
            if key.lower() in CASE_ATTRIBUTES
        }
        for event in trace:
            rows.append(
                {
                    CASE: case_id,
                    ACTIVITY: event[ACTIVITY],
                    TIME: event[TIME],
                    **attributes,
                    **{key: event[key] for key in EVENT_COLUMNS if key in event},
                }
            )
    frame = pd.DataFrame.from_records(rows)
    del rows, log
    if "start_timestamp" in frame:
        alternate_start = frame.pop("start_timestamp")
        frame[START] = (
            frame[START].combine_first(alternate_start)
            if START in frame
            else alternate_start
        )
    for column in (TIME, START):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame[TIME].isna().any():
        raise ValueError(f"Missing event completion timestamp in {path.name}")

    # Preserve the stored trace positions for published labels and row order for
    # simultaneous events. Detection methods can choose their own time axis.
    grouped = frame.groupby(CASE, sort=False)
    starts = grouped[START].min() if START in frame else grouped[TIME].min()
    starts = starts.fillna(grouped[TIME].min())
    ends = grouped[TIME].max()
    traces = pd.DataFrame(
        {
            "source_case_id": source_ids,
            "start": starts.reindex(case_ids).array,
            "end": ends.reindex(case_ids).array,
        }
    )
    preprocessing = {
        "duplicate_source_case_ids": duplicates,
        "case_identity": "zero-based XES trace position"
        if duplicates
        else "original XES trace identifier",
        "adjacent_source_order_start_inversions": int(
            traces["start"].diff().lt(pd.Timedelta(0)).sum()
        ),
        "adjacent_source_order_completion_inversions": int(
            traces["end"].diff().lt(pd.Timedelta(0)).sum()
        ),
        "case_start_coordinate": START
        if START in frame
        else "first observed event completion",
    }
    return frame, traces, preprocessing


def _episode(drift_id, kind, perspective, mode, start, end, **native):
    return {
        "id": drift_id,
        "kind": kind,
        "perspective": perspective,
        "mode": mode,
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
        **native,
    }


def _cdrift(data_dir: Path, name: str):
    member_name, boundaries, digest = CDRIFT[name]
    member = f"EvaluationLogs/{member_name}.xes.gz"
    url = f"{ARCHIVE_RAW}/{member}"
    path = _download(data_dir / f"{name}.xes.gz", url, digest)
    frame, traces, preprocessing = _read_log(path)
    scenario = member_name.split("/")[0].lower()
    # Ceravolo's stream changes between generated cases 499 and 500. The
    # archived XES serializes the numeric IDs as strings in lexical order.
    if scenario == "ceravolo":
        by_id = traces.set_index("source_case_id", drop=False)
        if set(by_id.index) != {str(index) for index in range(len(traces))}:
            raise ValueError(f"Unexpected Ceravolo trace identities for {name}")
    truth = []
    for number, offset in enumerate(boundaries, 1):
        archived_after = traces.iloc[offset + 1]
        if scenario == "ceravolo":
            before, after = by_id.loc[str(offset)], by_id.loc[str(offset + 1)]
            native = {
                "native_archive_last_pre_change_ordinal": offset,
                "native_archive_first_post_change_ordinal": offset + 1,
                "native_archive_first_post_change_case_id": archived_after.source_case_id,
                "native_archive_first_post_change_case_start": archived_after.start.isoformat(),
            }
        else:
            before, after = traces.iloc[offset], archived_after
            native = {
                "native_last_pre_change_trace_index": offset,
                "native_first_post_change_trace_index": offset + 1,
            }
        truth.append(
            _episode(
                f"d{number:02d}",
                "control_flow",
                "intra",
                "sudden",
                after.start,
                after.start,
                native_first_post_change_case_id=after.source_case_id,
                native_last_pre_change_case_start=before.start.isoformat(),
                native_first_post_change_case_completion=after.end.isoformat(),
                **native,
            )
        )
    if scenario == "ceravolo":
        truth_basis = "start of case 500, the first case of the changed half by numeric generator ID"
    else:
        truth_basis = "start of the first post-change trace in original XES trace order"
    return {
        "id": name,
        "family": "cdrift",
        "scenario": scenario,
        "source": {
            "repository": ARCHIVE_REPO,
            "revision": ARCHIVE_REVISION,
            "event_log": url,
            "sha256": digest,
            "ground_truth": ARCHIVE_LABELS,
            "archive_path": member,
            "truth_time_basis": truth_basis,
            **(
                {"original_drift_design": CERAVOLO_DESIGN}
                if scenario == "ceravolo"
                else {}
            ),
        },
        "preprocessing": preprocessing,
        "truth": truth,
    }, frame


def _padella(data_dir: Path, name: str):
    folder, filename, folder_id = PADELLA[name]
    kind, *parts = folder.split("/")
    scenario = parts[-1]
    baseline = kind == "baselines"
    member = f"baselines/{filename}" if baseline else f"{folder}/{filename}"
    readme = None if baseline else f"{folder}/README.md"
    archive = _download(data_dir / "drifted_logs.zip", PADELLA_DOWNLOAD, PADELLA_SHA256)
    path = data_dir / f"{name}.xes"
    with ZipFile(archive) as source:
        with source.open(member) as src:
            digest = file_digest(src, "sha256").hexdigest()
        if not path.is_file() or _sha256(path) != digest:
            with source.open(member) as src:
                _cache_copy(path, src, digest)
        description = source.read(readme).decode("utf-8") if readme else ""
    frame, traces, preprocessing = _read_log(path)
    truth = []
    if description:
        point_text = re.search(r"\*\*Change point\(s\):\*\* \[([^\]]+)\]", description)[
            1
        ]
        native_indices = [int(item) for item in point_text.split(",")]
        mode = parts[0]
        if len(native_indices) != (2 if mode == "gradual" else 1):
            raise ValueError(f"Unexpected Padella truth shape for {name}")
        ids = [f"case_{index:06d}" for index in native_indices]
        starts = traces.set_index("source_case_id").loc[ids, "start"]
        truth.append(
            _episode(
                "d01",
                "control_flow" if kind == "cf" else kind,
                "intra" if kind == "cf" else "resource",
                mode,
                starts.iloc[0],
                starts.iloc[-1],
                native_case_ids=ids,
                native_zero_based_case_labels=native_indices,
                target_activity=re.search(
                    r"\*\*Target activity:\*\* `([^`]+)`", description
                )[1],
                technique=re.search(r"\*\*Technique:\*\* ([^\n]+)", description)[1],
            )
        )
    return {
        "id": name,
        "family": "padella",
        "scenario": scenario,
        "source": {
            "dataset": PADELLA_RECORD,
            "archive_sha256": PADELLA_SHA256,
            "archive_path": member,
            "sha256": digest,
            "ground_truth_archive_path": readme,
            "thesis_files": f"https://drive.google.com/drive/folders/{folder_id}",
            "truth_time_basis": "start of the trace with the published case label"
            if truth
            else None,
        },
        "preprocessing": preprocessing,
        "truth": truth,
    }, frame


def _rheon(data_dir: Path, name: str):
    file_id, file_hash, label_id, label_hash = RHEON[name]
    path = _download(
        data_dir / f"{name}.xes", DRIVE_DOWNLOAD.format(file_id), file_hash
    )
    label_path = _download(
        data_dir / f"{name}.md", DRIVE_DOWNLOAD.format(label_id), label_hash
    )
    frame, _, preprocessing = _read_log(path)
    description = label_path.read_text(encoding="utf-8")
    counts = dict(
        re.findall(r"\| (generated_traces|generated_events) \| (\d+) \|", description)
    )
    if (frame[CASE].nunique(), len(frame)) != (
        int(counts["generated_traces"]),
        int(counts["generated_events"]),
    ):
        raise ValueError(
            f"XES case or event count differs from the Rheon metadata for {name}"
        )
    perspective_names = {
        "intra-case": "intra",
        "resource": "resource",
        "inter-case": "inter",
    }
    headers_and_bodies = re.split(
        r"^### (d\d+) — (\w+) \((intra-case|resource|inter-case), (sudden|gradual)\)\s*$",
        description.split("## Drifts", maxsplit=1)[1],
        flags=re.MULTILINE,
    )
    truth = []
    for index in range(1, len(headers_and_bodies), 5):
        drift_id, kind, perspective, mode, body = headers_and_bodies[index : index + 5]
        timing = re.search(
            r"^- Drift (?:point|window): ([\d. →]+)\s*\(([^)]+)\)", body, re.MULTILINE
        )
        fractions = [float(value) for value in timing[1].split("→")]
        bounds = [value.strip() for value in timing[2].split("→")]
        truth.append(
            _episode(
                drift_id,
                kind,
                perspective_names[perspective],
                mode,
                bounds[0],
                bounds[-1],
                native_generation_horizon_fractions=fractions,
            )
        )
    if not truth:
        raise ValueError(f"No drift labels in the Rheon metadata for {name}")
    return {
        "id": name,
        "family": "rheon",
        "scenario": name.split("_")[1],
        "source": {
            "thesis_files": DRIVE_THESIS,
            "event_log": DRIVE_VIEW.format(file_id),
            "sha256": file_hash,
            "ground_truth": DRIVE_VIEW.format(label_id),
            "ground_truth_sha256": label_hash,
            "generator_scenario": description.splitlines()[0].split(": ", 1)[1],
            "truth_time_basis": "generator metadata expressed in UTC",
        },
        "preprocessing": preprocessing,
        "truth": truth,
    }, frame


def load_all(
    data_dir: Path, names: set[str] | None = None
) -> Iterator[tuple[dict, pd.DataFrame]]:
    """Read selected complete event logs. Cache downloads outside the result tree."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    requested = set(default_ids()) if names is None else names
    unknown = requested.difference(default_ids())
    if unknown:
        raise ValueError(f"Unknown dataset IDs: {', '.join(sorted(unknown))}")
    for catalogue, loader in ((CDRIFT, _cdrift), (RHEON, _rheon), (PADELLA, _padella)):
        for name in catalogue:
            if name in requested:
                yield loader(data_dir, name)
