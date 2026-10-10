#!/usr/bin/env python3
"""Generate simulation_data/niles_gustavo_scenario_matrix.csv (params filled, metrics blank)."""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "simulation_data" / "niles_gustavo_scenario_matrix.csv"

COLS = [
    "scenario_id",
    "arm",
    "suite",
    "yaml_path",
    "probe_h_path",
    "app_mix",
    "mesh",
    "sim_ms",
    "status",
    "notes",
    # mapping / mitigation (screenshot scenarios)
    "mapping_type",
    "pipeline_replicas",
    "task_map",
    "infected_tasks",
    "infected_links",
    "mitigation",
    "replica_tasks",
    "reroute_hops_extra",
    "ht_type",
    "ht_mode",
    "ht_count",
    "ht_loc",
    "ht_port",
    "tau_us",
    "toff_us",
    "period_us",
    "ht_start_us",
    "xor_mask",
    "batch_size",
    "batch_delay_us",
    "probe_packet_size",
    "susm_threshold",
    "security_flow",
    "rho_hb",
    "w_hb",
    "w_n",
    "n_inf",
    "delta_hb_us",
    "niles_hb_count",
    "stage2_on",
    "ic_enable",
    "psb_enable",
    "slot_slack",
    "avg_slack",
    "golden_sample_id",
    "f_hb_hz",
    "c_sec_rate",
    # security diagnostic coverage (arm capability + per-HT hit)
    "cov_drop",
    "cov_bh",
    "cov_cb",
    "cov_xor",
    "cov_misroute",
    "cov_semantic_sdc",
    "diag_coverage",
    "diag_hit_planned",
    "PD",
    "DR",
    "FN",
    "FP",
    "TPR",
    "FPR",
    "LA_link",
    "LA_pe",
    "TTL_ms",
    "FL",
    "probes_total",
    "probes_success",
    "probes_failure",
    "probe_ht_overlap_pct",
    "ic_alerts",
    "bcast_count",
    "hb_probes",
    "hb_ht_overlap_pct",
    "suspicious_paths",
    "bsa_finalized",
    "osa_finalized",
    "silent_sdc",
    "label_preserved",
    "semantic_acc",
    "C_sec",
    "S_app",
    "eta_eff",
    "eta_norm",
    "asil_tax",
    "ticks_per_inference",
    "PD_model",
    "pareto_PD",
    "pareto_asil_tax",
]


def weights(rho):
    if rho <= 0:
        return 0, 1
    if rho >= 1:
        return 1, 0
    pairs = {0.05: (1, 19), 0.1: (1, 9), 0.2: (1, 4), 0.5: (1, 1)}
    return pairs.get(rho, (1, max(1, int(round((1 - rho) / rho)))))


# Arm-level diagnostic capability (1 = mechanism can diagnose that HT class).
ARM_COV = {
    "clean": {
        "cov_drop": 0,
        "cov_bh": 0,
        "cov_cb": 0,
        "cov_xor": 0,
        "cov_misroute": 0,
        "cov_semantic_sdc": 0,
    },
    "gustavo_only": {
        "cov_drop": 1,
        "cov_bh": 1,
        "cov_cb": 1,
        "cov_xor": 0,
        "cov_misroute": 0,
        "cov_semantic_sdc": 0,
    },
    "niles_only": {
        "cov_drop": 1,
        "cov_bh": 1,
        "cov_cb": 0,
        "cov_xor": 1,
        "cov_misroute": 0,
        "cov_semantic_sdc": 1,
    },
    "hybrid": {
        "cov_drop": 1,
        "cov_bh": 1,
        "cov_cb": 1,
        "cov_xor": 1,
        "cov_misroute": 0,
        "cov_semantic_sdc": 1,
    },
}

HT_TO_COV = {
    "none": None,
    "xor": "cov_xor",
    "bh": "cov_bh",
    "cb": "cov_cb",
    "drop": "cov_drop",
    "misroute": "cov_misroute",
}


def apply_coverage(row):
    arm = row.get("arm") or "clean"
    cov = ARM_COV.get(arm, ARM_COV["clean"])
    for k, v in cov.items():
        row[k] = str(v)
    vals = list(cov.values())
    row["diag_coverage"] = "%.3f" % (sum(vals) / float(len(vals)))
    ht = (row.get("ht_type") or "none").lower()
    key = HT_TO_COV.get(ht)
    if key is None:
        row["diag_hit_planned"] = ""
    else:
        row["diag_hit_planned"] = str(cov.get(key, 0))
    return row


def main():
    rows = []

    def add(**kw):
        r = {c: "" for c in COLS}
        r.update(kw)
        apply_coverage(r)
        rows.append(r)

    add(
        scenario_id="G-BL",
        arm="clean",
        suite="intermittent_campaign",
        yaml_path="testcases/intermittent_campaign/baseline.yaml",
        app_mix="Dijkstra+DTW+MPEG",
        mesh="5x5",
        sim_ms="20",
        status="planned",
        ht_type="none",
        ht_mode="off",
        ht_count="0",
        batch_size="0",
        batch_delay_us="0",
        probe_packet_size="0",
        security_flow="none",
        rho_hb="0",
        w_hb="0",
        w_n="1",
        ic_enable="0",
        psb_enable="0",
        notes="Gustavo clean baseline; HT off; 0 probes expected",
    )

    add(
        scenario_id="N-BL",
        arm="clean",
        suite="mlp784_niles",
        yaml_path="testcases/mlp784/mlp784_clean.yaml",
        app_mix="mlp784",
        mesh="4x4",
        sim_ms="25",
        status="planned",
        ht_type="none",
        ht_mode="off",
        ht_count="0",
        rho_hb="0",
        w_hb="0",
        w_n="1",
        n_inf="1",
        delta_hb_us="0",
        stage2_on="off",
        ic_enable="1",
        psb_enable="0",
        notes="NILES clean; rho=0; FP baseline",
    )

    for bs in (3, 5, 10):
        for bd in (3, 5, 10):
            add(
                scenario_id="G-PG-BH-%d-%d-p50" % (bs, bd),
                arm="gustavo_only",
                suite="paper_gustavo",
                yaml_path="testcases/paper_gustavo/bh_intermittent_size_%d_delay_%d_payload_50.yaml"
                % (bs, bd),
                app_mix="DTW",
                mesh="5x5",
                sim_ms="20",
                status="planned",
                ht_type="bh",
                ht_mode="intermittent",
                ht_count="1",
                ht_loc="[4,4]",
                ht_port="south",
                batch_size=str(bs),
                batch_delay_us=str(bd),
                probe_packet_size="50",
                susm_threshold="3",
                security_flow="BSA+OSA+ProbeBatches",
                notes="Phase-1 single intermittent BH; thesis LA_link grid",
            )

    for bs in (3, 5, 10):
        for bd in (3, 5, 10):
            add(
                scenario_id="G-PG-CB-%d-%d-p50" % (bs, bd),
                arm="gustavo_only",
                suite="paper_gustavo",
                yaml_path="testcases/paper_gustavo/cb_intermittent_size_%d_delay_%d_payload_50.yaml"
                % (bs, bd),
                app_mix="DTW",
                mesh="5x5",
                sim_ms="20",
                status="planned",
                ht_type="cb",
                ht_mode="intermittent",
                ht_count="1",
                ht_loc="[4,4]",
                ht_port="south",
                batch_size=str(bs),
                batch_delay_us=str(bd),
                probe_packet_size="50",
                susm_threshold="3",
                security_flow="BSA+OSA+ProbeBatches",
                notes="Phase-1 intermittent Credit Block",
            )

    for bs in (5, 10, 30):
        for bd in (10, 50, 100):
            add(
                scenario_id="G-I-%d-%d" % (bs, bd),
                arm="gustavo_only",
                suite="intermittent_campaign",
                yaml_path="testcases/intermittent_campaign/size%d_delay%d.yaml" % (bs, bd),
                probe_h_path="testcases/intermittent_campaign/size%d_delay%d.h" % (bs, bd),
                app_mix="Dijkstra+DTW+MPEG",
                mesh="5x5",
                sim_ms="20",
                status="planned",
                ht_type="bh",
                ht_mode="intermittent",
                ht_count="2",
                ht_loc="[0,3]+[4,1]",
                ht_port="S+N",
                batch_size=str(bs),
                batch_delay_us=str(bd),
                probe_packet_size="17",
                susm_threshold="3",
                security_flow="BSA+OSA+ProbeBatches",
                notes="Scen6-style two intermittent BH; collision sensitivity",
            )

    add(
        scenario_id="G-I-static-BH-5-50",
        arm="gustavo_only",
        suite="intermittent_campaign",
        yaml_path="testcases/intermittent_campaign/static_bh_size5_delay50.yaml",
        app_mix="Dijkstra+DTW+MPEG",
        mesh="5x5",
        sim_ms="20",
        status="planned",
        ht_type="bh",
        ht_mode="static",
        ht_count="1",
        batch_size="5",
        batch_delay_us="50",
        probe_packet_size="17",
        security_flow="BSA+OSA+ProbeBatches",
        notes="Static BH control vs intermittent",
    )

    rhos = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    deltas = [0, 50, 100, 1000]
    hts = [
        {
            "tag": "clean",
            "ht_type": "none",
            "ht_mode": "off",
            "ht_count": "0",
            "ht_loc": "",
            "ht_port": "",
            "xor_mask": "",
            "tau_us": "",
        },
        {
            "tag": "xor",
            "ht_type": "xor",
            "ht_mode": "static",
            "ht_count": "1",
            "ht_loc": "[1,0]",
            "ht_port": "E",
            "xor_mask": "0xFF",
            "tau_us": "",
        },
        {
            "tag": "xorI",
            "ht_type": "xor",
            "ht_mode": "intermittent",
            "ht_count": "1",
            "ht_loc": "[1,0]",
            "ht_port": "E",
            "xor_mask": "0xFF",
            "tau_us": "1000",
        },
        {
            "tag": "bhI",
            "ht_type": "bh",
            "ht_mode": "intermittent",
            "ht_count": "1",
            "ht_loc": "[1,0]",
            "ht_port": "E",
            "xor_mask": "",
            "tau_us": "1000",
        },
    ]

    for ht in hts:
        for rho in rhos:
            if ht["tag"] == "clean" and rho not in (0.0, 0.1, 1.0):
                continue
            for d in deltas:
                if rho == 0.0 and d != 0:
                    continue
                if rho == 1.0 and d not in (0, 50, 1000):
                    continue
                wh, wn = weights(rho)
                sid = "N-rho%g-d%d-%s" % (rho, d, ht["tag"])
                add(
                    scenario_id=sid,
                    arm="niles_only",
                    suite="mlp784_niles",
                    yaml_path="testcases/mlp784/scenarios/%s.yaml" % sid,
                    app_mix="mlp784",
                    mesh="4x4",
                    sim_ms="25",
                    status="planned",
                    ht_type=ht["ht_type"],
                    ht_mode=ht["ht_mode"],
                    ht_count=ht["ht_count"],
                    ht_loc=ht["ht_loc"],
                    ht_port=ht["ht_port"],
                    tau_us=ht["tau_us"],
                    xor_mask=ht["xor_mask"],
                    rho_hb=str(rho),
                    w_hb=str(wh),
                    w_n=str(wn),
                    n_inf="10",
                    delta_hb_us=str(d),
                    stage2_on="hb_only" if rho > 0 else "off",
                    ic_enable="1",
                    psb_enable="1",
                    slot_slack="2",
                    avg_slack="10",
                    golden_sample_id="0",
                    notes="NILES RR weight rho_hb; delta_hb analogue of BATCH_DELAY",
                )

    add(
        scenario_id="H-MLP-hb_only-ht_payload",
        arm="hybrid",
        suite="mlp784_niles",
        yaml_path="testcases/mlp784/mlp784_ht_payload.yaml",
        app_mix="mlp784",
        mesh="4x4",
        sim_ms="25",
        status="planned",
        ht_type="xor",
        ht_mode="static",
        ht_count="1",
        ht_loc="[1,0]",
        ht_port="E",
        xor_mask="0xFF",
        rho_hb="1",
        w_hb="1",
        w_n="0",
        n_inf="0",
        delta_hb_us="0",
        niles_hb_count="1",
        stage2_on="hb_only",
        ic_enable="1",
        psb_enable="1",
        notes="Existing measured hybrid arm; semantic HB under XOR",
    )

    add(
        scenario_id="H-MLP-gustavo_bridge-xor",
        arm="hybrid",
        suite="mlp784_niles",
        yaml_path="testcases/mlp784/mlp784_ht_payload.yaml",
        app_mix="mlp784",
        mesh="4x4",
        sim_ms="25",
        status="planned",
        ht_type="xor",
        ht_mode="static",
        ht_count="1",
        ht_loc="[1,0]",
        ht_port="E",
        batch_size="5",
        batch_delay_us="50",
        probe_packet_size="50",
        security_flow="BSA+OSA+ProbeBatches",
        rho_hb="0.2",
        w_hb="1",
        w_n="4",
        delta_hb_us="50",
        stage2_on="hb_only",
        ic_enable="1",
        psb_enable="1",
        notes="Apples-to-apples hybrid: Gustavo batch+delay + NILES rho",
    )

    for tau in (100, 500, 1000, 5000):
        add(
            scenario_id="N-rho0.1-d50-xorI-tau%d" % tau,
            arm="niles_only",
            suite="mlp784_niles",
            yaml_path="testcases/mlp784/scenarios/N-rho0.1-d50-xorI-tau%d.yaml" % tau,
            app_mix="mlp784",
            mesh="4x4",
            sim_ms="25",
            status="planned",
            ht_type="xor",
            ht_mode="intermittent",
            ht_count="1",
            ht_loc="[1,0]",
            ht_port="E",
            tau_us=str(tau),
            xor_mask="0xFF",
            rho_hb="0.1",
            w_hb="1",
            w_n="9",
            n_inf="10",
            delta_hb_us="50",
            stage2_on="hb_only",
            ic_enable="1",
            psb_enable="1",
            notes="Trojan window tau sweep; validate PD=1-exp(-f*tau)",
        )

    # --- Screenshot mapping scenarios: LeNet5 sequential + Custom MLP parallel ---
    # Mesh coords: row-major [x,y] with MPE at [0,3] top-left in figure (col0,row0 visual).
    # LeNet5 (left): sequential pipeline on top two rows; infected FC2/FC3 path; yellow replica FC2.
    lenet_map = (
        "MPE@[0,3];C1P1@[1,3];C2P2@[2,3];FC1@[3,3];"
        "FC2@[3,2];FC3@[2,2];FC2_replica@[0,2]"
    )
    for mit, mid, hops, notes in (
        (
            "none",
            "MAP-LENET-seq-infect",
            "0",
            "Screenshot LeNet5: green path MPE->C1P1->C2P2->FC1; red FC1->FC2->FC3 infected; no mitigation",
        ),
        (
            "task_replica_reroute",
            "MAP-LENET-seq-mitigate",
            "3",
            "Screenshot LeNet5: yellow FC2 replica @[0,2]; reroute MPE->FC2_rep->FC3 bypasses infected FC2",
        ),
    ):
        add(
            scenario_id=mid,
            arm="niles_only",
            suite="mapping_screenshot",
            yaml_path="testcases/lenet5/scenarios/%s.yaml" % mid,
            app_mix="lenet5",
            mesh="4x4",
            sim_ms="50",
            status="planned",
            mapping_type="sequential_pipeline",
            pipeline_replicas="1",
            task_map=lenet_map,
            infected_tasks="FC2@[3,2];FC3_inlink",
            infected_links="[3,3]->[3,2];[3,2]->[2,2]",
            mitigation=mit,
            replica_tasks="FC2@[0,2]" if mit != "none" else "",
            reroute_hops_extra=hops,
            ht_type="bh",
            ht_mode="static",
            ht_count="2",
            ht_loc="[3,2]+[3,2]->[2,2]",
            ht_port="router/link",
            rho_hb="0.2",
            w_hb="1",
            w_n="4",
            n_inf="10",
            delta_hb_us="50",
            stage2_on="hb_only",
            ic_enable="1",
            psb_enable="1",
            notes=notes,
        )

    # Custom MLP (right): 4 parallel row pipelines; cols = MPE/FC1, FC2, FC3, FC4.
    mlp_map = (
        "row0:MPE/FC1@[0,3]->FC2@[1,3]->FC3@[2,3]->FC4@[3,3];"
        "row1:MPE/FC1@[0,2]->FC2@[1,2]->FC3@[2,2]->FC4@[3,2];"
        "row2:MPE/FC1@[0,1]->FC2@[1,1]->FC3@[2,1]->FC4@[3,1];"
        "row3:MPE/FC1@[0,0]->FC2@[1,0]->FC3@[2,0]->FC4@[3,0]"
    )
    for mit, mid, hops, notes in (
        (
            "none",
            "MAP-MLP-par-infect",
            "0",
            "Screenshot Custom MLP: 4 parallel pipelines; infected FC2@[1,2] and FC3@[2,1]; no cross-row mitigation",
        ),
        (
            "cross_pipeline_replica",
            "MAP-MLP-par-mitigate",
            "4",
            "Screenshot Custom MLP: yellow vertical replicas/reroutes bypass infected FC2 row1 and FC3 row2",
        ),
    ):
        add(
            scenario_id=mid,
            arm="niles_only",
            suite="mapping_screenshot",
            yaml_path="testcases/mlp_parallel/scenarios/%s.yaml" % mid,
            app_mix="custom_mlp_parallel",
            mesh="4x4",
            sim_ms="50",
            status="planned",
            mapping_type="parallel_pipelines",
            pipeline_replicas="4",
            task_map=mlp_map,
            infected_tasks="FC2@[1,2];FC3@[2,1]",
            infected_links="[1,2]->[2,2];[2,1]->[3,1]",
            mitigation=mit,
            replica_tasks=(
                "FC2_via_row0;FC3_via_row1;FC2_via_row3;FC4_via_row2" if mit != "none" else ""
            ),
            reroute_hops_extra=hops,
            ht_type="bh",
            ht_mode="static",
            ht_count="2",
            ht_loc="[1,2]+[2,1]",
            ht_port="E",
            rho_hb="0.2",
            w_hb="1",
            w_n="4",
            n_inf="10",
            delta_hb_us="50",
            stage2_on="hb_only",
            ic_enable="1",
            psb_enable="1",
            notes=notes,
        )

    # Mapping x Gustavo probe and NILES rho corners for quantitative overlay
    for arm, sid_prefix, rho, bs, bd in (
        ("gustavo_only", "MAP-LENET-G", "", "5", "50"),
        ("niles_only", "MAP-LENET-N", "0.2", "", ""),
        ("hybrid", "MAP-LENET-H", "0.2", "5", "50"),
    ):
        add(
            scenario_id="%s-seq-mitigate" % sid_prefix,
            arm=arm,
            suite="mapping_screenshot",
            yaml_path="testcases/lenet5/scenarios/%s-seq-mitigate.yaml" % sid_prefix,
            app_mix="lenet5",
            mesh="4x4",
            sim_ms="50",
            status="planned",
            mapping_type="sequential_pipeline",
            pipeline_replicas="1",
            task_map=lenet_map,
            infected_tasks="FC2@[3,2]",
            infected_links="[3,3]->[3,2];[3,2]->[2,2]",
            mitigation="task_replica_reroute",
            replica_tasks="FC2@[0,2]",
            reroute_hops_extra="3",
            ht_type="bh",
            ht_mode="static",
            ht_count="2",
            ht_loc="[3,2]",
            batch_size=bs,
            batch_delay_us=bd,
            probe_packet_size="50" if bs else "",
            security_flow="BSA+OSA+ProbeBatches" if arm != "niles_only" else "",
            rho_hb=rho,
            w_hb="1" if rho else "",
            w_n="4" if rho else "",
            delta_hb_us="50" if rho else "",
            stage2_on="hb_only" if arm != "gustavo_only" else "off",
            ic_enable="0" if arm == "gustavo_only" else "1",
            psb_enable="0" if arm == "gustavo_only" else "1",
            notes="LeNet5 screenshot mapping under %s arm" % arm,
        )

    for arm, sid_prefix, rho, bs, bd in (
        ("gustavo_only", "MAP-MLP-G", "", "5", "50"),
        ("niles_only", "MAP-MLP-N", "0.2", "", ""),
        ("hybrid", "MAP-MLP-H", "0.2", "5", "50"),
    ):
        add(
            scenario_id="%s-par-mitigate" % sid_prefix,
            arm=arm,
            suite="mapping_screenshot",
            yaml_path="testcases/mlp_parallel/scenarios/%s-par-mitigate.yaml" % sid_prefix,
            app_mix="custom_mlp_parallel",
            mesh="4x4",
            sim_ms="50",
            status="planned",
            mapping_type="parallel_pipelines",
            pipeline_replicas="4",
            task_map=mlp_map,
            infected_tasks="FC2@[1,2];FC3@[2,1]",
            infected_links="[1,2]->[2,2];[2,1]->[3,1]",
            mitigation="cross_pipeline_replica",
            replica_tasks="cross_row yellow paths",
            reroute_hops_extra="4",
            ht_type="bh",
            ht_mode="static",
            ht_count="2",
            ht_loc="[1,2]+[2,1]",
            batch_size=bs,
            batch_delay_us=bd,
            probe_packet_size="50" if bs else "",
            security_flow="BSA+OSA+ProbeBatches" if arm != "niles_only" else "",
            rho_hb=rho,
            w_hb="1" if rho else "",
            w_n="4" if rho else "",
            delta_hb_us="50" if rho else "",
            stage2_on="hb_only" if arm != "gustavo_only" else "off",
            ic_enable="0" if arm == "gustavo_only" else "1",
            psb_enable="0" if arm == "gustavo_only" else "1",
            notes="Custom MLP parallel screenshot mapping under %s arm" % arm,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    yaml_count = emit_scenario_yamls(rows)
    targets = [OUT]
    alt = OUT.with_name(OUT.stem + "_v2.csv")
    try:
        with OUT.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        written = OUT
    except PermissionError:
        with alt.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        written = alt
        print("NOTE: %s locked; wrote %s" % (OUT.name, alt.name))

    print("Wrote %d data rows to %s" % (len(rows), written))
    print("Emitted %d scenario YAML files" % yaml_count)


def _ht_mask(port_char, east=True):
    """10-char HT mask: e0 e1 w0 w1 n0 n1 s0 s1 l0 l1."""
    chars = ["x"] * 10
    idx = 0 if east else 6  # east e0 or south s0
    chars[idx] = port_char
    return "".join(chars)


def _mlp784_yaml(scenario_id, ht_entries, secure="no", comment=""):
    lines = []
    if comment:
        lines.append("# %s" % comment)
    lines.append("# scenario_id: %s" % scenario_id)
    lines.append("hw:")
    lines.append("  page_size_KB: 256")
    lines.append("  tasks_per_PE: 1")
    lines.append("  repository_size_MB: 1")
    lines.append("  physical_channels: 2")
    lines.append("  model_description: hybrid")
    lines.append("  noc_buffer_size: 8")
    lines.append("  mpsoc_dimension: [4, 4]")
    lines.append("  cluster_dimension: [4, 4]")
    lines.append("  master_location: LT")
    lines.append("  session: no")
    lines.append("  gray_area:")
    lines.append("    rows: [0, 1, 2, 3]")
    lines.append("    cols: [0, 1, 2, 3]")
    lines.append("  open_port:")
    lines.append("    - port: [Injector, 90, 0, 3, N, 0, 2, W]")
    if ht_entries:
        lines.append("  ht:")
        for x, y, mask, note in ht_entries:
            lines.append("    # %s" % note)
            lines.append('    - router: [%d, %d, "%s"]' % (x, y, mask))
    lines.append("apps:")
    lines.append("  - name: mlp784")
    lines.append("    start_time_ms: 0")
    lines.append("    secure: %s" % secure)
    lines.append("    static_mapping:")
    lines.append("      mlp_master: [0, 0]")
    lines.append("      mlp_fc1: [1, 0]")
    lines.append("      mlp_fc2: [2, 0]")
    lines.append("      mlp_fc3: [3, 0]")
    lines.append("")
    return "\n".join(lines)


def _lenet_yaml(scenario_id, mapping, ht_entries, secure="yes", comment=""):
    lines = []
    if comment:
        lines.append("# %s" % comment)
    lines.append("# scenario_id: %s" % scenario_id)
    lines.append("hw:")
    lines.append("  page_size_KB: 256")
    lines.append("  tasks_per_PE: 1")
    lines.append("  repository_size_MB: 1")
    lines.append("  physical_channels: 2")
    lines.append("  model_description: hybrid")
    lines.append("  noc_buffer_size: 8")
    lines.append("  mpsoc_dimension: [4, 4]")
    lines.append("  cluster_dimension: [4, 4]")
    lines.append("  master_location: LT")
    lines.append("  session: no")
    lines.append("  gray_area:")
    lines.append("    rows: [0, 1, 2, 3]")
    lines.append("    cols: [0, 1, 2, 3]")
    lines.append("  open_port:")
    lines.append("    - port: [Injector, 90, 0, 3, N, 0, 2, W]")
    if ht_entries:
        lines.append("  ht:")
        for x, y, mask, note in ht_entries:
            lines.append("    # %s" % note)
            lines.append('    - router: [%d, %d, "%s"]' % (x, y, mask))
    lines.append("apps:")
    lines.append("  - name: lenet5")
    lines.append("    start_time_ms: 0")
    lines.append("    secure: %s" % secure)
    lines.append("    static_mapping:")
    for task, xy in mapping:
        lines.append("      %s: [%d, %d]" % (task, xy[0], xy[1]))
    lines.append("")
    return "\n".join(lines)


def _mlp_parallel_yaml(scenario_id, ht_entries, mitigate=False, comment=""):
    """4 parallel mlp784 pipelines (rows y=0..3); cols master,fc1,fc2,fc3."""
    lines = []
    if comment:
        lines.append("# %s" % comment)
    lines.append("# scenario_id: %s" % scenario_id)
    lines.append("# Custom MLP parallel mapping (screenshot): 4 row pipelines on 4x4 mesh")
    lines.append("hw:")
    lines.append("  page_size_KB: 256")
    lines.append("  tasks_per_PE: 1")
    lines.append("  repository_size_MB: 1")
    lines.append("  physical_channels: 2")
    lines.append("  model_description: hybrid")
    lines.append("  noc_buffer_size: 8")
    lines.append("  mpsoc_dimension: [4, 4]")
    lines.append("  cluster_dimension: [4, 4]")
    lines.append("  master_location: LT")
    lines.append("  session: no")
    lines.append("  gray_area:")
    lines.append("    rows: [0, 1, 2, 3]")
    lines.append("    cols: [0, 1, 2, 3]")
    lines.append("  open_port:")
    lines.append("    - port: [Injector, 90, 0, 3, N, 0, 2, W]")
    if ht_entries:
        lines.append("  ht:")
        for x, y, mask, note in ht_entries:
            lines.append("    # %s" % note)
            lines.append('    - router: [%d, %d, "%s"]' % (x, y, mask))
    lines.append("apps:")
    for y in range(4):
        # mitigation: swap infected row pipelines onto neighboring healthy PE columns via remap notes
        # Infected: FC2 at [1,2] (row y=2), FC3 at [2,1] (row y=1) — keep HT; remap only if mitigate
        lines.append("  - name: mlp784")
        lines.append("    start_time_ms: 0")
        lines.append("    secure: no")
        lines.append("    static_mapping:")
        if mitigate and y == 2:
            # bypass infected fc2@[1,2]: place fc2 on [1,3] (neighbor row) — approx yellow reroute
            lines.append("      mlp_master: [0, %d]" % y)
            lines.append("      mlp_fc1: [1, %d]" % 3)
            lines.append("      mlp_fc2: [2, %d]" % y)
            lines.append("      mlp_fc3: [3, %d]" % y)
        elif mitigate and y == 1:
            lines.append("      mlp_master: [0, %d]" % y)
            lines.append("      mlp_fc1: [1, %d]" % y)
            lines.append("      mlp_fc2: [2, %d]" % 0)
            lines.append("      mlp_fc3: [3, %d]" % y)
        else:
            lines.append("      mlp_master: [0, %d]" % y)
            lines.append("      mlp_fc1: [1, %d]" % y)
            lines.append("      mlp_fc2: [2, %d]" % y)
            lines.append("      mlp_fc3: [3, %d]" % y)
    lines.append("")
    return "\n".join(lines)


def emit_scenario_yamls(rows):
    """Create YAMLs for mlp784/lenet5/mlp_parallel scenario paths; skip existing Gustavo suites."""
    written = 0
    for r in rows:
        path = r.get("yaml_path") or ""
        if not path.startswith("testcases/mlp784/scenarios/") and not path.startswith(
            "testcases/lenet5/scenarios/"
        ) and not path.startswith("testcases/mlp_parallel/scenarios/"):
            continue
        sid = r["scenario_id"]
        ht = (r.get("ht_type") or "none").lower()
        mode = (r.get("ht_mode") or "off").lower()
        full = REPO / path
        full.parent.mkdir(parents=True, exist_ok=True)

        if path.startswith("testcases/mlp784/scenarios/"):
            ht_entries = []
            if ht == "xor":
                # f = payload corrupt (wrapper); east of fc1@[1,0]
                ht_entries.append((1, 0, _ht_mask("f", True), "payload XOR on fc1->fc2 east"))
            elif ht == "bh":
                ch = "I" if mode == "intermittent" else "b"
                ht_entries.append((1, 0, _ht_mask(ch, True), "blackhole on fc1->fc2 east"))
            text = _mlp784_yaml(
                sid,
                ht_entries,
                secure="no",
                comment="auto-generated NILES scenario; rho_hb=%s delta_hb_us=%s"
                % (r.get("rho_hb"), r.get("delta_hb_us")),
            )
            # annotate tau in comment for intermittent
            if r.get("tau_us"):
                text = "# planned_tau_us: %s (HT ON window; RTL intermittent period is architecture-fixed)\n" % r[
                    "tau_us"
                ] + text
            full.write_text(text, encoding="utf-8")
            written += 1
            continue

        if path.startswith("testcases/lenet5/scenarios/"):
            infect = "infect" in sid and "mitigate" not in sid
            # Screenshot-style: top row y=3 pipeline; FC2 infect at [3,2] or replica at [0,2]
            if infect:
                mapping = [
                    ("lenet_master", (0, 3)),
                    ("lenet_c1p1", (1, 3)),
                    ("lenet_c2p2", (2, 3)),
                    ("lenet_fc1", (3, 3)),
                    ("lenet_fc2", (3, 2)),
                    ("lenet_fc3", (2, 2)),
                ]
            else:
                mapping = [
                    ("lenet_master", (0, 3)),
                    ("lenet_c1p1", (1, 3)),
                    ("lenet_c2p2", (2, 3)),
                    ("lenet_fc1", (3, 3)),
                    ("lenet_fc2", (0, 2)),  # yellow replica / reroute
                    ("lenet_fc3", (2, 2)),
                ]
            # HT on south port of FC1@[3,3] toward infected FC2 seat [3,2]
            ht_entries = [
                (3, 3, _ht_mask("b", east=False), "BH south FC1->[3,2] infected seat"),
                (3, 2, _ht_mask("b", east=False), "BH on infected FC2 router"),
            ]
            full.write_text(
                _lenet_yaml(
                    sid,
                    mapping,
                    ht_entries,
                    comment="LeNet5 screenshot mapping; mitigation=%s" % r.get("mitigation"),
                ),
                encoding="utf-8",
            )
            written += 1
            continue

        if path.startswith("testcases/mlp_parallel/scenarios/"):
            mitigate = "mitigate" in sid
            ht_entries = [
                (1, 2, _ht_mask("b", True), "BH east infected FC2@[1,2]"),
                (2, 1, _ht_mask("b", True), "BH east infected FC3@[2,1]"),
            ]
            full.write_text(
                _mlp_parallel_yaml(
                    sid,
                    ht_entries,
                    mitigate=mitigate,
                    comment="Custom MLP parallel screenshot; mitigation=%s" % r.get("mitigation"),
                ),
                encoding="utf-8",
            )
            written += 1
            continue

    return written


if __name__ == "__main__":
    main()