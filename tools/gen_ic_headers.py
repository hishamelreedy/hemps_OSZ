#!/usr/bin/env python3

"""Generate software/include/ic_golden_*.h from golden JSON (HEMPS or ic_experiment format)."""



import argparse

import json

import os

import sys



IC_LAYER_ALIASES = {

    "l1": "FC1",

    "l2": "FC2",

    "l3": "FC3",

    "l4": "FC3",

    "fc1": "FC1",

    "fc2": "FC2",

    "fc3": "FC3",

}





def normalize_layers(data):

    if "layers" in data:

        return data["layers"]

    if "per_layer" in data:

        out = {}

        for key, layer in data["per_layer"].items():

            if not isinstance(layer, dict):

                continue

            name = IC_LAYER_ALIASES.get(key.lower(), key.upper())

            if name in out and name == "FC3" and key.lower() == "l4":

                continue

            out[name] = dict(layer)

        return out

    return data





def emit_array(name, values):

    lines = ["static const int %s[%d] = {" % (name, len(values))]

    for i in range(0, len(values), 8):

        chunk = ", ".join(str(int(v)) for v in values[i : i + 8])

        lines.append("    " + chunk + ",")

    lines.append("};")

    return lines





def main():

    parser = argparse.ArgumentParser(description="Generate IC golden headers for HEMPS CNN apps")

    parser.add_argument("json_path", help="Golden JSON (HEMPS or ic_experiment golden_bounds)")

    parser.add_argument("--network", default="mlp784", help="Network name prefix for output file")

    parser.add_argument(

        "--output",

        default=None,

        help="Output header path (default: software/include/ic_golden_<network>.h)",

    )

    parser.add_argument(

        "--vectors-json",

        default=None,

        help="Optional JSON with golden vectors if bounds file lacks them",

    )

    args = parser.parse_args()



    with open(args.json_path, "r", encoding="utf-8") as f:

        data = json.load(f)



    vectors = {}

    if args.vectors_json:

        with open(args.vectors_json, "r", encoding="utf-8") as f:

            vdata = json.load(f)

        for key, layer in normalize_layers(vdata).items():

            if isinstance(layer, dict):

                vec = layer.get("golden_vector") or layer.get("golden")

                if vec:

                    vectors[key.upper()] = vec



    out_path = args.output

    if out_path is None:

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        out_path = os.path.join(repo_root, "software", "include", "ic_golden_%s.h" % args.network)



    lines = [

        "#ifndef IC_GOLDEN_%s_H_" % args.network.upper(),

        "#define IC_GOLDEN_%s_H_" % args.network.upper(),

        "",

        '#include "ic_check.h"',

        "",

    ]



    layers = normalize_layers(data)

    for layer_name, layer in layers.items():

        if not isinstance(layer, dict):

            continue

        sym = layer_name.upper().replace("-", "_")

        vec = layer.get("golden_vector") or layer.get("golden") or vectors.get(sym)

        if vec:

            lines.extend(emit_array("GOLDEN_%s" % sym, vec))

            lines.append("")

            lines.extend(emit_array("GOLDEN_SLOT_%s" % sym, list(range(len(vec)))))

            lines.append("")



        cal_sym = "IC_CAL_%s" % sym

        gmin = int(round(float(layer.get("gmin", layer.get("min", layer.get("pool_min", 0))))))

        gmax = int(round(float(layer.get("gmax", layer.get("max", layer.get("pool_max", 0))))))

        gavg = int(round(float(layer.get("gavg", layer.get("avg", layer.get("mu", 0))))))

        avg_slack = int(layer.get("avg_slack", 10))

        slot_slack = int(layer.get("slot_slack", 2))

        safe_mm = int(layer.get("safe_slot_mismatches", 0))

        lines.append(

            "static const struct ic_layer_cal %s = {%d, %d, %d, %d, %d, %d};"

            % (cal_sym, gmin, gmax, gavg, avg_slack, slot_slack, safe_mm)

        )

        lines.append("")

        lines.append("#define IC_%s_SLOT_SLACK %d" % (sym, slot_slack))

        lines.append("")



    lines.append("#endif")

    lines.append("")



    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:

        f.write("\n".join(lines))



    print("Wrote %s" % out_path)

    return 0





if __name__ == "__main__":

    sys.exit(main())

