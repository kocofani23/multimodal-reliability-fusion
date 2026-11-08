import json
d = json.load(open("experiments/results/per_seed_results.json"))
for seed in sorted(d.keys()):
    print(f"\n=== Seed {seed} ===")
    for model in ["A_latefusion_noaug", "B_latefusion_moddrop", "C_gated_concat", "D_oracle_gated_concat"]:
        print(f"  {model}:")
        for cond, r in d[seed][model].items():
            if isinstance(r, dict) and "gate" in r:
                g = r["gate"]
                print(
                    f"    {cond:<25} "
                    f"w_img={g['w_image_mean']:.3f} +/- {g['w_image_std']:.3f}  "
                    f"w_txt={g['w_text_mean']:.3f} +/- {g['w_text_std']:.3f}"
                )
