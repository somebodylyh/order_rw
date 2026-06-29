# Expected Questions And Answers

## 1. 这个和 diffusion language model 有什么关系？

直接结论：这不是 diffusion LM 主线，而是 text-side AO-GPT order-controller 训练加速。

证据：AO-GPT 允许任意 reveal order；我们从 attention graph 读 order signal，再用 frozen `g_beta` 控制训练 order。

边界：不要把它讲成 diffusion 结果，也不要用 image/multimodal 结果支撑 text claim。

## 2. 为什么不是直接用 L2R？

直接结论：L2R 是 reference，不是唯一目标，也不是 upper bound。

证据：seed42 frozen `g_beta` from10k recovery 是 106.1%，from20k 是 101.5%，说明 controller order 在该模型上可以达到或略优于 L2R reference。Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md`。

边界：不要说 "beats L2R" 成 universal claim；说 reaches or exceeds L2R reference under canonical-order validation。

## 3. 为什么不是直接用 CDL teacher？

直接结论：CDL 是 offline teacher / extractor；`g_beta` 是可部署的 learned readout。

证据：pipeline 是 CDL label -> train `g_beta` -> frozen hook；当前主 acceleration evidence 来自 frozen `g_beta`，不是直接 CDL。

边界：CDL teacher matched result 在当前 package 里仍应标 pending / supplementary，除非另有 verified final table。

## 4. `g_beta` 是不是只是学了 L2R prior？

直接结论：不是，legacy B0 sanity 支持它读取 B structure。

证据：real B `tau_vs_l2r=0.967500`，Gaussian `-0.003353`，entry-shuffled `0.016488`，row/col shuffled `0.012748`，Gaussian pairwise tau `0.000313`。Source: `reports/evidence_package_20260613_verified/raw/gbeta_input_sanity_final.json`。

边界：zero B gives L2R because margin=0 tie-breaking; no B1 `g_beta` sanity yet。

## 5. 你怎么证明 B 里真的有信息？

直接结论：destroyed-B controls destroy the output order signal。

证据：real B high tau; Gaussian / shuffled / row-col shuffled near zero; real pairwise tau high while Gaussian pairwise tau near zero。

边界：non-L2R subset n=4 inconclusive，不用它做主证据。

## 6. B1 更新后，对原结果有没有影响？

直接结论：B1 强化 mechanism story，但不自动替代已完成的 B0 acceleration evidence。

证据：B1 clean-base ladder `L0H0 tau=1.0` at 10k/50k/60k；continuous seed124 final `L0H4 tau=0.957589`。

边界：existing `g_beta` sanity and frozen hook are B0 legacy path。

## 7. B0 和 B1 到底什么关系？

直接结论：B0 是 legacy controller extraction；B1/predictor 是 target collaborator-aligned diagnostic convention，B1 controller migration 正在进行。

证据：B0 hook provider uses `_attn_to_A_block_b0_vec`; B1 result files use `none_mode=predictor`。

边界：当前结果-bearing B1 不是 code enum `none_mode=b1` physical-remap variant。

## 8. 现有 acceleration 是否是 B1？

直接结论：不是。当前 verified acceleration 是 legacy B0 hook path。

证据：`selected_head_dataset.py` default `none_mode="b0"`；`hook_order_provider.py` mirrors B0 extraction；verified package labels acceleration as B0 legacy。

边界：如果要 claim B1-controller acceleration，需要 B1 `g_beta` + B1 hook rerun。

## 9. val_unstructured 为什么退化？

直接结论：这是 order specialization trade-off。

证据：frozen hook improves `val_ori_l2r_block` but worsens `val_unstructured_order`; seed123 from10k unstructured delta +0.424, seed42 from10k +1.283。Source: `reports/evidence_package_20260613_verified/02_VERIFIED_TABLES.md` Table V9。

边界：我们不 claim universal permutation-robust improvement。

## 10. order 是 per-sample 还是 batch-global？

直接结论：batch-global canonical order。

证据：pipeline distills batch-mean selected-head `B` and deploys a global controller order。

边界：没有 per-sample adaptive order 证据。

## 11. PE leakage 怎么解释？

直接结论：当前最稳说法是 fixed clean-permutation protocol 下 attention recovers physical-L2R-like structure；不要 claim no-PE discovery。

证据：model coordinate and physical coordinate are separated by fixed permutation; diagnostic remap tests physical order recovery。

边界：没有 no-PE / PE-free control。

## 12. head selection 是不是 oracle？

直接结论：selected heads are seed-dependent and selected through cheap diagnostics plus audition; not fully label-free end-to-end。

证据：seed123 uses L0H2; seed42 uses L0H4; verified package says selected heads are not max-|tau| heads。

边界：不要 claim fully label-free。

## 13. 只有 2 个 seed 够吗？

直接结论：对 boss update 足够形成 strong preliminary text-side story；对 paper submission，third seed 是高价值补强。

证据：two matched seed groups both positive; from10k/from20k step saving 33-42%。

边界：统计完整性仍有限。

## 14. 317M 有没有加速结果？

直接结论：没有 hook acceleration，只有 diagnostic。

证据：317M B0 scan has 6/256 strong heads, best `L0H10 tau=0.955357`。Source: `block_lo_arm_order_network/probe_results/large_random_baseline_16l16h1024d/head_scan_step5000.json`。

边界：no 317M hook; no B1 317M scan found。

## 15. image side 是否能放进论文？

直接结论：可以作为 motivation / future work，不宜作为主 evidence。

证据：verified text package says image side has diagnostics but no frozen hook training。

边界：do not claim image/multimodal solved。

## 16. 现在 AAAI/ICML/NeurIPS 够不够？

直接结论：text-side workshop/short-to-main-paper direction has a coherent core; top-tier main-paper strength depends on third seed, B1 hook closure, and reviewer-facing controls。

证据：current chain has mechanism + sanity + two-seed acceleration + B1 robustness。

边界：venue decision is strategic; evidence gaps should be explicit。

## 17. 下一步最值得补什么？

直接结论：highest value is claim-risk reduction, not broad expansion。

证据：B1 hook smoke would unify protocol; third seed improves statistics; label-free closure reduces oracle-selection critique。

边界：317M hook is high cost/high risk。

## 18. 如果 CDL teacher 比 `g_beta` 强怎么办？

直接结论：这不削弱 `g_beta`，反而说明 teacher contains useful order signal；`g_beta` is the deployable distilled approximation。

证据：CDL teacher seed2 from10k reaches 3.284 @50k but is supplementary / seed-mismatch in current table。

边界：do not compare cross-seed CDL as primary matched result unless matched run is verified。

## 19. 如果 B1 hook rerun 不如 B0 怎么办？

直接结论：then final paper can keep B0 as controller path and B1 as robustness diagnostic, or analyze B1 as a migration change that needs retuning。

证据：B1 already proves signal survives extraction alignment; controller performance depends on matching extraction, selected head, teacher labels, and hook path。

边界：do not promise B1 controller acceleration before result。

## 20. 论文最稳 claim 是什么？

直接结论：attention-derived order controllers can accelerate canonical-order training in text-side AO-GPT, and the upstream order-bearing attention signal survives B1/predictor-aligned extraction。

证据：33-42% step saving under B0 legacy hook; B1 clean-base and continuous diagnostics strong。

边界：not universal random-order improvement, not fully label-free, not B1 hook acceleration yet。

