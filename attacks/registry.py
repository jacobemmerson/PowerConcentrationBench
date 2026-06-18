"""Maps attack names (matching `docs/ATTACK_MANUAL.md` section headings and
`configs/` filenames) to their implementation classes, and builds a configured
instance from a run-config block.

Each attack-implementing agent registers its own implementations here as they're
built. Entries are independent of each other — this file should only ever need
additive edits, never changes to existing entries, so the four agents can extend
it without colliding.
"""
from __future__ import annotations

from typing import Any, Callable

from attacks.base import Attack, AttackConfig
from attacks.prompt_based.autodan_ga import AutoDANGAAttack
from attacks.prompt_based.autodan_hga import AutoDANHGAAttack
from attacks.prompt_based.best_of_n import BestOfNAttack
from attacks.prompt_based.cipher import CipherAttack
from attacks.prompt_based.crescendo import CrescendoAttack
from attacks.prompt_based.deep_inception import DeepInceptionAttack
from attacks.prompt_based.gptfuzzer import GPTFuzzerAttack
from attacks.prompt_based.many_shot import ManyShotJailbreakAttack
from attacks.prompt_based.pair import PAIRAttack
from attacks.prompt_based.tap import TAPAttack
from models.loader import LoadedModel, load_model

ATTACK_REGISTRY: dict[str, Callable[..., Attack]] = {
    # Populated incrementally by each attack-implementing agent as it lands.
    #   "gcg": attacks.gradient_based.gcg.GCGAttack,
    #   "refusal_ablation": attacks.representation_based.refusal_ablation.RefusalAblationAttack,
    #   "lora_finetune": attacks.weight_tampering.lora_finetune.LoRAFinetuneAttack,
    "autodan_ga": AutoDANGAAttack,
    "autodan_hga": AutoDANHGAAttack,
    "many_shot": ManyShotJailbreakAttack,
    "best_of_n": BestOfNAttack,
    "pair": PAIRAttack,
    "tap": TAPAttack,
    "crescendo": CrescendoAttack,
    "deep_inception": DeepInceptionAttack,
    "cipher": CipherAttack,
    "gptfuzzer": GPTFuzzerAttack,
}


def load_attack(attack_spec: dict[str, Any], target: LoadedModel) -> Attack:
    """Instantiate a configured attack from a run-config block shaped like:

        attack:
          name: gcg
          attacker_model: ...      # optional — only attacks that need an attacker/mutator LLM
          judge_model: ...         # optional — only attacks whose search loop needs an in-loop judge
          query_budget: 5000
          extra: {suffix_length: 20, top_k: 256, batch_size: 512}

    Any `attacker_model`/`judge_model` dependency is resolved here, by registry
    name, through the exact same `models/loader.py` path as the target — so every
    model in a run, whatever its role, is loaded identically and reproducibly,
    and the resulting config alone is enough to reproduce the run.
    """
    name = attack_spec["name"]
    try:
        attack_cls = ATTACK_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown attack '{name}'. Registered attacks: {sorted(ATTACK_REGISTRY)}") from None

    config = AttackConfig(
        name=name,
        target_model=target.name,
        attacker_model=attack_spec.get("attacker_model"),
        judge_model=attack_spec.get("judge_model"),
        seed=attack_spec.get("seed", 0),
        output_dir=attack_spec.get("output_dir", "results"),
        query_budget=attack_spec.get("query_budget"),
        extra=attack_spec.get("extra", {}),
    )

    dependencies: dict[str, Any] = {}
    if config.attacker_model:
        dependencies["attacker"] = load_model(config.attacker_model)
    if config.judge_model:
        dependencies["in_loop_judge_model"] = load_model(config.judge_model)

    return attack_cls(config, target, **dependencies)
