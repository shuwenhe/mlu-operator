"""Check the complete local DeepSeek-V2-Lite BF16 tensor and tokenizer contract."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "integration" / "deepseek"))
from fixtures import load_headers


def main(root):
    cfg = json.loads((root / "config.json").read_text())
    h, vocab, layers = (cfg[key] for key in ("hidden_size", "vocab_size", "num_hidden_layers"))
    heads = cfg["num_attention_heads"]
    qdim = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    latent = cfg["kv_lora_rank"]
    expected = {"model.embed_tokens.weight": [vocab, h], "lm_head.weight": [vocab, h],
                "model.norm.weight": [h]}

    def mlp(prefix, width):
        expected.update({prefix + ".gate_proj.weight": [width, h],
                         prefix + ".up_proj.weight": [width, h],
                         prefix + ".down_proj.weight": [h, width]})

    for layer in range(layers):
        name = f"model.layers.{layer}."
        expected.update({name + "input_layernorm.weight": [h],
                         name + "post_attention_layernorm.weight": [h],
                         name + "self_attn.q_proj.weight": [heads * qdim, h],
                         name + "self_attn.kv_a_proj_with_mqa.weight": [latent + cfg["qk_rope_head_dim"], h],
                         name + "self_attn.kv_a_layernorm.weight": [latent],
                         name + "self_attn.kv_b_proj.weight": [
                             heads * (cfg["qk_nope_head_dim"] + cfg["v_head_dim"]), latent],
                         name + "self_attn.o_proj.weight": [h, heads * cfg["v_head_dim"]]})
        prefix = name + "mlp"
        if layer < cfg["first_k_dense_replace"] or layer % cfg["moe_layer_freq"]:
            mlp(prefix, cfg["intermediate_size"])
            continue
        expected[prefix + ".gate.weight"] = [cfg["n_routed_experts"], h]
        for expert in range(cfg["n_routed_experts"]):
            mlp(f"{prefix}.experts.{expert}", cfg["moe_intermediate_size"])
        mlp(prefix + ".shared_experts", cfg["moe_intermediate_size"] * cfg["n_shared_experts"])

    found = load_headers(root)
    if {key: value[2] for key, value in found.items()} != expected:
        raise ValueError("Checkpoint tensor names or shapes differ from the configured architecture")
    print(f"checkpoint_metadata=PASS layers={layers} tensors={len(found)} "
          f"shards={len({value[0] for value in found.values()})} "
          f"bf16_bytes={sum(2 * math.prod(shape) for shape in expected.values())}", flush=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    assert all(0 <= token < vocab for token in tok.get_vocab().values())
    for name in ("bos", "eos"):
        token = cfg[name + "_token_id"]
        assert tok.id_to_token(token) is not None
    prompt = "The capital of France is"
    ids = tok.encode(prompt, add_special_tokens=True).ids
    print(f"tokenizer=PASS tokenizer_vocab={tok.get_vocab_size()} model_vocab={vocab} prompt={prompt!r} ids={ids} decoded={tok.decode(ids)!r}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
