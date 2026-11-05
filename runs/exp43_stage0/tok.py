"""§43 stage 0, tokenizer side (no model weights): chat-template fit, ' unknown' ids, ':' last token, cached refusal
first ids. Folds 0-1 only. HF_TOKEN loaded from .env, never printed."""
import json, os, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
from transformers import AutoTokenizer
from memctl.analysis import answer_selection as sel
from memctl.analysis.jacobian import wrap_chat
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D

MODELS = {"llama3b": "meta-llama/Llama-3.2-3B-Instruct", "gemma4b": "google/gemma-3-4b-it",
          "phi4mini": "microsoft/Phi-4-mini-instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct"}
P = "<<PROMPT>>"


def ids(tok, t): return list(tok(t, add_special_tokens=False).input_ids)


def first_after(tok, cont, text):
    a, b = ids(tok, cont), ids(tok, cont + text)
    return b[len(a)] if b[:len(a)] == a and len(b) > len(a) else None


def main(fams):
    out = {}
    for fam in fams:
        tok = AutoTokenizer.from_pretrained(MODELS[fam])
        r = {"model": MODELS[fam], "bos": tok.bos_token, "eos": tok.eos_token,
             "add_bos_default": ids(tok, "x") != list(tok("x").input_ids)}
        user = wrap_chat(tok, P)
        r["chat_user_only"] = user
        try:
            r["chat_system_user"] = tok.apply_chat_template([{"role": "system", "content": "<<SYS>>"}, {"role": "user", "content": P}],
                                                            tokenize=False, add_generation_prompt=True)
        except Exception as e:
            r["chat_system_user"] = f"ERROR {type(e).__name__}: {e}"
        r["tail_after_prompt"] = user.split(P, 1)[1]
        r["head_before_prompt"] = user.split(P, 1)[0]
        cids = ids(tok, user)
        r["n_bos_in_chat_ids"] = cids.count(tok.bos_token_id) if tok.bos_token_id is not None else None
        # ' unknown' variants after 'Answer:' (on a real note continuation and on the bare marker)
        cont = "Some note.\n" + sel.ANSWER_MARKER
        r["unknown_ids"] = {}
        for t in (" unknown", " Unknown", "unknown", "Unknown", " UNKNOWN", "\nunknown", " unknown."):
            b = ids(tok, cont + t)[len(ids(tok, cont)):]
            r["unknown_ids"][t] = {"ids": b, "pieces": tok.convert_ids_to_tokens(b), "first": first_after(tok, cont, t)}
        r["plain_ids_' unknown'"] = ids(tok, " unknown")
        r["marker_ids"] = {"Answer:": tok.convert_ids_to_tokens(ids(tok, "Answer:")),
                           "note\\nAnswer:": tok.convert_ids_to_tokens(ids(tok, "note.\nAnswer:")),
                           "note Answer:": tok.convert_ids_to_tokens(ids(tok, "note. Answer:"))}
        if fam == "qwen3b":
            out[fam] = r; continue
        recs = [x for f in D.FOLDS for x in D.load_records24(fam, f)[0]]
        pop = D.population(recs)
        last_colon, last_tokens, n_chat = 0, collections.Counter(), []
        ref_first, ans_first, prompt_lens = collections.Counter(), collections.Counter(), []
        for x in pop:
            note, answer = sel.split_note(x.output)
            if note is None:
                continue
            chat = wrap_chat(tok, x.prompt)
            cont = sel.frozen_continuation("A", note)
            full = ids(tok, chat) + ids(tok, cont)
            prompt_lens.append(len(full))
            last = tok.convert_ids_to_tokens([full[-1]])[0]
            last_tokens[last] += 1
            last_colon += tok.decode([full[-1]]).endswith(":")
            fid = first_after(tok, cont, answer)
            piece = None if fid is None else tok.convert_ids_to_tokens([fid])[0]
            (ref_first if sel.is_refusal_answer(sel.agent_answer(x.output)[0]) else ans_first)[(fid, piece)] += 1
        r["population_marked"] = sum(last_tokens.values())
        r["last_token_is_colon"] = {"k": last_colon, "n": sum(last_tokens.values())}
        r["last_token_pieces"] = {str(k): v for k, v in last_tokens.items()}
        r["cached_refusal_first_ids"] = {f"{k[0]} {k[1]!r}": v for k, v in ref_first.items()}
        r["cached_answer_first_ids_top"] = {f"{k[0]} {k[1]!r}": v for k, v in ans_first.most_common(8)}
        r["a_arm_prompt_tokens"] = {"min": min(prompt_lens), "median": sorted(prompt_lens)[len(prompt_lens) // 2], "max": max(prompt_lens)}
        # all-rows refusals (folds 0-1, any view) for a larger sample of refusal starts
        allref = collections.Counter()
        for x in recs:
            note, answer = sel.split_note(x.output)
            if note is not None and sel.is_refusal_answer(sel.agent_answer(x.output)[0]):
                fid = first_after(tok, sel.frozen_continuation("A", note), answer)
                allref[f"{fid} {tok.convert_ids_to_tokens([fid])[0]!r}"] += 1
        r["cached_refusal_first_ids_all_rows"] = dict(allref)
        out[fam] = r
        print(fam, json.dumps({k: v for k, v in r.items() if not k.startswith("chat_")}, indent=0, ensure_ascii=False)[:2500])
        print("SYSTEM+USER:", repr(r["chat_system_user"])[:600])
        print("USER ONLY:", repr(r["chat_user_only"])[:600])
    Path("runs/exp43_stage0/tok.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1].split(","))
