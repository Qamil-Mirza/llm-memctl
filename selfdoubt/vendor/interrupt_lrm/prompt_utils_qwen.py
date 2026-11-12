# Vendored from https://github.com/dynamic-lm/interrupt-lrm at commit b87d61bd2e5e998f34878fb0ee5fb2f5550f98c6
# (src/prompt_utils.py), Copyright the interrupt-lrm authors (Wu, Miroyan et al.), Apache License 2.0 (see LICENSE here).
# Function bodies below the VERBATIM marker are copied unchanged (checked by selfdoubt/tests/test_stage0.py).
# Only the functions needed for Qwen3 math runs are kept; nothing else is modified.


def make_mistral_chat_completion_request(*a, **k):  # the original imports this from vllm; Mistral is never used here
    raise NotImplementedError("Mistral path not vendored")


QWEN_END_THINKING_TOKEN = "</think>"
QWEN_END_ANSWER_TOKEN = "<|im_end|>"

# ---- VERBATIM below ----
def prompt_formatting_model(model_name, tokenizer, messages):
    if "qwen3" in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    elif "deepseek-r1" in model_name.lower() or "nemotron" in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "gpt-oss" in model_name.lower():
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="high",
        )
    elif "mistral" in model_name.lower():
        request = make_mistral_chat_completion_request(messages, tools=None)
        encoded = tokenizer.mistral.encode_chat_completion(request)
        # Patch for Magistral as they won't always enable thinking...
        formatted_prompt = encoded.text + " [THINK]"
    else:
        raise ValueError(f"Model {model_name} not supported")
    return formatted_prompt


def extract_reasoning_trace(output_text: str, model_name: str):
    if "gpt-oss" in model_name.lower():
        # Format: <|channel|>analysis<|message|>...<|end|>...<|channel|>final<|message|>...<|return|>
        if "<|channel|>analysis<|message|>" in output_text:
            # between the <|channel|>analysis<|message|> and <|end|>
            reasoning_trace = output_text.split("<|end|>")[0]
        else:
            reasoning_trace = output_text
    elif (
        "qwen3" in model_name.lower()
        or "deepseek-r1" in model_name.lower()
        or "nemotron" in model_name.lower()
    ):
        if "</think>" in output_text:
            reasoning_trace = output_text.split("</think>")[0]
        else:
            reasoning_trace = output_text
    elif "mistral" in model_name.lower():
        if "[/THINK]" in output_text:
            reasoning_trace = output_text.split("[/THINK]")[0]
        else:
            reasoning_trace = output_text
    return reasoning_trace


def thinking_token_check(model_name, reasoning_trace):
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        if "<think>\n" not in reasoning_trace:
            reasoning_trace = "<think>\n" + reasoning_trace
    elif "gpt-oss" in model_name.lower():
        if "<|channel|>analysis<|message|>" not in reasoning_trace:
            reasoning_trace = "<|channel|>analysis<|message|>\n" + reasoning_trace
    elif "mistral" in model_name.lower():
        if "[/THINK]" not in reasoning_trace:
            reasoning_trace = "[/THINK]\n" + reasoning_trace
    return reasoning_trace


def truncate_reasoning_trace(model_name, reasoning_trace, tokenizer, interrupt_pos):
    reasoning_trace_ids = tokenizer.encode(reasoning_trace)
    reasoning_trace_ids = reasoning_trace_ids[:interrupt_pos]
    reasoning_trace = tokenizer.decode(reasoning_trace_ids)
    # We need to enssure that the open bracket is not removed
    reasoning_trace = thinking_token_check(model_name, reasoning_trace)
    return reasoning_trace


def close_reasoning_trace(model_name, reasoning_trace, force_answer=False):
    assert any(
        m in model_name.lower()
        for m in ["qwen3", "deepseek-r1", "gpt-oss", "mistral", "nemotron"]
    ), f"Model {model_name} not supported"
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        if force_answer:
            reasoning_trace += "\n</think>\n\n"
        else:
            reasoning_trace += "\n</think>\n\n<|im_end|>"
    elif "gpt-oss" in model_name.lower():
        if force_answer:
            # End the current thinking channel and then start the final (answer) channel
            reasoning_trace += "<|end|><|start|>assistant<|channel|>final<|message|>"
        else:
            # End the current thinking channel
            reasoning_trace += "<|end|>"
    elif "mistral" in model_name.lower():
        if force_answer:
            reasoning_trace += "\n\n[/THINK]"
        else:
            reasoning_trace += "\n\n[/THINK]</s>"
    elif "nemotron" in model_name.lower():
        if force_answer:
            reasoning_trace += "\n</think>"
        else:
            reasoning_trace += "\n</think><SPECIAL_12>"
    return reasoning_trace


def open_close_reasoning_trace(model_name):
    assert any(
        m in model_name.lower()
        for m in ["qwen3", "deepseek-r1", "gpt-oss", "mistral", "nemotron"]
    ), f"Model {model_name} not supported"
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        return "<think>\n\n</think>\n\n"
    elif "gpt-oss" in model_name.lower():
        return "<|channel|>analysis<|message|>\n<|end|><|start|>assistant<|channel|>final<|message|>"
    elif "mistral" in model_name.lower():
        return "[THINK]\n\n[/THINK]"
    elif "nemotron" in model_name.lower():
        return "<think>\n\n</think>"


def format_user_turn(model_name, update_instructions):
    assert any(
        m in model_name.lower()
        for m in ["qwen3", "deepseek-r1", "gpt-oss", "mistral", "nemotron"]
    ), f"Model {model_name} not supported"
    if "qwen3" in model_name.lower() or "deepseek-r1" in model_name.lower():
        user_turn = f"\n<|im_start|>user\n{update_instructions}\n<|im_end|>\n<|im_start|>assistant\n"
    elif "gpt-oss" in model_name.lower():
        user_turn = (
            f"<|start|>user<|message|>{update_instructions}<|end|><|start|>assistant"
        )
    elif "mistral" in model_name.lower():
        user_turn = f"[INST]{update_instructions}[/INST]"
    elif "nemotron" in model_name.lower():
        user_turn = (
            f"<SPECIAL_11>User\n{update_instructions}\n"
            "<SPECIAL_11>Assistant\n<think>\n"
        )
    return user_turn


def get_math_messages_qwen(ex, custom_prompts, mode, problem_field_name):
    prompt = "Q: " + ex[problem_field_name] + "\n" + "A:"
    prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        + prompt
    )
    messages = [{"role": "user", "content": prompt}]
    if custom_prompts is not None and custom_prompts.get("system_prompt", "") != "":
        messages = [
            {"role": "system", "content": custom_prompts["system_prompt"]}
        ] + messages
    return messages


def format_subsequent_input_prompt(args, ex, custom_prompts, tokenizer):
    # Step 1: Format update instructions
    update_instructions = ""
    if args.mode == "subsequent_intervene" or args.mode == "subsequent_intervene_hard":
        update_instructions = ex["update"]
        update_prefix = custom_prompts.get("update_prefix", "")
        update_suffix = custom_prompts.get("update_suffix", "")
        update_instructions = update_prefix + update_instructions + update_suffix

    # Step 2: Truncate reasoning trace
    formatted_prompt = ex["formatted_input_prompt"][-1]
    total_reasoning_length = ex["metadata"][0]["total_reasoning_length"]
    reasoning_trace = extract_reasoning_trace(ex["output"][-1], args.model_name)
    if int(args.interrupt_pos) == 1:
        # We don't truncate the reasoning trace for full reasoning trace cases
        truncated_reasoning_trace = reasoning_trace
        if total_reasoning_length > 130000:
            truncated_reasoning_trace = truncate_reasoning_trace(
                args.model_name, reasoning_trace, tokenizer, 130000
            )
    else:
        interrupt_pos = int(args.interrupt_pos * total_reasoning_length)
        truncated_reasoning_trace = truncate_reasoning_trace(
            args.model_name, reasoning_trace, tokenizer, interrupt_pos
        )

    # Step 3: Append update instructions
    if args.interrupt_role == "assistant":
        prompts_prefix = custom_prompts.get("reasoning_prefix", "")
        prompts_suffix = custom_prompts.get("reasoning_suffix", "")
        update_instructions = prompts_prefix + update_instructions + prompts_suffix
        updated_reasoning_trace = truncated_reasoning_trace + update_instructions
        # Add the closing bracket for the hard interrupt mode
        if (
            args.mode == "subsequent_interrupt"
            or args.mode == "subsequent_intervene_hard"
        ):
            updated_reasoning_trace = close_reasoning_trace(
                args.model_name, updated_reasoning_trace, force_answer=True
            )
        if args.mode == "subsequent_interrupt_extreme":
            updated_reasoning_trace = close_reasoning_trace(
                args.model_name, updated_reasoning_trace, force_answer=True
            )
            if args.task == "code":
                updated_reasoning_trace += "\n```"
            elif args.task == "math":
                updated_reasoning_trace += "\n\\boxed{"
        return formatted_prompt + updated_reasoning_trace

    elif args.interrupt_role == "user":
        truncated_reasoning_trace = close_reasoning_trace(
            args.model_name, truncated_reasoning_trace, force_answer=False
        )
        user_turn = format_user_turn(args.model_name, update_instructions)
        if args.mode == "subsequent_intervene_hard":
            user_turn += open_close_reasoning_trace(
                args.model_name
            )  # in the second turn, don't allow the model to think
        return formatted_prompt + truncated_reasoning_trace + user_turn
