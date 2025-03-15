from unsloth.chat_templates import get_chat_template



def inference_unsloth(model, tokenizer):

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "phi-4",
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

    messages = [
        {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids = inputs, max_new_tokens = 64, use_cache = True, temperature = 1.5, min_p = 0.1
    )
    tokenizer.batch_decode(outputs)

    return outputs


if __name__ == "__main__":
    inference_unsloth()