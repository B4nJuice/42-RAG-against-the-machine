from src.chunking.chunk import Chunk

from transformers import AutoTokenizer, AutoModelForCausalLM

class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B") -> None:
        self.model_name: str = model_name
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/tmp/transformers_models_cache"
            )
        self.model: AutoModelForCausalLM =\
            AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    dtype="auto",
                    cache_dir="/tmp/transformers_models_cache"
                )

    def answer(self, query: str, chunks: list[Chunk]):
        context = "\n\n".join(c.content for c in chunks)

        messages = [
            {
                "role": "system",
                "content": "Answer only using the provided context."
            },
            {
                "role": "user",
                "content": f"""
                Context:
                {context}
    
                Question:
                {query}
                """
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        print(inputs)

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=False,
        )

        input_length = inputs["input_ids"].shape[-1]

        answer = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return answer