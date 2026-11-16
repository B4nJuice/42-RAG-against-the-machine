from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chunking.chunk import Chunk
from src.utils.logger import Logger, LogLevel


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

    def _call(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

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

    def answer(self, query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(c.content for c in chunks)

        messages = [
            {
                "role": "system",
                "content": "Answer only using the provided context. "
                "Be concise. "
                "Normal text format. "
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

        return self._call(messages)

    def rephrase(self, query: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Rephrase the user's query using synonyms and different wording. "
                "Return exactly one rephrased query. "
            },
            {
                "role": "user",
                "content": f"""
                query:
                {query}
                """
            }
        ]

        rephrased_query: str = self._call(messages).strip()

        Logger.log("Rephrased query: " + rephrased_query, LogLevel.DEBUG)

        return rephrased_query
