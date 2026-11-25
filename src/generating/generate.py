
from typing import Any, cast

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)

from src.chunking.chunk import Chunk
from src.errors.unsupported_device_error import UnsupportedDeviceError


class Generator:
    model_name: str
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    query_cache: dict[int, str]

    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir="/tmp/transformers_models_cache"
            )
        self.model =\
            AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    dtype="auto",
                    cache_dir="/tmp/transformers_models_cache"
                )
        self.query_cache = {}

    def _call(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        if not isinstance(inputs, BatchEncoding):
            raise UnsupportedDeviceError(str(self.model.device))

        inputs = inputs.to(self.model.device)

        generate: Any = cast(Any, self.model.generate)
        outputs = generate(
            inputs.input_ids,
            max_new_tokens=256,
            do_sample=False,
        )

        input_length = inputs["input_ids"].shape[-1]

        answer: str | list[str] = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        )

        return "".join(answer)

    def answer(self, query: str, chunks: list[Chunk]) -> str:
        key = hash((query, hash(tuple(chunks))))

        if self.query_cache.get(key):
            return self.query_cache[key]

        context = "\n\n".join(c.content for c in chunks)

        examples = [
            ("Where can I find information about using generative models \
in vLLM?", "Information about using generative models in vLLM can be found \
on the generative models page, as referenced in the supported models \
documentation."),
            ("What conditions must be met for vLLM's ModelRunner to use CUDA \
graphs instead of the regular model?", "Two conditions must be met: \
`prefill_meta` must be `None` and \
`decode_meta.use_cuda_graph` must be `True`. When both conditions are \
satisfied, the ModelRunner uses `self.graph_runners[virtual_engine]` \
instead of `self.model`."),
                ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Provide direct instructions or explanations."
                    "Never use markdown, "
                    "never use code blocks, never use formatting symbols. "
                    "Answer in plain text only."
                ),
            }
        ]
        for q, a in examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}",
            }
        )

        answer = self._call(messages)

        self.query_cache[key] = answer

        return answer

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

        return rephrased_query
