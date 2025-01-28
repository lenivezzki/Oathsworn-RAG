import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.manual_seed(42)


class RulesAssistant:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
                                                model_name, 
                                                torch_dtype=torch.float16,
                                                device_map="auto"
                                            )
        # self.model.to('mps')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        

    async def _prepare_input(self, question: str, context: str, system_prompt: str):
        prompt = f"Вопрос: {question}, Контекстная информация: {context}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    async def generate_response(self, question: str, context: str, system_prompt: str, max_new_tokens: int=350):
        input_text = await self._prepare_input(question, context, system_prompt)
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to('mps')
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.5
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response