import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

class Translator:
    available_llms = ["flan-t5", "phi-2"]

    def _is_available_llm(self, llm_name) -> bool:
        return llm_name in self.available_llms

    def load_llm(self, llm_name):
        if not self._is_available_llm:
            return None     
        if llm_name=="flan-t5":
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        elif llm_name=="phi-2":
            torch.set_default_device("cuda")
            model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        return model, tokenizer

    def translate__with_flan_t5(self, prompt):
        model, tokenizer = self.load_llm("flan-t5")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        # outputs = model.generate(input_ids)
        outputs = model.generate(input_ids,max_length= 60)
        # outputs = model.generate(input_ids,max_new_tokens=4000)
        return tokenizer.decode(outputs[0])

    def translate__with_phi_2(self, prompt):
        model, tokenizer = self.load_llm("phi-2")
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=200) 
        text = tokenizer.batch_decode(outputs)[0]
        outputs_2 = model.generate(**inputs,max_new_tokens=200) 
        text_2 = tokenizer.batch_decode(outputs_2)[0]
        prompt_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        output_tokens_1 = tokenizer.convert_ids_to_tokens(outputs[0])
        output_tokens_2 = tokenizer.convert_ids_to_tokens(outputs_2[0])
        num_prompt_tokens = len(prompt_tokens)
        num_output_tokens = len(output_tokens_1)
        num_output_tokens_2 = len(output_tokens_2)
        print("Number of tokens in prompt:", num_prompt_tokens)
        print("Number of tokens from max_length output:", num_output_tokens)
        print("Number of tokens from max_new_tokens output:", num_output_tokens_2)

    def translate(self, llm_name, prompt):
        if not self.is_available_llm:
            return None
        if llm_name=="flan-t5":
            self.translate__with_flan_t5(prompt)
        elif llm_name=="phi-2":
            self.translate__with_phi_2(prompt)

if __name__ == "__main__":
    PROMPT = "translate English to German: How old are you?"
    PROMPT = '''def print_prime(n):
        """
        Print all primes between 1 and n
        """'''
    translator = Translator
    for llm_name in Translator.available_llms:
        translated_text = translator.translate(llm_name, PROMPT)
        print(translated_text)