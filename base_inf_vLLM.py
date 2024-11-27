import torch
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMService:
    def __init__(self, model_name, hf_token):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        print("Loading model and tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

        # Pass model to vLLM and move to GPU
        self.llm = LLM(model=model, device=self.device)
        print("Model and tokenizer loaded successfully.")

    def generate_response(self, input_text):
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Generate output
        output = self.llm.generate(inputs['input_ids'])

        # Decode and return the generated text
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"
    hf_token = HF_Token  

    llm_service = LLMService(model_name, hf_token)

    # Run multiple inferences
    texts = [
        "Explain the theory of relativity in simple terms",
	<Other sample prompts>
    ]

    for text in texts:
        print(f"Input: {text}")
        response = llm_service.generate_response(text)
        print(f"Generated Response: {response}\n")
