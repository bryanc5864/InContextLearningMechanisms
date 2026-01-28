"""Model loading and management using HuggingFace transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


class HookedModel:
    """Wrapper around HuggingFace model providing hook-based activation access.

    Provides an interface for:
    - Extracting residual stream activations at any layer
    - Injecting/replacing activations via forward hooks
    - Generating text with or without interventions
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        config = model.config
        self.n_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size

    def get_layer_module(self, layer: int):
        """Get the transformer layer module for hook registration."""
        return self.model.model.layers[layer]

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text, returning token IDs on device."""
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].to(self.device)

    def to_string(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate(self, prompt: str, max_new_tokens: int = 30) -> str:
        """Generate text from a prompt, return only the generated part."""
        input_ids = self.tokenize(prompt)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Take only the first line (stop at newline)
        first_line = text.split("\n")[0].strip()
        return first_line

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        layers: list[int] | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run forward pass and cache residual stream activations.

        Args:
            input_ids: Token IDs, shape (1, seq_len).
            layers: Layers to cache. If None, cache all layers.

        Returns:
            Dict mapping layer -> activation tensor of shape (seq_len, d_model).
        """
        if layers is None:
            layers = list(range(self.n_layers))

        cache = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is a tuple; output[0] is the hidden states
                hidden = output[0]
                cache[layer_idx] = hidden[0].detach().cpu()  # (seq_len, d_model)
            return hook_fn

        for l in layers:
            h = self.get_layer_module(l).register_forward_hook(make_hook(l))
            hooks.append(h)

        with torch.no_grad():
            self.model(input_ids)

        for h in hooks:
            h.remove()

        return cache

    def forward_with_intervention(
        self,
        input_ids: torch.Tensor,
        layer: int,
        position: int,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass with activation replacement at (layer, position).

        Args:
            input_ids: Token IDs, shape (1, seq_len).
            layer: Layer index for intervention.
            position: Token position for intervention.
            vector: Replacement vector, shape (d_model,).

        Returns:
            Logits tensor.
        """
        vector = vector.to(device=self.device, dtype=self.dtype)

        def hook_fn(module, input, output):
            hidden = output[0]
            hidden[0, position, :] = vector
            # Return modified output (preserve tuple structure)
            return (hidden,) + output[1:]

        h = self.get_layer_module(layer).register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = self.model(input_ids)

        h.remove()
        return outputs.logits

    def generate_with_intervention(
        self,
        prompt: str,
        layer: int,
        position: int,
        vector: torch.Tensor,
        max_new_tokens: int = 30,
    ) -> str:
        """Generate text with activation intervention on the first forward pass.

        The hook replaces the activation at (layer, position) during the
        initial prompt processing. Subsequent autoregressive steps are unhooked.
        """
        input_ids = self.tokenize(prompt)
        vector = vector.to(device=self.device, dtype=self.dtype)

        # First pass with hook
        def hook_fn(module, input, output):
            hidden = output[0]
            hidden[0, position, :] = vector
            return (hidden,) + output[1:]

        h = self.get_layer_module(layer).register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = self.model(input_ids).logits

        h.remove()

        # Greedy decode first token
        next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        generated = [next_token]

        # Continue generating without hook
        all_ids = torch.cat([input_ids, next_token], dim=1)
        for _ in range(max_new_tokens - 1):
            with torch.no_grad():
                logits = self.model(all_ids).logits
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated.append(next_token)
            all_ids = torch.cat([all_ids, next_token], dim=1)

            # Stop on newline or EOS
            tok_str = self.tokenizer.decode(next_token[0])
            if "\n" in tok_str or next_token.item() == self.tokenizer.eos_token_id:
                break

        gen_ids = torch.cat(generated, dim=1)[0]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def load_model(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    device: str = "cuda:3",
    dtype: torch.dtype = torch.float16,
) -> HookedModel:
    """Load model and tokenizer, return wrapped HookedModel."""
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    wrapped = HookedModel(model, tokenizer)
    print(f"Loaded on {device}. {wrapped.n_layers} layers, d_model={wrapped.d_model}")
    return wrapped


def get_model_info(model: HookedModel) -> dict:
    """Extract key model configuration."""
    return {
        "name": model.model.config._name_or_path,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "vocab_size": model.vocab_size,
        "device": str(model.device),
    }
