"""CrossModelConnector: prefill a long prompt with a small same-family model and map its KV cache into the served
model's paged KV blocks (closed-form ridge maps, RoPE strip/re-apply, optional sink template and native tail)."""
from .connector import CrossModelConnector  # noqa: F401
