from backend.app.core import llm

def test_build_prompt_structure():
    prompt = llm.build_prompt("What is X?", ["Ctx1", "Ctx2"])
    assert "<|system|>" in prompt
    assert "<|context|>" in prompt
    assert "<|question|>" in prompt
    assert "What is X?" in prompt


def test_get_config_keys():
    cfg = llm.get_config()
    for key in ["model", "backend", "device", "max_input_tokens", "max_generation_tokens"]:
        assert key in cfg
