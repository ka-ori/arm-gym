from kaggle.dataset import render_prompt, user_prompt
from kaggle.reward_fn import extract_assembly, format_reward


class DummyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def test_render_prompt_disables_thinking_and_prefills_assembly() -> None:
    tokenizer = DummyTokenizer()

    prompt = render_prompt(
        tokenizer,
        "void f(void) {}",
        ".text\n.global f\nf:\n\tret\n",
    )

    assert tokenizer.kwargs["add_generation_prompt"] is True
    assert tokenizer.kwargs["enable_thinking"] is False
    assert prompt.endswith("<assembly>\n")


def test_extract_assembly_accepts_prefilled_completion_body() -> None:
    completion = "\t.text\n.global f\nf:\n\tret\n</assembly>\nextra ignored"

    assert extract_assembly(completion) == ".text\n.global f\nf:\n\tret"


def test_format_reward_prefers_closed_assembly_blocks() -> None:
    rewards = format_reward(
        completions=[
            "\t.text\n.global f\nf:\n\tret\n</assembly>",
            "I will explain the optimization first.",
        ]
    )

    assert rewards == [0.2, 0.0]


def test_user_prompt_includes_baseline_fallback() -> None:
    prompt = user_prompt("int f(void){return 0;}", "f:\n\tret\n")

    assert "copy the baseline assembly exactly" in prompt
    assert "...AArch64 assembly only..." not in prompt
