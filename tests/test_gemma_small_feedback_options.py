import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rlpi.attack.learners.common import feedback_utils


class GemmaSmallFeedbackOptionsTests(unittest.TestCase):
    def test_request_options_are_model_specific(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Answer: 0"),
                    logprobs=SimpleNamespace(content=[]),
                )
            ]
        )
        create = MagicMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        openai_module = SimpleNamespace(
            OpenAI=MagicMock(return_value=client)
        )

        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(feedback_utils, "openai", openai_module),
        ):
            feedback_utils._compare_with_openai(
                "prompt", "gemma-small", verbose=False
            )
            feedback_utils._compare_with_openai(
                "prompt", "unrelated-model", verbose=False
            )

        gemma_kwargs = create.call_args_list[0].kwargs
        unrelated_kwargs = create.call_args_list[1].kwargs
        self.assertEqual(
            gemma_kwargs["extra_body"],
            {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )
        self.assertNotIn("extra_body", unrelated_kwargs)


if __name__ == "__main__":
    unittest.main()
