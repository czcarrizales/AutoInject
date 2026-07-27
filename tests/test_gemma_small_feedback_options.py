import unittest
from math import exp
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rlpi.attack.learners.common import feedback_utils
from rlpi.attack.learners.trl_suffix import reward_utils


class GemmaSmallFeedbackOptionsTests(unittest.TestCase):
    @staticmethod
    def _response(
        content="Answer: 0", logprob_0=None, logprob_1=None
    ):
        logprobs = []
        if logprob_0 is not None and logprob_1 is not None:
            logprobs = [
                SimpleNamespace(token="Answer:", top_logprobs=[]),
                SimpleNamespace(
                    token="0",
                    top_logprobs=[
                        SimpleNamespace(token="0", logprob=logprob_0),
                        SimpleNamespace(token="1", logprob=logprob_1),
                    ],
                ),
            ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    logprobs=SimpleNamespace(content=logprobs),
                )
            ]
        )

    @staticmethod
    def _openai_module(create):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        return SimpleNamespace(OpenAI=MagicMock(return_value=client))

    def test_request_options_are_model_specific(self):
        response = self._response()
        create = MagicMock(return_value=response)
        openai_module = self._openai_module(create)

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

    def test_transient_503_responses_are_retried_until_success(self):
        transient_error = RuntimeError("no healthy upstream")
        transient_error.status_code = 503
        create = MagicMock(
            side_effect=[
                transient_error,
                transient_error,
                self._response(logprob_0=-0.2, logprob_1=-1.2),
            ]
        )
        on_model_call = MagicMock()

        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
            patch.object(feedback_utils.time, "sleep") as sleep,
        ):
            prob_1, prob_0, is_better, _ = (
                feedback_utils._compare_with_openai(
                    "prompt",
                    "gemma-small",
                    verbose=False,
                    on_model_call=on_model_call,
                )
            )

        self.assertEqual(create.call_count, 3)
        on_model_call.assert_called_once_with()
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertAlmostEqual(prob_1, exp(-1.2) / (exp(-0.2) + exp(-1.2)))
        self.assertAlmostEqual(prob_0, exp(-0.2) / (exp(-0.2) + exp(-1.2)))
        self.assertFalse(is_better)

    def test_transient_retry_exhaustion_raises(self):
        transient_error = RuntimeError("no healthy upstream")
        transient_error.status_code = 503
        create = MagicMock(side_effect=transient_error)

        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
            patch.dict(
                feedback_utils.os.environ,
                {"AUTOINJECT_FEEDBACK_MAX_RETRY_SECONDS": "1"},
            ),
            patch.object(
                feedback_utils.time,
                "monotonic",
                side_effect=[0.0, 0.0, 1.0],
            ),
            patch.object(feedback_utils.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "no healthy upstream"):
                feedback_utils._compare_with_openai(
                    "prompt", "gemma-small", verbose=False
                )

        self.assertEqual(create.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_permanent_error_is_not_retried(self):
        permanent_error = RuntimeError("invalid API key")
        permanent_error.status_code = 401
        create = MagicMock(side_effect=permanent_error)

        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
            patch.object(feedback_utils.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid API key"):
                feedback_utils._compare_with_openai(
                    "prompt", "gemma-small", verbose=False
                )

        create.assert_called_once()
        sleep.assert_not_called()

    def test_zero_zero_probabilities_are_rejected(self):
        create = MagicMock(return_value=self._response(content="unparseable"))

        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
        ):
            with self.assertRaisesRegex(ValueError, "zero/zero"):
                feedback_utils._compare_with_openai(
                    "prompt", "gemma-small", verbose=False
                )

    def test_valid_probabilities_are_preserved(self):
        response = self._response(logprob_0=-2.0, logprob_1=-0.5)
        create = MagicMock(return_value=response)

        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
        ):
            actual = feedback_utils._compare_with_openai(
                "prompt", "gemma-small", verbose=False
            )

        expected_prob_1 = exp(-0.5) / (exp(-2.0) + exp(-0.5))
        expected_prob_0 = exp(-2.0) / (exp(-2.0) + exp(-0.5))
        self.assertAlmostEqual(actual[0], expected_prob_1)
        self.assertAlmostEqual(actual[1], expected_prob_0)
        self.assertTrue(actual[2])

    def test_grpo_feedback_failure_is_not_converted_to_zero_zero(self):
        with patch.object(
            reward_utils,
            "compare_suffix_with_previous",
            side_effect=RuntimeError("feedback unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "feedback unavailable"):
                reward_utils._get_gpt_score_if_enabled(
                    completion="new",
                    injection_goal="goal",
                    feedback_model="gemma-small",
                    gpt_config={"enabled": True},
                    best_suffix="previous",
                )


if __name__ == "__main__":
    unittest.main()
