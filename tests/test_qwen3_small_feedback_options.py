import subprocess
import unittest
from math import exp
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rlpi.attack.learners.common import feedback_utils
from rlpi.attack.learners.trl_suffix import reward_utils
from rlpi.attack.learners.trl_suffix.learner import TRLSuffixLearner


FROZEN_COMMIT = "33c2cbdb332d0a9b06f498d4eeac1f11c4491247"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class Qwen3SmallFrozenFidelityTests(unittest.TestCase):
    @staticmethod
    def _response(content="analysis\nAnswer: 1", logprobs=()):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    logprobs=SimpleNamespace(content=list(logprobs)),
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

    def _compare(self, response, model="qwen3-small"):
        create = MagicMock(return_value=response)
        openai_module = self._openai_module(create)
        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(feedback_utils, "openai", openai_module),
        ):
            result = feedback_utils._compare_with_openai(
                "prompt", model, verbose=False
            )
        return result, create.call_args.kwargs, openai_module

    def test_thinking_option_is_limited_to_gemma_and_qwen(self):
        expected = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        for model in ("gemma-small", "qwen3-small"):
            with self.subTest(model=model):
                _, request, _ = self._compare(
                    self._response(), model=model
                )
                self.assertEqual(request["extra_body"], expected)

        _, request, _ = self._compare(
            self._response(), model="unrelated-model"
        )
        self.assertNotIn("extra_body", request)

    def test_qwen_request_keeps_frozen_parameters_and_client_defaults(self):
        _, request, openai_module = self._compare(self._response())
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], 350)
        self.assertIs(request["logprobs"], True)
        self.assertEqual(request["top_logprobs"], 5)
        openai_module.OpenAI.assert_called_once_with(api_key=None)

    def test_frozen_logprob_extraction_is_preserved(self):
        answer_token = SimpleNamespace(
            token="1",
            top_logprobs=[
                SimpleNamespace(token="0", logprob=-2.0),
                SimpleNamespace(token="1", logprob=-0.5),
            ],
        )
        result, _, _ = self._compare(
            self._response(
                logprobs=[
                    SimpleNamespace(token="Answer:", top_logprobs=[]),
                    answer_token,
                ]
            )
        )
        expected_1 = exp(-0.5) / (exp(-2.0) + exp(-0.5))
        expected_0 = exp(-2.0) / (exp(-2.0) + exp(-0.5))
        self.assertAlmostEqual(result[0], expected_1)
        self.assertAlmostEqual(result[1], expected_0)
        self.assertTrue(result[2])

    def test_textual_fallbacks_and_permissive_answer_parsing(self):
        cases = (
            ("reason\nAnswer: 1 trailing text", (1.0, 0.0, True)),
            ("reason\nAnswer: 0 trailing text", (0.0, 1.0, False)),
            ("reason\nAnswer: 01", (0.0, 1.0, False)),
        )
        for content, expected in cases:
            with self.subTest(content=content):
                result, _, _ = self._compare(self._response(content=content))
                self.assertEqual(result[:3], expected)
                self.assertEqual(result[3], "reason")

    def test_unusable_text_returns_frozen_zero_zero_sentinel(self):
        result, _, _ = self._compare(
            self._response(content="malformed output")
        )
        self.assertEqual(result, (0.0, 0.0, False, ""))

    def test_direct_field_access_preserves_incidental_exceptions(self):
        malformed = (
            SimpleNamespace(),
            SimpleNamespace(choices=[]),
            self._response(content=None),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Answer: 1")
                    )
                ]
            ),
        )
        for response in malformed:
            with self.subTest(response=response):
                with self.assertRaises((AttributeError, IndexError)):
                    self._compare(response)

    def test_openai_unavailable_returns_frozen_neutral_result(self):
        with patch.object(feedback_utils, "HAS_OPENAI", False):
            result = feedback_utils._compare_with_openai(
                "prompt", "qwen3-small", verbose=False
            )
        self.assertEqual(result, (0.0, 0.0, False, ""))

    def test_grpo_request_exception_falls_back_to_empirical_only(self):
        with patch.object(
            reward_utils,
            "compare_suffix_with_previous",
            side_effect=RuntimeError("request failed"),
        ):
            result = reward_utils._get_gpt_score_if_enabled(
                completion="new",
                injection_goal="goal",
                feedback_model="qwen3-small",
                gpt_config={"enabled": True},
                best_suffix="old",
            )
        self.assertEqual(result, (None, 0.0, 0.0, False, ""))

        reward = feedback_utils.compute_adaptive_reward(
            security_score=0.6,
            utility_score=0.2,
            gpt_score=result[0],
            gpt_config={"enabled": True},
        )
        self.assertAlmostEqual(reward, 0.7 * 0.6 + 0.3 * 0.2)

    def test_content_none_uses_the_same_grpo_fallback(self):
        create = MagicMock(return_value=self._response(content=None))
        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
        ):
            result = reward_utils._get_gpt_score_if_enabled(
                completion="new",
                injection_goal="goal",
                feedback_model="qwen3-small",
                gpt_config={"enabled": True},
                best_suffix="old",
            )
        self.assertEqual(result, (None, 0.0, 0.0, False, ""))

    def test_outer_user_task_configuration_is_still_discarded(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.current_suffix = "new"
        learner.best_suffix = "old"
        learner.injection_task = SimpleNamespace(_original_goal="goal")
        learner.verbose = False
        learner.experiment_reporting = None
        learner._get_gpt_config = MagicMock(
            side_effect=[
                {"enabled": True, "model": "qwen3-small"},
                {"enabled": True, "model": "qwen3-small"},
            ]
        )

        with patch(
            "rlpi.attack.learners.trl_suffix.learner."
            "compute_gpt_feedback_for_iteration",
            return_value=(0.0, 0.0, False, ""),
        ) as compute:
            learner._compute_gpt_feedback("specific user task")

        self.assertEqual(learner._get_gpt_config.call_count, 2)
        self.assertNotIn(
            "user_task_str",
            compute.call_args.kwargs["gpt_config"],
        )

    def test_first_cycle_sentinel_and_reward_formula_are_unchanged(self):
        gpt_config = {
            "enabled": True,
            "model": "qwen3-small",
            "thresholds": {
                "sparse": 0.1,
                "medium": 0.3,
                "dense": 0.5,
            },
            "weights": {
                "sparse": 0.8,
                "medium": 0.5,
                "transition": 0.3,
                "dense": 0.1,
            },
        }
        self.assertEqual(
            feedback_utils.compute_gpt_feedback_for_iteration(
                current_suffix="new",
                previous_suffix=None,
                injection_goal="goal",
                gpt_config=gpt_config,
            ),
            (0.0, 0.0, False, ""),
        )
        self.assertAlmostEqual(
            feedback_utils.compute_adaptive_reward(
                security_score=0.2,
                utility_score=0.4,
                gpt_score=0.8,
                gpt_config=gpt_config,
            ),
            0.5 * (0.7 * 0.2 + 0.3 * 0.4) + 0.5 * 0.8,
        )

    def test_runtime_files_match_frozen_reference_except_qwen_option(self):
        paths = (
            "src/rlpi/agentdojo/config/learner/trl_suffix.yaml",
            "src/rlpi/agentdojo/config/learner/"
            "trl_suffix_meta_secalign_repro.yaml",
            "src/rlpi/attack/learners/trl_suffix/learner.py",
            "src/rlpi/attack/learners/trl_suffix/reward_utils.py",
            "src/rlpi/attack/learners/trl_suffix/utils.py",
        )
        for path in paths:
            with self.subTest(path=path):
                frozen = subprocess.run(
                    ["git", "show", f"{FROZEN_COMMIT}:{path}"],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertEqual(
                    (REPOSITORY_ROOT / path).read_text(), frozen
                )

        feedback_path = (
            "src/rlpi/attack/learners/common/feedback_utils.py"
        )
        frozen = subprocess.run(
            ["git", "show", f"{FROZEN_COMMIT}:{feedback_path}"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        corrected = (
            REPOSITORY_ROOT / feedback_path
        ).read_text().replace(
            'if model in {"gemma-small", "qwen3-small"}:',
            'if model == "gemma-small":',
            1,
        )
        self.assertEqual(corrected, frozen)


if __name__ == "__main__":
    unittest.main()
