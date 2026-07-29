import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rlpi.attack.learners.common import feedback_utils
from rlpi.attack.learners.trl_suffix import reward_utils
from rlpi.attack.learners.trl_suffix.learner import TRLSuffixLearner


class Qwen3SmallFeedbackOptionsTests(unittest.TestCase):
    @staticmethod
    def _response(
        content="Answer: 1",
        alternatives=(("0", -2.0), ("1", -0.2)),
        emitted_token="1",
        emitted_logprob=None,
        include_logprobs=True,
    ):
        choice = SimpleNamespace(message=SimpleNamespace(content=content))
        if include_logprobs:
            if emitted_logprob is None:
                matching_logprobs = [
                    logprob
                    for token, logprob in alternatives or ()
                    if token.strip() == emitted_token.strip()
                    and isinstance(logprob, (int, float))
                    and math.isfinite(logprob)
                ]
                emitted_logprob = (
                    max(matching_logprobs)
                    if matching_logprobs
                    else -0.2
                )
            token_data = SimpleNamespace(
                token=emitted_token,
                logprob=emitted_logprob,
                top_logprobs=(
                    None
                    if alternatives is None
                    else [
                        SimpleNamespace(token=token, logprob=logprob)
                        for token, logprob in alternatives
                    ]
                ),
            )
            choice.logprobs = SimpleNamespace(content=[token_data])
        return SimpleNamespace(choices=[choice])

    @staticmethod
    def _openai_module(create):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        return SimpleNamespace(OpenAI=MagicMock(return_value=client))

    def _compare(self, response, model="qwen3-small", **kwargs):
        create = MagicMock(return_value=response)
        with (
            patch.object(feedback_utils, "HAS_OPENAI", True),
            patch.object(
                feedback_utils, "openai", self._openai_module(create)
            ),
        ):
            result = feedback_utils._compare_with_openai(
                "prompt", model, verbose=False, **kwargs
            )
        return result, create.call_args.kwargs

    def test_reasoning_options_are_model_and_configuration_specific(self):
        cases = [
            (
                "gemma-small",
                None,
                {"chat_template_kwargs": {"enable_thinking": False}},
            ),
            (
                "qwen3-small",
                False,
                {"chat_template_kwargs": {"enable_thinking": False}},
            ),
            (
                "qwen3-small",
                True,
                {"chat_template_kwargs": {"enable_thinking": True}},
            ),
        ]
        for model, reasoning_enabled, expected_extra_body in cases:
            with self.subTest(model=model, enabled=reasoning_enabled):
                _, request = self._compare(
                    self._response(),
                    model=model,
                    feedback_reasoning_enabled=reasoning_enabled,
                )
                self.assertEqual(
                    request["extra_body"], expected_extra_body
                )
                self.assertIs(request["logprobs"], True)
                self.assertEqual(request["top_logprobs"], 5)

        _, unrelated_request = self._compare(
            self._response(),
            model="unrelated-model",
            feedback_reasoning_enabled=None,
        )
        self.assertNotIn("extra_body", unrelated_request)

    def test_configured_logprob_request_values_reach_client(self):
        _, request = self._compare(
            self._response(),
            feedback_logprobs=False,
            feedback_top_logprobs=3,
        )
        self.assertIs(request["logprobs"], False)
        self.assertEqual(request["top_logprobs"], 3)

    def test_content_none_raises_controlled_exception(self):
        response = self._response(content=None)
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError,
            r"message\.content is None",
        ):
            self._compare(response)

    def test_missing_and_empty_choices_raise_controlled_exception(self):
        for response in (SimpleNamespace(), SimpleNamespace(choices=[])):
            with self.subTest(response=response):
                with self.assertRaisesRegex(
                    feedback_utils.FeedbackValidationError, "no choices"
                ):
                    self._compare(
                        response,
                        feedback_strict_logprob_extraction=True,
                    )

    def test_strict_final_answer_rejects_malformed_line(self):
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError, "final line"
        ):
            self._compare(
                self._response(content="Answer: 1 extra"),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_final_answer_rejects_answer_01(self):
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError, "final line"
        ):
            self._compare(
                self._response(content="Answer: 01"),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_mode_rejects_missing_logprobs(self):
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError,
            r"choice\.logprobs is missing",
        ):
            self._compare(
                self._response(include_logprobs=False),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_mode_rejects_none_and_empty_logprobs_content(self):
        for content in (None, []):
            response = self._response()
            response.choices[0].logprobs.content = content
            with self.subTest(content=content):
                with self.assertRaisesRegex(
                    feedback_utils.FeedbackValidationError,
                    r"logprobs\.content is empty",
                ):
                    self._compare(
                        response,
                        feedback_strict_logprob_extraction=True,
                    )

    def test_strict_mode_rejects_missing_emitted_answer_token(self):
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError,
            "answer digit token was not found",
        ):
            self._compare(
                self._response(emitted_token="not-a-digit"),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_mode_rejects_none_and_empty_top_logprobs(self):
        for alternatives in (None, ()):
            with self.subTest(alternatives=alternatives):
                with self.assertRaisesRegex(
                    feedback_utils.FeedbackValidationError,
                    "both digit 0 and digit 1",
                ):
                    self._compare(
                        self._response(alternatives=alternatives),
                        feedback_strict_logprob_extraction=True,
                    )

    def test_strict_mode_rejects_missing_alternative_digit(self):
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError,
            "both digit 0 and digit 1",
        ):
            self._compare(
                self._response(alternatives=(("1", -0.2),)),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_mode_rejects_non_numeric_logprob(self):
        with self.assertRaisesRegex(
            feedback_utils.FeedbackValidationError, "must be numeric"
        ):
            self._compare(
                self._response(
                    alternatives=(("0", "not-numeric"), ("1", -0.2))
                ),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_mode_rejects_non_finite_logprobs(self):
        for non_finite in (math.nan, math.inf, -math.inf):
            with self.subTest(logprob=non_finite):
                with self.assertRaisesRegex(
                    feedback_utils.FeedbackValidationError,
                    "must be finite",
                ):
                    self._compare(
                        self._response(
                            alternatives=(
                                ("0", non_finite),
                                ("1", -0.2),
                            )
                        ),
                        feedback_strict_logprob_extraction=True,
                    )

    def test_strict_mode_rejects_invalid_softmax_denominator(self):
        for weights in ((0.0, 0.0), (math.inf, 1.0)):
            with self.subTest(weights=weights):
                with (
                    patch.object(
                        feedback_utils.np,
                        "exp",
                        side_effect=weights,
                    ),
                    self.assertRaisesRegex(
                        feedback_utils.FeedbackValidationError,
                        "denominator must be finite and positive",
                    ),
                ):
                    self._compare(
                        self._response(),
                        feedback_strict_logprob_extraction=True,
                    )

    def test_strict_mode_strips_tokens_and_uses_stable_softmax(self):
        result, _ = self._compare(
            self._response(
                alternatives=((" 0 ", -10001.0), ("\t1\n", -10000.0)),
                emitted_token=" 1\n",
            ),
            feedback_strict_logprob_extraction=True,
        )
        prob_1, prob_0, is_better, _ = result

        self.assertTrue(math.isfinite(prob_0))
        self.assertTrue(math.isfinite(prob_1))
        self.assertAlmostEqual(prob_0 + prob_1, 1.0)
        self.assertAlmostEqual(prob_1, math.exp(1.0) / (1.0 + math.exp(1.0)))
        self.assertTrue(is_better)

    def test_duplicate_normalized_zero_uses_maximum_logprob(self):
        result, _ = self._compare(
            self._response(
                alternatives=(
                    ("0", -9.0),
                    (" 0", -1.0),
                    ("1", -2.0),
                )
            ),
            feedback_strict_logprob_extraction=True,
        )
        self.assertAlmostEqual(
            result[1], math.exp(1.0) / (1.0 + math.exp(1.0))
        )

    def test_duplicate_normalized_one_uses_maximum_logprob(self):
        result, _ = self._compare(
            self._response(
                alternatives=(
                    ("0", -2.0),
                    ("1", -9.0),
                    (" 1", -1.0),
                ),
                emitted_token=" 1",
            ),
            feedback_strict_logprob_extraction=True,
        )
        self.assertAlmostEqual(
            result[0], math.exp(1.0) / (1.0 + math.exp(1.0))
        )

    def test_duplicate_order_does_not_change_probabilities_or_reward(self):
        alternatives = (
            ("0", -8.0),
            (" 0", -1.0),
            ("1", -2.0),
            (" 1", -7.0),
        )
        forward, _ = self._compare(
            self._response(alternatives=alternatives),
            feedback_strict_logprob_extraction=True,
        )
        reversed_result, _ = self._compare(
            self._response(alternatives=tuple(reversed(alternatives))),
            feedback_strict_logprob_extraction=True,
        )

        self.assertAlmostEqual(forward[0], reversed_result[0])
        self.assertAlmostEqual(forward[1], reversed_result[1])
        gpt_config = {
            "enabled": True,
            "thresholds": {"sparse": 0.1, "medium": 0.3, "dense": 0.5},
            "weights": {
                "sparse": 0.8,
                "medium": 0.5,
                "transition": 0.3,
                "dense": 0.1,
            },
        }
        forward_reward = feedback_utils.compute_adaptive_reward(
            security_score=0.0,
            utility_score=0.0,
            gpt_score=forward[0],
            gpt_config=gpt_config,
        )
        reversed_reward = feedback_utils.compute_adaptive_reward(
            security_score=0.0,
            utility_score=0.0,
            gpt_score=reversed_result[0],
            gpt_config=gpt_config,
        )
        self.assertAlmostEqual(forward_reward, reversed_reward)

    def test_strict_mode_never_uses_textual_one_hot_fallback(self):
        with self.assertRaises(feedback_utils.FeedbackValidationError):
            self._compare(
                self._response(content="Answer: 1", alternatives=()),
                feedback_strict_logprob_extraction=True,
            )

    def test_strict_mode_validates_logprob_request_configuration(self):
        cases = [
            (
                {"feedback_logprobs": False},
                "feedback_logprobs=True",
            ),
            (
                {"feedback_top_logprobs": 1},
                "integer of at least 2",
            ),
            (
                {"feedback_top_logprobs": True},
                "integer of at least 2",
            ),
        ]
        for options, message in cases:
            create = MagicMock(return_value=self._response())
            with (
                self.subTest(options=options),
                patch.object(feedback_utils, "HAS_OPENAI", True),
                patch.object(
                    feedback_utils,
                    "openai",
                    self._openai_module(create),
                ),
                self.assertRaisesRegex(
                    feedback_utils.FeedbackConfigurationError, message
                ),
            ):
                feedback_utils._compare_with_openai(
                    "prompt",
                    "qwen3-small",
                    verbose=False,
                    feedback_strict_logprob_extraction=True,
                    **options,
                )
            create.assert_not_called()

    def test_non_strict_historical_text_fallback_remains_available(self):
        result, _ = self._compare(
            self._response(content="analysis\nAnswer: 0", alternatives=())
        )
        self.assertEqual(result[:3], (0.0, 1.0, False))

    def test_grpo_threads_hosted_feedback_configuration(self):
        with patch.object(
            reward_utils,
            "compare_suffix_with_previous",
            return_value=(0.8, 0.2, True, "reasoning"),
        ) as compare:
            reward_utils._get_gpt_score_if_enabled(
                completion="new",
                injection_goal="goal",
                feedback_model="qwen3-small",
                gpt_config={
                    "enabled": True,
                    "feedback_reasoning_enabled": False,
                    "feedback_strict_logprob_extraction": True,
                    "feedback_logprobs": True,
                    "feedback_top_logprobs": 3,
                },
                best_suffix="old",
            )

        self.assertIs(
            compare.call_args.kwargs["feedback_reasoning_enabled"], False
        )
        self.assertIs(
            compare.call_args.kwargs[
                "feedback_strict_logprob_extraction"
            ],
            True,
        )
        self.assertIs(compare.call_args.kwargs["feedback_logprobs"], True)
        self.assertEqual(
            compare.call_args.kwargs["feedback_top_logprobs"], 3
        )

    def test_legacy_positional_constructor_binding_is_preserved(self):
        legacy_args = (
            "attack-model",
            "feedback-model",
            "victim",
            "user",
            "cpu",
            7,
            21,
            0.6,
            0.8,
            11,
            4,
            2,
            1e-6,
            3,
            5,
            6,
            0.2,
            2.0,
            1e-6,
            0.9,
            1,
            False,
            None,
            False,
            0,
            True,
            0.11,
            0.22,
            0.33,
            0.44,
            0.55,
            0.66,
            0.77,
        )
        with (
            patch.object(
                TRLSuffixLearner, "_init_model_and_tokenizer"
            ),
            patch.object(TRLSuffixLearner, "_init_grpo_trainer"),
            patch.object(TRLSuffixLearner, "_init_output_files"),
        ):
            learner = TRLSuffixLearner(*legacy_args)

        self.assertEqual(learner.gpt_sparse_threshold, 0.11)
        self.assertEqual(learner.gpt_medium_threshold, 0.22)
        self.assertEqual(learner.gpt_dense_threshold, 0.33)
        self.assertEqual(learner.gpt_sparse_weight, 0.44)
        self.assertEqual(learner.gpt_medium_weight, 0.55)
        self.assertEqual(learner.gpt_transition_weight, 0.66)
        self.assertEqual(learner.gpt_dense_weight, 0.77)
        self.assertIsNone(learner.feedback_reasoning_enabled)
        self.assertIs(learner.feedback_strict_logprob_extraction, False)
        self.assertIs(learner.feedback_logprobs, True)
        self.assertEqual(learner.feedback_top_logprobs, 5)

    def test_checkpoint_provenance_matches_request_configuration(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.max_suffix_length = 20
        learner.temperature = 0.7
        learner.top_p = 0.9
        learner.grpo_num_generations = 8
        learner.grpo_num_iterations = 1
        learner.grpo_learning_rate = 1e-7
        learner.min_experiences_for_training = 10
        learner.feedback_model = "qwen3-small"
        learner.feedback_reasoning_enabled = False
        learner.feedback_strict_logprob_extraction = True
        learner.feedback_logprobs = True
        learner.feedback_top_logprobs = 3
        learner.experience_count = 0
        learner.training_count = 0
        learner.best_suffix = None
        learner.best_reward = -math.inf
        learner.early_stopped = False
        learner.attack_model_name = "attack-model"
        learner.policy = SimpleNamespace(
            state_dict=lambda: {},
            dtype="float32",
        )

        _, request = self._compare(
            self._response(),
            feedback_reasoning_enabled=(
                learner.feedback_reasoning_enabled
            ),
            feedback_strict_logprob_extraction=(
                learner.feedback_strict_logprob_extraction
            ),
            feedback_logprobs=learner.feedback_logprobs,
            feedback_top_logprobs=learner.feedback_top_logprobs,
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            with patch(
                "rlpi.attack.learners.trl_suffix.learner.torch.save"
            ):
                learner.save_model(str(checkpoint_path))
            state = json.loads(
                (Path(directory) / "checkpoint_state.json").read_text()
            )

        snapshot = state["config_snapshot"]
        self.assertEqual(
            snapshot["feedback_model"], learner.feedback_model
        )
        self.assertEqual(
            snapshot["feedback_reasoning_enabled"],
            learner.feedback_reasoning_enabled,
        )
        self.assertEqual(
            snapshot["feedback_strict_logprob_extraction"],
            learner.feedback_strict_logprob_extraction,
        )
        self.assertEqual(
            snapshot["feedback_logprobs"], request["logprobs"]
        )
        self.assertEqual(
            snapshot["feedback_top_logprobs"],
            request["top_logprobs"],
        )

    def test_outer_loop_retains_user_task_in_gpt_config(self):
        learner = TRLSuffixLearner.__new__(TRLSuffixLearner)
        learner.current_suffix = "new suffix"
        learner.best_suffix = "old suffix"
        learner.injection_task = SimpleNamespace(_original_goal="goal")
        learner.verbose = False
        learner.experiment_reporting = None
        learner._get_gpt_config = MagicMock(
            return_value={"enabled": True, "model": "qwen3-small"}
        )

        with patch(
            "rlpi.attack.learners.trl_suffix.learner."
            "compute_gpt_feedback_for_iteration",
            return_value=(0.8, 0.2, True, "reasoning"),
        ) as compute:
            result = learner._compute_gpt_feedback("specific user task")

        self.assertEqual(result, (0.8, 0.2, True, "reasoning"))
        self.assertEqual(
            compute.call_args.kwargs["gpt_config"]["user_task_str"],
            "specific user task",
        )


if __name__ == "__main__":
    unittest.main()
