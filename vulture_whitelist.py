# Vulture whitelist — verified FALSE POSITIVES (do not delete the referenced symbols).
#
# Generated during the dead-code cleanup and manually verified per entry.
# Usage:  vulture codebase/src/gnn vulture_whitelist.py --min-confidence 60
# A clean run (only the intentional open SUGGEST items remain) proves no
# unreviewed dead code regressed.
#
# Categories of the entries below:
#   * nn.Module.forward            -> invoked by PyTorch via __call__
#   * _on_step                     -> Stable-Baselines3 BaseCallback hook
#   * get_attr/set_attr/env_method/env_is_wrapped -> SB3 VecEnv API (must implement)
#   * on_validation_batch_end      -> PyTorch-Lightning / GraphGym Trainer hook
#   * weighted_cross_entropy_loss, ExpressionNodeEncoder -> GraphGym registry
#                                     (@register_loss / @register_node_encoder, dispatched by string)
#   * performance_np               -> referenced via eval(f"performance_np.{agg}()")
#   * __class__                    -> deliberate Data-subclass upcast (load-bearing)
#   * loader_graphgym .slices/.*_graph_index/.curated_eval_* -> GraphGym InMemoryDataset
#                                     / config-driven attributes set dynamically
#   * .return_value/.side_effect/.enrich -> unittest.mock attributes in tests
#   * clear_loader_cache           -> pytest autouse fixture (invoked by the framework)

_.forward  # unused method (codebase/src/gnn/benchmark_inference_time.py:93)
_.forward  # unused method (codebase/src/gnn/benchmark_inference_time.py:115)
_.get_attr  # unused method (codebase/src/gnn/reinforcement_learning/mathematica_vec_env.py:476)
_.set_attr  # unused method (codebase/src/gnn/reinforcement_learning/mathematica_vec_env.py:480)
_.env_method  # unused method (codebase/src/gnn/reinforcement_learning/mathematica_vec_env.py:485)
_.env_is_wrapped  # unused method (codebase/src/gnn/reinforcement_learning/mathematica_vec_env.py:490)
_._on_step  # unused method (codebase/src/gnn/reinforcement_learning/ppo_optuna_callback.py:36)
_.forward  # unused method (codebase/src/gnn/reinforcement_learning/sb3_extractor.py:17)
_._on_step  # unused method (codebase/src/gnn/reinforcement_learning/train_best.py:76)
_.forward  # unused method (codebase/src/gnn/shared/models/gnn_backbones.py:102)
_.forward  # unused method (codebase/src/gnn/shared/models/gnn_backbones.py:162)
_.__class__  # unused attribute (codebase/src/gnn/shared/utils/graph_converter.py:399)
_.is_split  # unused attribute (codebase/src/gnn/supervised_learning/aggregate_graphgym.py:239)
weighted_cross_entropy_loss  # unused function (codebase/src/gnn/supervised_learning/loader_graphgym.py:32)
_.classification_binary  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:192)
_.on_validation_batch_end  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:212)
ExpressionNodeEncoder  # unused class (codebase/src/gnn/supervised_learning/loader_graphgym.py:225)
_.forward  # unused method (codebase/src/gnn/supervised_learning/loader_graphgym.py:238)
_.forward  # unused method (codebase/src/gnn/supervised_learning/loader_graphgym.py:276)
_.slices  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:345)
_.train_graph_index  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:346)
_.val_graph_index  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:347)
_.test_graph_index  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:348)
_.curated_eval_period  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:390)
_.curated_eval_on_test_highscore  # unused attribute (codebase/src/gnn/supervised_learning/loader_graphgym.py:391)
_.on_validation_batch_end  # unused method (codebase/src/gnn/supervised_learning/main_graphgym.py:365)
performance_np  # unused variable (codebase/src/gnn/supervised_learning/run_results/eval_metrics.py:29)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_audit_fixes.py:130)
_.enrich  # unused attribute (codebase/src/gnn/tests/test_eval_metrics.py:53)
clear_loader_cache  # unused function (codebase/src/gnn/tests/test_graph_loader.py:9)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_mathematica_env_finalize.py:33)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_mathematica_env_finalize.py:34)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_mathematica_env_finalize.py:36)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_mathematica_vec_env.py:39)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_mathematica_vec_env.py:40)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_mathematica_vec_env.py:42)
clear_loader_cache  # unused function (codebase/src/gnn/tests/test_training_pipeline_smoke.py:14)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_trial_switch.py:47)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_trial_switch.py:48)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_trial_switch.py:50)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:52)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:53)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:72)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:77)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:82)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:87)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:106)
_.side_effect  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:109)
_.return_value  # unused attribute (codebase/src/gnn/tests/test_unified_loader.py:115)
