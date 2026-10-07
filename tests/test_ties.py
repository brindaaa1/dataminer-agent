"""赛跑打平：与最优差距小于 MDE 的模型效果相当，按固定偏好顺序推荐，不让 LLM 在噪声里挑（v7–v9：20 次运行 18 次打平）。"""
from evaluation.ties import race_tie

ORDER = ["lgbm", "random_forest", "lr_scorecard", "catboost"]


def test_within_mde_is_a_tie_and_preference_order_decides():
    """v9b hotel s1：随机森林 0.8337、catboost 0.8256，差 0.0081 < MDE 0.0083，打平；按偏好选随机森林之前的 lgbm 不在打平里。"""
    t = race_tie({"catboost": 0.8256, "lgbm": 0.8186, "random_forest": 0.8337}, 0.0083, ORDER)
    assert t["tied"] == ["random_forest", "catboost"] and t["recommended"] == "random_forest" and t["best"] == "random_forest"


def test_tie_records_which_models_raced():
    assert race_tie({"lgbm": 0.65}, 0.01, ORDER)["models"] == ["lgbm"]


def test_clear_winner_is_recommended_even_if_last_in_order():
    t = race_tie({"catboost": 0.8366, "lgbm": 0.8241, "random_forest": 0.8282}, 0.0083, ORDER)
    assert t["tied"] == ["catboost"] and t["recommended"] == "catboost"


def test_preferred_model_wins_a_tie_against_a_higher_score():
    t = race_tie({"lgbm": 0.6570, "catboost": 0.6602, "random_forest": 0.6516}, 0.0099, ORDER)
    assert t["recommended"] == "lgbm" and t["tied"] == ["lgbm", "random_forest", "catboost"]


def test_models_not_in_order_go_last_by_score_and_no_mde_means_no_tie():
    t = race_tie({"xgb": 0.70, "zz": 0.701}, 0.01, ORDER)
    assert t["tied"] == ["zz", "xgb"] and t["recommended"] == "zz"
    assert race_tie({"lgbm": 0.69, "catboost": 0.70}, 0.0, ORDER)["recommended"] == "catboost"
