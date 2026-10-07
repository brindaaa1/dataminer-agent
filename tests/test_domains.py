"""领域手册（knowhow/domains/）：不再只聚焦风控。通用手册 _default 打底，具体领域覆盖或追加；数据集在 data_dictionary 里声明 domain。"""
from data.knowhow import AVAIL_RANK, dataset_domain, domains, load_config, load_domain


def test_default_and_credit_risk_playbooks_merge():
    cfg = load_config()
    d = load_domain("_default", cfg)
    assert d["metric"]["primary"] == "auc" and "pr_auc" in d["metric"]["guards"] and d["feature_concepts"] == []
    c = load_domain("credit_risk", cfg)
    assert c["metric"]["guards"] == {"ks": 0.02}                                     # 领域覆盖通用的指标建议
    assert "last_" in c["leakage_name_patterns"] and "pymnt" in c["leakage_name_patterns"]   # 泄漏线索在通用之后追加
    assert any("40%" in r for r in c["sanity_rules"]) and any("SHAP" in r for r in c["sanity_rules"])
    assert {f["id"] for f in c["feature_concepts"]} >= {"multi_lending", "delinquency_history"}


def test_available_domains_exclude_the_template():
    names = {d["name"] for d in domains(load_config())}
    assert {"_default", "credit_risk"} <= names and "_template" not in names


def test_dataset_declares_its_domain():
    cfg = load_config()
    assert dataset_domain("lending_club", cfg) == "credit_risk" and dataset_domain("home_credit", cfg) == "credit_risk"
    assert dataset_domain("hotel_bookings", cfg) == "_default"


def test_generic_availability_values():
    """接入和酒店数据都用通用取值（不再用 at_booking / at_checkin 这类酒店说法）。"""
    assert AVAIL_RANK["at_prediction"] == AVAIL_RANK["history"] == 0 and AVAIL_RANK["before_outcome"] == 2
    assert "at_booking" not in AVAIL_RANK and "at_checkin" not in AVAIL_RANK


def test_domain_context_for_planning_is_compact():
    """规划时给 LLM 的领域信息：标题、正类含义、特征概念（只留 id / 概念 / 方向 / 理由）、泄漏线索，不带接入用的字段。"""
    from data.knowhow import domain_context
    c = domain_context("lending_club", load_config())
    assert c["title"].startswith("信贷风控") and c["positive_meaning"] == "违约或严重逾期"
    assert set(c["feature_concepts"][0]) == {"id", "concept", "direction", "rationale"}
    assert "pymnt" in c["leakage_name_patterns"] and "match_keywords" not in c
    h = domain_context("hotel_bookings", load_config())
    assert h["title"] == "通用二分类" and h["feature_concepts"] == []
