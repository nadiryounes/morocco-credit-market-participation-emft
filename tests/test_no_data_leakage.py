import yaml
def test_ml_feature_set():
    with open("config/analysis_config.yaml") as f:
        config = yaml.safe_load(f)
    features = config.get("features", [])
    expected = [
        "EMPLOYMENT",
        "FIRM_AGE",
        "DIRECT_EXPORT_SHARE_PCT",
        "FEMALE_OWNER",
        "EXTERNAL_AUDIT",
        "ELEC_OBSTACLE",
        "SECTOR",
        "REGION"
    ]
    assert features == expected, "Feature set does not match locked definition"
    assert "EXPORTER" not in features, "FAIL if deprecated EXPORTER replaces DIRECT_EXPORT_SHARE_PCT in the ML pipeline."
