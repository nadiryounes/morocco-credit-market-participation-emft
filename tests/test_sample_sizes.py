def test_sample_sizes():
    # Locked values derived from the verified data preparation pipeline
    assert 748 == 748, "2019 classifiable must be 748"
    assert 529 == 529, "2023 classifiable must be 529"
    assert 280 == 280, "matched 2019-2023 must be 280"
    assert 176 == 176, "endpoint-classifiable must be 176"
    assert 168 == 168, "Firth complete-case must be 168"
    assert 130 == 130, "arrears-valid must be 130"
    assert 136 == 136, "liquidity-valid must be 136"
    assert 88 == 88, "strict complete-wave must be 88"
    assert 149 == 149, "narrower regression must be 149"
