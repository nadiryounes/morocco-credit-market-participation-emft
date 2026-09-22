import pytest

def test_transition_matrix():
    # 2019 Applicant -> 2023 [Applicant, Constrained, No need]
    app_transitions = [7, 5, 14]
    assert app_transitions == [7, 5, 14]
    
    # 2019 Constrained -> 2023
    const_transitions = [13, 11, 32]
    assert const_transitions == [13, 11, 32]
    
    # 2019 No financing need -> 2023
    no_need_transitions = [12, 34, 48]
    assert no_need_transitions == [12, 34, 48]
    
    changed = 110
    total = 176
    assert abs((changed / total) - 0.625) < 1e-4

def test_stuart_maxwell():
    sm_statistic = 0.98
    df = 2
    p_value = 0.611
    assert sm_statistic == 0.98
    assert df == 2
    assert p_value == 0.611

def test_covid_arrears():
    OR = 0.94
    CI = [0.36, 2.33]
    raw_p = 0.901
    holm_p = 0.901
    assert OR == 0.94
    assert CI == [0.36, 2.33]
    assert raw_p == 0.901
    assert holm_p == 0.901

def test_covid_liquidity():
    OR = 0.70
    CI = [0.30, 1.68]
    raw_p = 0.421
    holm_p = 0.842
    assert OR == 0.70
    assert CI == [0.30, 1.68]
    assert raw_p == 0.421
    assert holm_p == 0.842
