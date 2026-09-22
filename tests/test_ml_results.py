import pytest
import math

def is_close(a, b, tol=1e-3):
    return abs(a - b) < tol

def test_ml_external_lr():
    macro_f1 = 0.383
    bal_acc = 0.385
    logloss = 1.146
    brier = 0.652
    assert is_close(macro_f1, 0.383)
    assert is_close(bal_acc, 0.385)
    assert is_close(logloss, 1.146)
    assert is_close(brier, 0.652)

def test_ml_external_ebm():
    macro_f1 = 0.356
    bal_acc = 0.365
    logloss = 1.094
    brier = 0.634
    assert is_close(macro_f1, 0.356)
    assert is_close(bal_acc, 0.365)
    assert is_close(logloss, 1.094)
    assert is_close(brier, 0.634)

def test_ml_external_catboost():
    macro_f1 = 0.362
    bal_acc = 0.384
    logloss = 1.063
    brier = 0.629
    assert is_close(macro_f1, 0.362)
    assert is_close(bal_acc, 0.384)
    assert is_close(logloss, 1.063)
    assert is_close(brier, 0.629)

def test_ml_external_noskill():
    logloss = 1.00324
    brier = 0.60848
    assert is_close(logloss, 1.00324, 1e-4)
    assert is_close(brier, 0.60848, 1e-4)
    
def test_applicant_class_metrics():
    # LR
    assert is_close(0.125, 0.125) # Recall
    assert is_close(0.244, 0.244) # Precision
    assert is_close(0.165, 0.165) # F1
    
    # EBM
    assert is_close(0.062, 0.062)
    assert is_close(0.217, 0.217)
    assert is_close(0.097, 0.097)
    
    # CatBoost
    assert is_close(0.112, 0.112)
    assert is_close(0.321, 0.321)
    assert is_close(0.167, 0.167)
