from era_py import load_farr_rda

def test_load_farr_rda_smoke():
    # Pick a tiny dataset you know exists in farr/data
    obj = load_farr_rda("camp_attendance")
    assert obj is not None