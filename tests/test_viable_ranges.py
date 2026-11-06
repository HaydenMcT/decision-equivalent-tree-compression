from utils.dontcare_utils import ViableRanges

def test_viable_ranges():
    all_ftrs = ["a_<=_2", "a_<=_5", "a_<=_8", "a_<=_19", "b_<=_8", "b_<=_12", "c_<=_2", "c_<=_18"]
    known_ftrs = {
        "a_<=_5": False,
        "a_<=_12": True,
        "b_<=_10": True
    }
    vr = ViableRanges(
        known_ftrs=known_ftrs,
        feature_space=all_ftrs,
        original_name_getter=lambda x: (x.split("_<=_")[0], float(x.split("_<=_")[1]))
    )

    assert vr.deduce_threshold_state("a_<=_19") == True
    assert vr.deduce_threshold_state("a_<=_8") is None
    assert vr.deduce_threshold_state("a_<=_2") == False

    assert vr.deduce_threshold_state("b_<=_12") == True
    assert vr.deduce_threshold_state("b_<=_8") is None

    assert vr.deduce_threshold_state("c_<=_18") is None
    assert vr.deduce_threshold_state("c_<=_2") is None