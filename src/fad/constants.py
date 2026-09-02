"""Project-wide biological and leakage constraints."""

MUTATION_POSITIONS = (30, 31, 32, 33, 89, 90, 91, 92, 127, 128, 129, 130, 131)

FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "label",
        "total_energy_change",
        "r_psp_MMGBSA_dG_Bind",
        "dock",
        "gbsa",
        "Gap1",
        "Gap2",
        "Gap3",
    }
)

EXPENSIVE_NON_MAINLINE_COLUMNS = frozenset({"r_i_docking_score"})

LABEL_GAP2_WEIGHT = 0.961
LABEL_GBSA_WEIGHT = 0.039
MFE_ENGINE = "NUPACK"

