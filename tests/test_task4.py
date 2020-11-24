from pprint import pprint

import pandas as pd

import task4


def test():
    pd.set_option('display.max_rows', 500)
    similarity_gesture_pairs = task4.get_modified_results_after_probabilistic_feedback(
        "570",
        # ["561", "562", "563", "563", "564", "565", "566", "567", "568", "560"],
        # [str(x) for x in range(559, 589+1)],
        # [str(x) for x in [250, 253, 258, 263, 265, 577, 582, 1]],
        # [str(x) for x in range(1, 10+1)] + ["1"]*1,
        # ["561"] * 10,
        # ["571", "572", "573", "574", "575", "576"] + ["570"]*100, ["260", "261", "262"] + ["261"]*5,
        ["571", "572", "573", "574", "575"] + ["570"] * 3 + ["574"] + ["571"], ["572", "573"],
        [],
        "../latent_features.csv"
    )
    print("Outputs:")
    pprint(similarity_gesture_pairs)


if __name__ == "__main__":
    test()
