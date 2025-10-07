import pickle
import numpy as np
import os

class HeartDiseasePredictPipeline:
    def __init__(self, pipeline_path: str):
        if not os.path.isfile(pipeline_path):
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        # Load a single pipeline that contains both your preprocessors and model
        with open(pipeline_path, 'rb') as f:
            self.pipeline = pickle.load(f)

    def predict(self, input_data: list):
        """
        input_data: list or array-like of length N_features, in the same order
                    used during training.
        Returns a single prediction (e.g. 0 or 1).
        """
        arr = np.array(input_data, dtype=float).reshape(1, -1)
        return self.pipeline.predict(arr)[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Heart Disease Prediction")
    parser.add_argument(
        "--pipeline", "-p",
        default="heart_disease_pipeline.pkl",
        help="Path to your pickled sklearn Pipeline"
    )
    parser.add_argument(
        "features", nargs="+", type=float,
        help="The feature values in the exact order used at training"
    )
    args = parser.parse_args()

    pred_pipe = HeartDiseasePredictPipeline(args.pipeline)
    # simple length check
    if len(args.features) != pred_pipe.pipeline.named_steps['preprocessor'].transformers_[0][2]:
        print("⚠️  WARNING: feature count mismatch!")
    result = pred_pipe.predict(args.features)
    print("Prediction:", result)