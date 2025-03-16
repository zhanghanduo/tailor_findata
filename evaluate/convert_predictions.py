import json
import os
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert predictions to the format expected by the evaluation function")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input predictions JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the converted predictions JSON file")
    return parser.parse_args()


def convert_predictions(input_file, output_file):
    """
    Convert predictions from the flat list format to the nested dictionary format
    expected by the evaluation function.
    
    Args:
        input_file: Path to the input predictions JSON file
        output_file: Path to save the converted predictions JSON file
    """
    # Load the input predictions
    with open(input_file, 'r') as f:
        predictions = json.load(f)
    
    # Convert to the expected format
    evaluation_format = {}
    for pred in predictions:
        example_id = pred["id"]
        evaluation_format[example_id] = [{
            "id": example_id,
            "pred_prog": pred["predicted"]
        }]
    
    # Save the converted predictions
    with open(output_file, 'w') as f:
        json.dump(evaluation_format, f, indent=2)
    
    print(f"Converted predictions saved to {output_file}")
    print(f"Converted {len(predictions)} predictions to the expected format")


def main():
    """Main function to convert predictions."""
    args = parse_args()
    convert_predictions(args.input_file, args.output_file)


if __name__ == "__main__":
    main() 