import json
import argparse
import pprint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inspect predictions file format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the predictions JSON file to inspect")
    return parser.parse_args()


def inspect_predictions(input_file):
    """
    Inspect the structure of a predictions file to help diagnose format issues.
    
    Args:
        input_file: Path to the predictions JSON file to inspect
    """
    # Load the predictions
    with open(input_file, 'r') as f:
        predictions = json.load(f)
    
    # Print basic information
    print(f"Predictions type: {type(predictions)}")
    print(f"Predictions length: {len(predictions) if hasattr(predictions, '__len__') else 'N/A'}")
    
    # Inspect the structure
    if isinstance(predictions, dict):
        print("\nThis is a dictionary with keys:")
        print(f"Number of keys: {len(predictions.keys())}")
        print(f"Sample keys: {list(predictions.keys())[:5]}")
        
        # Check the first item
        first_key = next(iter(predictions))
        first_value = predictions[first_key]
        print(f"\nFirst key: {first_key}")
        print(f"First value type: {type(first_value)}")
        
        if isinstance(first_value, list):
            print(f"First value length: {len(first_value)}")
            if len(first_value) > 0:
                print(f"First value[0] type: {type(first_value[0])}")
                if isinstance(first_value[0], dict):
                    print(f"First value[0] keys: {list(first_value[0].keys())}")
                    print("\nSample of first value[0]:")
                    pprint.pprint(first_value[0])
                else:
                    print("\nSample of first value[0]:")
                    pprint.pprint(first_value[0])
        else:
            print("\nSample of first value:")
            pprint.pprint(first_value)
    
    elif isinstance(predictions, list):
        print("\nThis is a list with items:")
        print(f"Number of items: {len(predictions)}")
        
        if len(predictions) > 0:
            print(f"\nFirst item type: {type(predictions[0])}")
            if isinstance(predictions[0], dict):
                print(f"First item keys: {list(predictions[0].keys())}")
                print("\nSample of first item:")
                pprint.pprint(predictions[0])
            else:
                print("\nSample of first item:")
                pprint.pprint(predictions[0])
    
    else:
        print(f"\nUnexpected type: {type(predictions)}")
        print("\nSample of predictions:")
        pprint.pprint(predictions)


def main():
    """Main function to inspect predictions."""
    args = parse_args()
    inspect_predictions(args.input_file)


if __name__ == "__main__":
    main() 