import json

def create_final_json():
    """
    Create a final.json file with the specific example from the prompt.
    """
    # Create the data structure
    data = [
        {
            "qa": {
                "question": "what was the percentage change in the net cash from operating activities from 2008 to 2009",
                "answer": "14.1%"
            }
        }
    ]
    
    # Save to file
    output_path = "app/final.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Created final.json at {output_path}")
    return output_path

if __name__ == "__main__":
    create_final_json() 