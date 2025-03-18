#!/bin/bash
set -e

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Financial RAG Evaluation System${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
python -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data results examples

# Move sample data files if they exist
if [ -f "data/sample_train.json" ] && [ -f "data/sample_knowledge.json" ]; then
    echo -e "${GREEN}Sample data files already exist.${NC}"
else
    echo -e "${YELLOW}Copying sample data files to data directory...${NC}"
    # Check if the sample files are in the current directory
    if [ -f "sample_train.json" ]; then
        cp sample_train.json data/
    else
        echo -e "${RED}sample_train.json not found. Please place it in the data directory.${NC}"
    fi
    
    if [ -f "sample_knowledge.json" ]; then
        cp sample_knowledge.json data/
    else
        echo -e "${RED}sample_knowledge.json not found. Please place it in the data directory.${NC}"
    fi
fi

# Set execute permissions for Python scripts
echo -e "${YELLOW}Setting execute permissions for scripts...${NC}"
chmod +x run.py
chmod +x evaluation/run.py
chmod +x examples/openai_integration.py

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "To run the evaluation, activate the virtual environment and execute:"
echo -e "  source venv/bin/activate"
echo -e "  python run.py"
echo ""
echo -e "To run with custom paths:"
echo -e "  python run.py --train_path <path_to_train> --knowledge_path <path_to_knowledge>"
echo ""
echo -e "For detailed results, add the --verbose flag:"
echo -e "  python run.py --verbose"
echo ""
echo -e "To integrate with OpenAI's API, see the example:"
echo -e "  export OPENAI_API_KEY=your_api_key_here"
echo -e "  python examples/openai_integration.py" 