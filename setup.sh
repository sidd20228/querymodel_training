#!/bin/bash

# Setup script for LLMxCPG benchmarking
# Configures HuggingFace username and prepares environment

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔧 LLMxCPG Setup Configuration${NC}"
echo -e "${BLUE}========================================${NC}"

# Get HuggingFace username
echo -e "\n${YELLOW}Enter your HuggingFace username:${NC}"
read -p "Username: " HF_USERNAME

if [ -z "$HF_USERNAME" ]; then
    echo -e "${YELLOW}⚠️  No username provided. You can update it later.${NC}"
    HF_USERNAME="your-username"
fi

# Get HuggingFace token
echo -e "\n${YELLOW}Enter your HuggingFace token (or press Enter to skip):${NC}"
echo "Get your token from: https://huggingface.co/settings/tokens"
read -s -p "Token: " HF_TOKEN
echo

# Update train_all_models.sh with username
TRAIN_SCRIPT="./training/train_all_models.sh"

if [ -f "$TRAIN_SCRIPT" ]; then
    echo -e "\n${GREEN}📝 Updating training script with your username...${NC}"
    
    # Create backup
    cp "$TRAIN_SCRIPT" "${TRAIN_SCRIPT}.backup"
    
    # Replace username in all HuggingFace repo IDs
    sed -i.tmp "s/your-username/$HF_USERNAME/g" "$TRAIN_SCRIPT"
    rm "${TRAIN_SCRIPT}.tmp" 2>/dev/null || true
    
    echo -e "${GREEN}✅ Training script updated!${NC}"
else
    echo -e "${YELLOW}⚠️  Training script not found at: $TRAIN_SCRIPT${NC}"
fi

# Save token to environment
if [ -n "$HF_TOKEN" ]; then
    echo -e "\n${GREEN}💾 Saving HuggingFace token...${NC}"
    
    # Add to .env file
    echo "export HUGGING_FACE_TOKEN=$HF_TOKEN" > .env
    
    # Add to shell config
    SHELL_CONFIG=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    fi
    
    if [ -n "$SHELL_CONFIG" ]; then
        if ! grep -q "HUGGING_FACE_TOKEN" "$SHELL_CONFIG"; then
            echo "" >> "$SHELL_CONFIG"
            echo "# LLMxCPG HuggingFace Token" >> "$SHELL_CONFIG"
            echo "export HUGGING_FACE_TOKEN=$HF_TOKEN" >> "$SHELL_CONFIG"
            echo -e "${GREEN}✅ Token added to $SHELL_CONFIG${NC}"
            echo -e "${YELLOW}   Run: source $SHELL_CONFIG${NC}"
        fi
    fi
    
    # Set for current session
    export HUGGING_FACE_TOKEN=$HF_TOKEN
    echo -e "${GREEN}✅ Token set for current session${NC}"
fi

# Create necessary directories
echo -e "\n${GREEN}📁 Creating directories...${NC}"
mkdir -p models
mkdir -p benchmark/benchmark_results
mkdir -p inference/results
echo -e "${GREEN}✅ Directories created${NC}"

# Check Python dependencies
echo -e "\n${GREEN}🐍 Checking Python dependencies...${NC}"
python3 -c "import torch; print('✅ PyTorch installed')" 2>/dev/null || echo "❌ PyTorch not installed"
python3 -c "import transformers; print('✅ Transformers installed')" 2>/dev/null || echo "❌ Transformers not installed"
python3 -c "import unsloth; print('✅ Unsloth installed')" 2>/dev/null || echo "❌ Unsloth not installed"
python3 -c "import pandas; print('✅ Pandas installed')" 2>/dev/null || echo "❌ Pandas not installed"

# Check Joern
echo -e "\n${GREEN}🔍 Checking Joern installation...${NC}"
if command -v joern &> /dev/null; then
    JOERN_PATH=$(which joern)
    echo -e "${GREEN}✅ Joern found at: $JOERN_PATH${NC}"
else
    echo -e "${YELLOW}⚠️  Joern not found in PATH${NC}"
    echo -e "   Install from: https://github.com/joernio/joern/releases"
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}📋 Configuration Summary:${NC}"
echo -e "   HuggingFace Username: $HF_USERNAME"
if [ -n "$HF_TOKEN" ]; then
    echo -e "   HuggingFace Token: ${GREEN}Set ✓${NC}"
else
    echo -e "   HuggingFace Token: ${YELLOW}Not set${NC}"
fi

echo -e "\n${GREEN}🚀 Next Steps:${NC}"
echo -e "1. Review updated training script:"
echo -e "   ${BLUE}cat training/train_all_models.sh${NC}"
echo -e ""
echo -e "2. Install missing dependencies (if any):"
echo -e "   ${BLUE}pip install -r training/requirements.txt${NC}"
echo -e "   ${BLUE}pip install pandas${NC}"
echo -e ""
echo -e "3. Start training:"
echo -e "   ${BLUE}cd training && ./train_all_models.sh${NC}"
echo -e ""
echo -e "4. Run benchmark after training:"
echo -e "   ${BLUE}cd benchmark && python run_benchmark.py --models ../models/qwen* --model_names 7B 14B 32B${NC}"

if [ -n "$HF_TOKEN" ]; then
    echo -e "\n${GREEN}💡 Tip: Your token is saved. Models will automatically push to HuggingFace!${NC}"
else
    echo -e "\n${YELLOW}💡 Tip: Set token later with: export HUGGING_FACE_TOKEN=your_token${NC}"
fi

echo -e "\n${GREEN}📚 Documentation:${NC}"
echo -e "   Quick Start: ${BLUE}cat QUICKSTART.md${NC}"
echo -e "   Benchmark Guide: ${BLUE}cat benchmark/README.md${NC}"
echo -e "   Training Guide: ${BLUE}cat training/README.md${NC}"

echo -e "\n${GREEN}Happy coding! 🚀${NC}\n"
