#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}           CV EXTRACTOR LAUNCHER               ${NC}"
echo -e "${BLUE}================================================${NC}"

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}Error: Docker is not running.${NC}"
        echo -e "${YELLOW}Please start Docker Desktop and try again.${NC}"
        exit 1
    fi
}

# Function to start with Docker
start_with_docker() {
    echo -e "${GREEN}Starting CV Extractor with Docker...${NC}"
    check_docker
    
    echo -e "${YELLOW}Building and starting containers...${NC}"
    docker-compose up --build
}

# Function to start with Python directly
start_with_python() {
    echo -e "${GREEN}Starting CV Extractor with Python...${NC}"
    echo -e "${YELLOW}Checking for Python...${NC}"
    
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Error: Python not found.${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Checking for required packages...${NC}"
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements.txt
    fi
    
    echo -e "${GREEN}Starting the application...${NC}"
    $PYTHON_CMD run_web.py
}

# Function to stop Docker containers
stop_docker() {
    echo -e "${YELLOW}Stopping Docker containers...${NC}"
    docker-compose down
    echo -e "${GREEN}Docker containers stopped.${NC}"
}

# Main menu
show_menu() {
    echo -e "\n${YELLOW}Choose an option:${NC}"
    echo -e "${BLUE}1.${NC} Start with Docker (recommended)"
    echo -e "${BLUE}2.${NC} Start with Python directly"
    echo -e "${BLUE}3.${NC} Stop running Docker containers"
    echo -e "${BLUE}4.${NC} Exit"
    
    read -p "Enter your choice [1-4]: " choice
    
    case $choice in
        1) start_with_docker ;;
        2) start_with_python ;;
        3) stop_docker ;;
        4) echo -e "${GREEN}Exiting. Goodbye!${NC}"; exit 0 ;;
        *) echo -e "${RED}Invalid choice. Please try again.${NC}"; show_menu ;;
    esac
}

# Show the menu
show_menu 