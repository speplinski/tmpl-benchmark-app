#!/bin/bash

# Kolory i formatowanie
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # Resetowanie koloru

SEPARATOR="${YELLOW}===================================================${NC}"

# Użycie wget lub curl
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo -e "${RED}Error:${NC} Please install ${YELLOW}wget${NC} or ${YELLOW}curl${NC} to download the checkpoints."
    echo -e "${YELLOW}To install them on Ubuntu, run:${NC}"
    echo -e "  sudo apt update && sudo apt install -y wget curl"
    exit 1
fi

# Sprawdzanie obecności 7z
if ! command -v 7z &> /dev/null; then
    echo -e "${RED}Error:${NC} 7z is not installed. Please install it to proceed."
    echo -e "${YELLOW}To install it on Ubuntu, run:${NC}"
    echo -e "  sudo apt update && sudo apt install -y p7zip-full"
    exit 1
fi

# Definiowanie URL i katalogu
TMPL_BASE_URL="https://storage.googleapis.com/polish_landscape/dataset"
TMPL_dataset_SPN="simulation_masks_spn_white-dunes_0145_s580"
TMPL_dataset_SPN_url="${TMPL_BASE_URL}/${TMPL_dataset_SPN}.7z"
DATASET_DIR=$TMPL_dataset_SPN

# Tworzenie katalogu
echo -e "${SEPARATOR}"
echo -e "${YELLOW}Creating dataset directory...${NC}"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR" || { echo -e "${RED}Failed to change to $DATASET_DIR directory${NC}"; exit 1; }
echo -e "${GREEN}Directory created: ${DATASET_DIR}${NC}"

# Pobieranie danych
echo -e "${SEPARATOR}"
echo -e "${YELLOW}Downloading dataset SPN checkpoint...${NC}"
if ! $CMD "$TMPL_dataset_SPN_url"; then
    echo -e "${RED}Failed to download checkpoint from $TMPL_dataset_SPN_url${NC}"
    exit 1
fi
echo -e "${GREEN}Dataset downloaded successfully.${NC}"

# Rozpakowywanie danych
echo -e "${SEPARATOR}"
echo -e "${YELLOW}Extracting the dataset into '${DATASET_DIR}' directory...${NC}"
if ! 7z x "${TMPL_dataset_SPN}.7z"; then
    echo -e "${RED}Failed to extract dataset.${NC}"
    exit 1
fi
echo -e "${GREEN}Dataset extracted successfully.${NC}"

# Usuwanie pliku tymczasowego
rm "${TMPL_dataset_SPN}.7z"
echo -e "${GREEN}Temporary file removed: ${TMPL_dataset_SPN}.7z${NC}"

# Powrót do katalogu nadrzędnego
cd ..
echo -e "${SEPARATOR}"
echo -e "${GREEN}All operations completed successfully. The dataset is ready to use.${NC}"