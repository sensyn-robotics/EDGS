#!/bin/bash
# Increase swap from 2GB to 16GB to prevent system freeze from RAM exhaustion.
# Run with sudo: sudo bash script/increase_swap.sh
set -euo pipefail

TARGET_SIZE="16G"
SWAPFILE="/swapfile"

echo "Current swap:"
swapon --show
free -h | grep Swap
echo ""

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Must run as root (sudo)"
    exit 1
fi

# Turn off current swap
echo "Disabling current swap..."
swapoff "$SWAPFILE" 2>/dev/null || true

# Resize
echo "Resizing $SWAPFILE to $TARGET_SIZE..."
fallocate -l "$TARGET_SIZE" "$SWAPFILE"
chmod 600 "$SWAPFILE"

# Format and enable
echo "Formatting and enabling swap..."
mkswap "$SWAPFILE"
swapon "$SWAPFILE"

echo ""
echo "New swap:"
swapon --show
free -h | grep Swap
echo ""
echo "Done. Swap increased to $TARGET_SIZE."
