#!/bin/bash

# Path to your target shell script
TARGET_SCRIPT="train_resnet34.sh"

# Check if the target script exists
if [ ! -f "$TARGET_SCRIPT" ]; then
    echo "Error: $TARGET_SCRIPT does not exist."
    exit 1
fi

# Check if the target script is executable
if [ ! -x "$TARGET_SCRIPT" ]; then
    echo "Error: $TARGET_SCRIPT is not executable."
    exit 1
fi

# Schedule the script to run after 6 hours
echo "$TARGET_SCRIPT" | at now + 6 hours

# Confirm the scheduling
echo "Script $TARGET_SCRIPT has been scheduled to run in 6 hours."

# Display scheduled jobs
echo "Currently scheduled jobs:"
atq