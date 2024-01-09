#!/bin/bash

# Check if three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <model_size (S/XS)> <train_every (power of 2)> <gpu_number (0/1/2/3)>"
    exit 1
fi

# Read the arguments
model_size=$1
train_every=$2
gpu_number=$3

# Validate the model_size argument
if [ "$model_size" != "S" ] && [ "$model_size" != "XS" ]; then
    echo "Invalid model size: $model_size. Use 'S' or 'XS'."
    exit 1
fi

# Validate the train_every argument
if ! [[ "$train_every" =~ ^(1|2|4|8|16|32|64)$ ]]; then
    echo "Invalid train_every value: $train_every. Use a power of 2 between 1 and 64."
    exit 1
fi

# Validate the gpu_number argument
if ! [[ "$gpu_number" =~ ^(0|1|2|3)$ ]]; then
    echo "Invalid GPU number: $gpu_number. Use a number between 0 and 3."
    exit 1
fi

# Construct the experiment name based on the model size
if [ "$model_size" = "S" ]; then
    exp_name="dreamer_v3_100k_ms_pacman"
else
    exp_name="dreamer_v3_XS_100k_ms_pacman"
fi

# Construct the command
command_to_run="python sheeprl.py exp=$exp_name fabric=gpu fabric.devices=[$gpu_number] algo.train_every=$train_every"

# Get the start time
start_time=$(date +%s)
start_time_readable=$(date)

echo "Start Time: $start_time_readable"

# Run the command
eval $command_to_run

# Get the end time
end_time=$(date +%s)
end_time_readable=$(date)

echo "End Time: $end_time_readable"

# Calculate elapsed time
elapsed=$((end_time - start_time))

echo "Elapsed Time: $elapsed seconds"

# Optional: Save to a log file
{
    echo "Command: $command_to_run"
    echo "Start Time: $start_time_readable"
    echo "End Time: $end_time_readable"
    echo "Elapsed Time: $elapsed seconds \\n"
    echo ""
} >> program_run_log.txt


