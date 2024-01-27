# !/bin/bash

# Change the working directory to where ./ch is located
cd /home/wfp23/ADL/mase/machop

# Define the absolute path to the output directory
OUTPUT_DIR="/home/wfp23/ADL/mase/mase_output"

# Define your batch sizes here
BATCH_SIZES=(32 64 128 256)

# Loop over each batch size
for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    # Create a folder name based on the batch size
    FOLDER_NAME="$OUTPUT_DIR/batch_${BATCH_SIZE}"
    
    # Create the folder if it doesn't exist
    mkdir -p "$FOLDER_NAME"
    
    # Print the current configuration
    echo "Running model with Batch Size: $BATCH_SIZE"
    
    # Run
    ./ch train jsc-tiny jsc --max-epochs 10 --batch-size $BATCH_SIZE --learning-rate 0.001 --project-dir "$FOLDER_NAME"
done

# Define your epoch numbers here
EPOCHS=(10 20 50 100)

# Loop over each epoch number
for EPOCH in "${EPOCHS[@]}"
do
    # Create a folder name based on the epoch number
    FOLDER_NAME="$OUTPUT_DIR/epochs_${EPOCH}"
    
    # Create the folder if it doesn't exist
    mkdir -p "$FOLDER_NAME"
    
    # Print the current configuration
    echo "Running model with Epochs: $EPOCH"
    
    # Run 
    ./ch train jsc-tiny jsc --max-epochs $EPOCH --batch-size 256 --learning-rate 0.001 --project-dir "$FOLDER_NAME"
done

# Define your learning rates here
LEARNING_RATES=(0.000001 0.00001 0.0001 0.001 0.01 0.1)

# Loop over each learning rate
for LR in "${LEARNING_RATES[@]}"
do
    # Create a folder name based on the learning rate
    FOLDER_NAME="$OUTPUT_DIR/learning_rate_${LR}"
    
    # Create the folder if it doesn't exist
    mkdir -p "$FOLDER_NAME"
    
    # Print the current configuration
    echo "Running model with Learning Rate: $LR"
    
    # Run 
    ./ch train jsc-tiny jsc --learning-rate $LR --project-dir "$FOLDER_NAME"
done