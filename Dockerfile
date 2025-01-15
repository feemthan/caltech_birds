# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir matplotlib seaborn scikit-learn tqdm pillow

# Make port 80 available to the world outside this container
EXPOSE 80

# Run script.py when the container launches
CMD ["python", "test_resnet.py", "--config", "test_config.json"]