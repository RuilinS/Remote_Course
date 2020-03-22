# Use an official Python runtime as a parent image
FROM python:3.6.3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app
ADD app.py /app
ADD train.py /app
ADD get_digit_info.py /app
ADD model.pth /app
ADD ip.txt /app
ADD templates/base.html /app/templates

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]

