# Local LLM Setup Using Ollama with Docker

This repository provides step-by-step instructions for setting up and interacting with local
large language models (LLMs) using Ollama with Docker. Two models are showcased: 
`tinyllama` and `gemma:2B`. Additionally, a Python script is provided to interact with the local LLM.

## Prerequisites

- Docker installed on your device

## Setup Instructions

## Step 1: Pull the Ollama Docker Image

Pull the official Ollama Docker image from Docker Hub:

```sh
docker pull ollama/ollama
```

## Step 2: Start the Ollama Container

Run the following command to start the Ollama container:

```sh
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama-container ollama/ollama
```

### Command Breakdown:

- `docker run`: Spins up the container.
- `-d`: Runs the container in detached mode.
- `--gpus=all`: Utilizes all available GPUs, if you don't have GPU you can remove this flag.
- `-v ollama:/root/.ollama`: Mounts the container's root directory to your device's directory.
- `-p 11434:11434`: Exposes the container's port 11434 to your device's

## Step 3: Verify Ollama Server is Running

Check if the Ollama server is up by visiting http://localhost:11434. You should see a message saying "Ollama is running". If not, start the server with:

```sh
docker exec ollama-container ollama serve
```

## Step 4: Run the Model

### Tinyllama Model

Run the tinyllama model with:
```sh
docker exec ollama-container ollama run tinyllama
```

This command downloads the `tinyllama` model and starts it on port _11434_.

### Gemma:2B Model

To switch to the gemma:2B model, run:
```sh
docker exec ollama-container ollama run gemma:2b
```
## Step 4: Interact with the Model

Use the provided REST API to interact with the running model. Here is an example CURL command to send a request to the locally running Ollama server:

```sh
curl http://localhost:11434/api/chat -d '{
  "model": "tinyllama",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```
You can find the curated documentation of all the REST APIs [here](https://github.com/ollama/ollama/blob/main/docs/api.md).

## Python Script
A Python script is available in this repository to interact with the Ollama server. You can find the script [here](test-ollama.py),

## Model Response Analysis

Here are the responses from the `tinyllama` and `gemma:2B` models for the question "Why is the sky blue?"

### Tinyllama Response:

The reason why the sky is blue is that it's composed of different colors, including red, orange, yellow, green, blue, indigo, and violet, which are all visible in the visible spectrum of light. The exact color balance depends on the amount of sunlight that hits the Earth at any given time. At low altitudes, such as those near the equator, there is less sunlight and the sky tends to appear more gray or brown due to absorption by green plants. As you go higher in the sky, the angle of incidence of sunlight increases, resulting in a brighter sky with more blue light. The balance between the different colors varies depending on various factors like weather patterns, atmospheric conditions, and distance from the equator.

### Gemma:2B Response:
The sky appears blue due to Rayleigh scattering. <br>

**Rayleigh Scattering** is a scattering process that occurs when sunlight interacts with particles smaller than the wavelength 
of light. These particles can be molecules in the atmosphere, dust particles, or other celestial bodies.

* **Blue light has a shorter wavelength than other colors**. This means that when blue light enters the atmosphere, it has more 
time to interact with molecules and particles.
* **Blue light is scattered more strongly than other colors**. This is because blue light has a higher frequency and more 
photons per wavelength.
* **Scattered blue light is scattered in all directions**. This is why the sky appears blue to us.

**Other factors that contribute to the blue color of the sky:**

* **Dust and pollution:** Dust and pollution particles can scatter all colors of light equally, making the sky appear gray or 
black.
* **Sun's position:** The sky appears blue during the day because the sun's light has to travel through the Earth's atmosphere. 
The atmosphere is denser at higher altitudes, which causes blue light to be scattered more strongly.
* **Angle of view:** The sky appears blue from a variety of angles, depending on where you are in the world. This is because 
the angle of view changes the amount of blue light that is scattered.

Overall, Rayleigh scattering is the main reason why the sky appears blue. This beautiful color is a reminder of the wonders of 
nature and the power of light to shape our perception of the world.


## Analysis of the responses

- **Tinyllama**: Provides a broad explanation involving sunlight, colors, and atmospheric conditions but does not mention Rayleigh scattering.
- **Gemma:2B**: Gives a scientifically accurate explanation emphasizing Rayleigh scattering and additional factors like dust, pollution, and the sun's position.

Gemma's response is more accurate and detailed, specifically mentioning Rayleigh scattering, the primary reason for the sky's blue color. The reason behind `Gemma` outperformed `tinyllama` is because Gemma:2b model is trained with 2 billion parameters whereas `tinyllama` is trained on 1.1 Billion parameters.

Additional Resources

You can find a video of the entire setup and interaction process [here](local%20LLM.mp4).