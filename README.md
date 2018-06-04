# Ada **Lovelace** 

Neural networks for sentiment analysis and phrase classification.

## Installation
#### Docker
You can build the docker image
```bash
cd /path/to/haterz
docker build -t haterz .
```

..and then starting a container in interactive mode

```bash
docker container run -it haterz
```

#### Downloading dependencies
If you don't download the docker image (recommended) then you can ensure you have the corrected dependencies by running the `requirements.txt`. If going this route, I would recommend a virtualenv or conda env before downloading. 

```bash
cd /path/to/haterz
pip3 -r requirements.txt
```

