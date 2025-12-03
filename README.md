# Multidimensional dataframes for analysis of sensor data

All contents of this project is according to the items and descriptions in the 'multidim_sensor_analysis.tex' and generated pdf file from the same file.

- [About](#about)
- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About
The purpose of this repo is to generate a workflow for creation and analysis of dataframes in pandas.

---

## Installation

### Clone the repo uisng following command:
```bash
git clone git@github.com:afshinzamani12/Multi-dimensional-dataframe-analysis.git
cd Multi-dimensional-dataframe-analysis
```

### Build docker image (The image was around 1.82 GB in my ubuntu system)
```bash
docker build -t multidim .
```

### Run the program inside docker
```bash
docker run --name multidim_sensor_analysis --rm -v "$(pwd):/Multi-dimensional-dataframe-analysis" -p 8000:8000 multidim:latest
```

Review the contents of "Dockerfile". "python3.8-slim-bullseye" was used to that supports installation of debian for livetex (Latex compiler) on the image.

---

## Features
- pandas
- numpy
- matplotlib

---

## Contributing
Contributions to push the current project further in machine learning or creating better automated piplines for data analysis are welcome!
(fork, commit, branch, push, pull request)

---

## License
This repo is distributed under MIT license.

---

## Contact

emial: afshin.zamani89@gmail.com
linkedin: /in/afshin-zamani
GitHub: @afshinzamani12
