# NetGenRL
## Introduction
NetGenRL is a sequence adversarial generation network based on reinforcement learning, which is designed for labeled network traffic generation.

## Setup
NetGenRL is written under Python 3.12. Following Python package is needed before running.
* pytorch == 2.5.1
* numpy
* scipy
* gensim
* POT

## Run

### Data preparation

* Put training data (pcap format) into subfolder of pcap folder (`data-pcap` default)
* Make sure pcap files in training data is named as "[label].pcap"
    * An example of training data structure is following:

```
data-pcap
└── iscx
    ├── bittorrent.pcap
    ├── email.pcap
    ├── facebook.pcap
    ├── ftps.pcap
    ├── netflix.pcap
    ├── skype.pcap
    ├── spotify.pcap
    ├── vimeo.pcap
    ├── voipbuster.pcap
    └── youtube.pcap
```

### Configuration editing

* Edit configuration in `config.json`, the meaning of field is following:
    * path: folder path to get and save data (except for the folder where the training data is located, all other folders can be automatically generated)
        * dataset: name of training dataset, also the name of subfolder under training data folder
        * pcap\_folder: training data folder
        * json\_folder: json fommat training data folder (convert in pre-processing)
        * bins\_folder: folder of bins
        * model\_folder: folder of trained generation model
        * result\_folder: folder to save generation model
    * model\_paras: hyper-parameters of generation model
    * attributes: packet attributes of network traffic to generate. If users want to add or change attributes, the code of `pre_process/pcap_process.py` should be modified.

### Run code

* Run code with `python driver.py`
* The json fommat of generation data can be found in result\_folder
    * The generated traffic is a JSON list, with each item representing a bidirectional flow.
    * A flow includes src/dst ip and src/dst port, "series" of other packet level attributes.
    * Reverse data packets are represented by signed packet lengths.
    * "time" represents the time interval of the packet, and the first packet defaults to 0.