# Automatic Speech Recognition

## Installation

- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

```sh
wget -c https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
```

- command

```sh
DATE=20231103 && BATCH=32 && nohup python train.py >> logs/${DATE}_asr_batch_${BATCH}.out 2>&1 &
```
