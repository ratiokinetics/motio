# motio

```
pip install -r requirements.txt
```

### Train

```
python train.py
```

### Interface

```
python server.py
```

Boots run artifacts into RAM and serves http://localhost:8000/interface.html. Enter a sequence of cluster codes (6–10 bits each) by typing them space-separated. Shorter codes prefix-match deeper clusters; matching docs appear on the right.

### TODO

- Scale: bigger model and SAE dataset
- Account for dead branches in interface
- Attach Activation verbalizer
- Add matrix heatmap to show interesting and recurring patterns
- Graphic step to display a sentence
- Score to measure how close to cluster some sequence is
- Allow to user enter to enter a sentence 
- min 2 codes to be shown
- better colours
- Add interesting codes that I discovered 
- Autointerpretability. 