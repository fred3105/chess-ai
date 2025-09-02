# chess-ai

Create an expert chess network

There is available code to load PGN games

## TODO

- [ ] Replace chess package with custom class
- [ ] Make chess AI available online
- [ ] Create openings dataset to train on openings
- [ ] Create endgames dataset to train on endings
- [ ] Look into selecting better positions for training, seems to have a lot of redundancy so far
- [ ] Move ordering to prioritize searching “interesting” move sequences
- [ ] Endgame Tablebase for precomputed winning endgame positions
- [ ] Opening Book for precomputed openings
- [ ] Only search a couple hundred thousand positions at 4 ply

## Get the database

[download games here](https://odysee.com/@Toadofsky:b/Lichess-Elite-Database:b) and rename it to data

Your file structure should look like this

- data /
  - lichess_elite_2013_09.pgn
  - ...

## Run tests with

uv run pytest

## To train the evaluation function

uv run train.py

## Results for the evaluation function

Current model is the best one so far, probably need to improve how it's trained, particularly how the board positions are chosen. Needs lots of endgame positions and not be too redundent on openings
Epoch 100: Train Loss=0.3053, Val Loss=0.3092, Val MAE=0.3769
