# chess-ai

Create an expert chess network

There is available code to load PGN games

## TODO

- [ ] Replace chess package with custom class
- [ ] Implement min max algorithm to create the chess AI and play with it
- [ ] Look into selecting better positions for training, seems to have a lot of redundancy so far

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

Current model is the best one so far, probabaly needs to train for longer
