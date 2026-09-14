#!/usr/bin/env bash

for i in {12..14}; do
    python3 train_kan.py --seed ${i} --force
done
for i in {12..14}; do
    python3 train_qkan.py --seed ${i} --force
done
