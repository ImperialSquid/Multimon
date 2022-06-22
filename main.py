import os

import pokebase as pb
from requests.exceptions import HTTPError
import cv2


def get_type_indexes():
    types = dict()
    i = 1
    while True:
        try:
            types[pb.type_(i).name] = i
            i += 1
        except HTTPError:
            break
    types[""] = 0
    return types


def get_generation_indexes():
    generations = dict()
    i = 1
    while True:
        try:
            for pokemon in pb.generation(i).pokemon_species:
                generations[pokemon.name] = i
            i += 1
        except HTTPError:
            break
    return generations

def main():
    pass


if __name__ == '__main__':
    print(len(get_generation_indexes()))

