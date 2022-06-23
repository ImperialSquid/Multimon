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
    # get index maps for types and generations
    type_map = get_type_indexes()
    gen_map = get_generation_indexes()

    # write the header
    with open("data.txt", "w") as f:
        f.write("name,type1,type2,gen,hp,att,def,spatt,spdef,speed,hght,wght\n")

    # loop over pokemon until an error is raised from the endpoint not being found
    index = 1
    data = {}
    while True:
        try:
            pokemon = pb.pokemon(index)

            # get types and pad with empty entry if only one exists
            types = ([pokemon.types[n].type.name for n in range(len(pokemon.types))] + [""])[:2]
            type1, type2 = sorted([type_map[t] for t in types])

            gen = gen_map[pokemon.name]
            hp, att, def_, spatt, spdef, speed = [pokemon.stats[n].base_stat for n in range(6)]
            height, weight = [pokemon.height, pokemon.weight]

            data[index] = [type1, type2, gen, hp, att, def_, spatt, spdef, speed, height, weight]
            # print(type1, type2, gen, hp, att, def_, spatt, spdef, speed, height, weight)
        except HTTPError:  # if the pokemon is not found, break the loop
            break
        index += 1


if __name__ == '__main__':
    main()

