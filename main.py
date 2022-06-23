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


def get_pokemon_data(type_map, gen_map):
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
    return data


def save_pokemon_images(index, back=False):
    sprite = pb.sprite("pokemon", index + 1, back=back)

    filename = str(index*2+int(back)).zfill(4) + ".png"
    filepath = os.path.join("./sprites/raw", filename)
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.write(sprite.img_data)


def main():
    # get index maps for types and generations
    type_map = get_type_indexes()
    gen_map = get_generation_indexes()

    # write header to csv
    with open("data.csv", "w") as f:
        f.write("index,type1,type2,gen,hp,att,def,spatt,spdef,speed,height,weight\n")

    # get pokemon data
    data = get_pokemon_data(type_map, gen_map)

    index = 0
    while True:
        try:
            save_pokemon_images(index)
            save_pokemon_images(index, back=True)

            with open("data.csv", "a") as f:
                f.write(str(index*2).zfill(4) + ".png" +
                        ",".join(str(x) for x in data[index+1]) + "\n")
                f.write(str(index*2+1).zfill(4) + ".png" +
                        ",".join(str(x) for x in data[index+1]) + "\n")

            if index % 100 == 0:
                print("Downloaded {} images".format(index))

            index += 1
        except HTTPError:
            break
        break


if __name__ == '__main__':
    main()

