import os

import pokebase as pb
from requests.exceptions import HTTPError
import cv2
import logging


def get_type_indexes():
    log.info("Getting type indexes")

    types = dict()
    i = 1
    while True:
        try:
            types[pb.type_(i).name] = i
            i += 1
        except HTTPError:
            break
    types[""] = 0

    log.info("Got type indexes")
    log.debug(types)
    return types


def get_generation_indexes():
    log.info("Getting generation indexes")

    generations = dict()
    i = 1
    while True:
        try:
            for pokemon in pb.generation(i).pokemon_species:
                generations[pokemon.name] = i
            i += 1
        except HTTPError:
            break

    log.info("Got generation indexes")
    log.debug(generations)
    return generations


def get_pokemon_data(index, gen_map=None, type_map=None):
    p = pb.pokemon(index)

    # check pokemon name in gen map
    if p.name not in gen_map:
        # if not, fetch the variants' names from the species
        names = [var.pokemon.name for var in pb.pokemon_species(p.species.name).varieties]

        # basculin's third variant is bugged so skip it
        if "basculin" == p.species.name:
            names = names[:2]
    else:
        names = [p.name]

    for name in names:
        if len(names) > 1:
            pokemon = pb.pokemon(name)
        else:
            pokemon = p  # skip refetching if only one variant

        # get types and pad with empty entry if only one exists
        types = ([pokemon.types[n].type.name for n in range(len(pokemon.types))] + [""])[:2]
        type1, type2 = sorted([type_map[t] for t in types])

        gen = gen_map[pokemon.species.name]
        hp, att, def_, spatt, spdef, speed = [pokemon.stats[n].base_stat for n in range(6)]
        height, weight = [pokemon.height, pokemon.weight]

        yield [name, type1, type2, gen, hp, att, def_, spatt, spdef, speed, height, weight]


def get_all_pokemon_data(type_map, gen_map):
    log.info("Getting pokemon data")

    # loop over pokemon species until an error is raised from the endpoint not being found
    index = 1
    data = {}
    while True:
        try:
            for row in get_pokemon_data(index, gen_map, type_map):
                data[row[0]] = row[1:]
                log.debug(row)

            # log every 10th pokemon
            if index % 10 == 0:
                log.info(f"Got data for {index} species")
        except HTTPError:  # if the pokemon is not found, break the loop
            break
        index += 1

    log.info("Got pokemon data")
    return data


def save_pokemon_images(index, name, back=False):
    sprite = pb.sprite("pokemon", name, back=back)

    filename = str(index*2+int(back)).zfill(4) + ".png"
    filepath = os.path.join("./sprites/raw", filename)
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.write(sprite.img_data)
            log.debug(f"Saved sprite for {name} ({back=})")
    else:
        log.debug(f"Skipping {name}")


def main():
    # get index maps for types and generations
    type_map = get_type_indexes()
    gen_map = get_generation_indexes()

    # write header to csv
    with open("data.csv", "w") as f:
        f.write("index,type1,type2,gen,hp,att,def,spatt,spdef,speed,height,weight\n")

    # get pokemon data
    data = get_all_pokemon_data(type_map, gen_map)

    for index, name in enumerate(data):
        try:
            save_pokemon_images(index, name)
            save_pokemon_images(index, name, back=True)

            with open("data.csv", "a") as f:
                f.write(str(index*2).zfill(4) + ".png" +
                        ",".join(str(x) for x in data[name]) + "\n")
                f.write(str(index*2+1).zfill(4) + ".png" +
                        ",".join(str(x) for x in data[name]) + "\n")

            # log every 10th pokemon
            if index % 10 == 0 and index != 0:
                log.debug(f"Saved data and sprites for {index}")

            index += 1
        except HTTPError:
            break
        break


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # disable requests logging
    log = logging.getLogger()

    main()

    log.info("Done!")
