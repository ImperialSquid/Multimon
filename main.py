import os

import pokebase as pb
from requests.exceptions import HTTPError
import cv2
import logging
import pandas as pd


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

        # some variants are bugged and give Nones at the end of the list, so we need to remove them
        if p.species.name in ["basculin", "toxtricity-amped"]:
            names = names[:-1]
    else:
        names = [p.name]

    results = []

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

        results.append([name, type1, type2, gen, hp, att, def_, spatt, spdef, speed, height, weight])
        log.debug(f"Got data for {name} {[name, type1, type2, gen, hp, att, def_, spatt, spdef, speed, height, weight]}")

    return results


def save_pokemon_images(filename, index, back=False):
    sprite = pb.sprite("pokemon", index, back=back)

    filepath = os.path.join("./sprites/raw", filename)
    if not os.path.exists(filepath):
        with open(filepath, "wb") as f:
            f.write(sprite.img_data)
            log.debug(f"Saved sprite for filename {filename}")
    else:
        log.debug(f"Skipping {filename}")


def postprocess_pokemon_data():
    log.info("Postprocessing pokemon data...")

    # load csv data, figure our standardised and normalised values and append to every row
    df = pd.read_csv("data.csv")
    for stat in ["hp", "att", "def", "spatt", "spdef", "speed", "height", "weight"]:
        df[stat+"std"] = df[stat].apply(lambda x: (x - df[stat].mean()) / df[stat].std())
        df[stat+"norm"] = df[stat].apply(lambda x: (x - df[stat].min()) / (df[stat].max() - df[stat].min()))
    df.to_csv("data.csv", index=False)

    log.info("Postprocessed pokemon data")


def postprocess_pokemon_images(size=(128, 128)):
    log.info("Postprocessing pokemon images...")

    # load images, standardise size, remove transparency and save to sprites/processed
    for filename in os.listdir("./sprites/raw"):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join("./sprites/raw", filename))
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            cv2.imwrite(os.path.join("./sprites/processed", filename), img)
            log.debug(f"Saved processed image for {filename}")

    log.info("Postprocessed pokemon images")


def main():
    # get index maps for types and generations
    type_map = get_type_indexes()
    gen_map = get_generation_indexes()

    # write header to csv
    with open("data.csv", "w") as f:
        f.write("index,type1,type2,gen,hp,att,def,spatt,spdef,speed,height,weight\n")

    index = 1
    while True:
        try:
            # get pokemon data
            data = get_pokemon_data(index, gen_map, type_map)

            for row in data:
                for back in [False, True]:
                    with open("data.csv", "a") as f:
                        name = row[0] + "_" + ["front", "back"][int(back)]
                        iname = str((index-1)*2+int(back)).zfill(4) + ".png"

                        f.write(iname + "," + ",".join(str(x) for x in row[1:]) + "\n")
                        log.debug(f"Saved data for {name}")

                        save_pokemon_images(iname, index, back)

        except HTTPError as e:
            log.error(e)
            break
        index += 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # disable requests logging
    log = logging.getLogger()

    main()

    log.info("Done!")
