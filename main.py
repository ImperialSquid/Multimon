import logging
import os
from zlib import crc32

import cv2
import pandas as pd
import pokebase as pb
from requests.exceptions import HTTPError


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
            gen_species = pb.generation(i).pokemon_species
            log.debug(f"Found {len(gen_species)} species for generation {i}")
            for pokemon in gen_species:
                generations[pokemon.name] = i - 1
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
        # some variants give erroneous None names, so we need to check for that
        names = [var.pokemon.name for var in pb.pokemon_species(p.species.name).varieties
                 if var.pokemon.name is not None]
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
    os.makedirs("./sprites/raw", exist_ok=True)
    filepath = os.path.join("./sprites/raw", filename)

    if not os.path.exists(filepath):
        sprite = pb.sprite("pokemon", index, back=back)
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
        df[stat] = (df[stat] - df[stat].min()) / (df[stat].max() - df[stat].min())
    df.to_csv("data.csv", index=False)

    log.info("Postprocessed pokemon data")


def postprocess_pokemon_images(size=(256, 256)):
    log.info("Postprocessing pokemon images...")

    os.makedirs("./sprites/processed", exist_ok=True)

    # load images, standardise size, remove transparency and save to sprites/processed
    for filename in os.listdir("./sprites/raw"):
        if filename.endswith(".png"):
            img = cv2.imread(os.path.join("./sprites/raw", filename), cv2.IMREAD_UNCHANGED)

            # credit to https://stackoverflow.com/a/53737420
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [0, 0, 0, 0]
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img = cv2.resize(img, size)

            # cv2.imshow("image", img)
            # if cv2.waitKey(0) == 27:
            #     pass

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
        
    with open("partitions.csv", "w") as f:
        f.write("index,split\n")

    index = 1
    while True:
        try:
            # get pokemon data
            data = get_pokemon_data(index, gen_map, type_map)

            for ri, row in enumerate(data):
                for back in [False, True]:
                    name = row[0] + "_" + ["front", "back"][int(back)]
                    iname = str((index-1)*2+int(back)).zfill(4) + "_" + str(ri) + ".png"

                    try:
                        save_pokemon_images(iname, index, back)
                    except HTTPError:
                        log.warning(f"Could not get sprite for {name}")
                        continue

                    with open("data.csv", "a") as f:
                        f.write(iname + "," + ",".join(str(x) for x in row[1:]) + "\n")
                        log.debug(f"Saved data for {name}")

                    with open("partitions.csv", "a") as f:
                        # hash the name to get a partition value in [0:1]
                        part = float(crc32(name.encode("utf-8")) & 0xffffffff) / 2**32
                        # train/test/validation split is 0.7/0.15/0.15
                        split = str(2 if part > 0.85 else int(part > 0.7))  # [0.85:1] -> 2, [0.7:0.85] -> 1, else 0

                        f.write(iname + "," + split + "\n")
                        log.debug(f"Saved partition for {name}")

        except ValueError as e:
            break
        index += 1
    postprocess_pokemon_images()
    postprocess_pokemon_data()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # disable requests logging
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    main()

    log.info("Done!")
