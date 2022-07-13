# Multimon ![](lucario.png)

## Description
Multimon is a dataset specifically made for Multitask Learning (MTL) based on the popular Pokémon videogame series. It contains sprites of every pokemon in the series and has labels for multiple tasks.

## Dataset Structure
Note that due to copyright restrictions, the dataset is not available for direct download. Other than the main script, the files described below are not present but are constructed by running the script main.py.

### [The Main Script](main.py)
This script contains everything you need to construct the dataset. It will download the sprites and labels from the [PokeAPI](https://pokeapi.co/), perform postprocessing (standardising the sprites and labels) and construct partitions for train/test and train/test/validation splits.

### [The Sprites](sprites/)
Sprites are stored in two directories, `sprites/processed` for the actual sprites to use and `sprites/raw` for raw images (these are provided in case you want to do custom postprocessing, though it's unlikely you'll ever need to)

### [The Data](data.csv)
This is the main file containing the labels. It contains the following columns. 

| Name                                          | Description                                  | Type    | 
|-----------------------------------------------|----------------------------------------------|---------|
| `index`                                       | The filename of the sprite                   | string  |
| `type1`, `type2`[^1]                          | The first and second type of the pokemon     | int     |
| `gen`                                         | The generation the pokemon was introduced in | integer |
| `hp_raw`, `hp_norm`, `hp_std`[^2]             | The HP stat of the pokemon                   | float   |
| `atk_raw`, `atc_norm`, `atc_std`[^2]          | The Attack stat of the pokemon               | float   |
| `def_raw`, `def_norm`, `def_std`[^2]          | The Defense stat of the pokemon              | float   |
| `spatk_raw`, `spatk_norm`, `spatk_std`[^2]    | The Special Attack stat of the pokemon       | float   |
| `spdef_raw`, `spdef_norm`, `spdef_std`[^2]    | The Special Defense stat of the pokemon      | float   |
| `spd_raw`, `spd_norm`, `spd_std`[^2]          | The Speed stat of the pokemon                | float   |
| `height_raw`, `height_norm`, `height_std`[^2] | The height of the pokemon                    | float   |
| `weight_raw`, `weight_norm`, `weight_std`[^2] | The weight of the pokemon                    | float   |

[^1]: Pokémon have one or two of 17 types, for consistency we create an 18th "null" type. `type1` and `type2` should be interpreted as a **two-hot vector together**.

[^2]: hp and all the columns below are provided as the raw value, as well as the normalised and standardised values (suffixed as "_raw", "_norm" and "_std" respectively)

### [Partitions](partitions.csv)
Partitions are an important part of a dataset to ensure consistency between runs. Since this repo only contains the script for constructing the dataset, the partitions are also not available for direct download. To ensure consistency  the partitions are created by hashing the name of the pokemon and using that value to assign partitions. This method is consistent between python versions, download instances and also in the event of more Pokémon being released.

The partitions have an 80/20 split for training/test setups and a 70/15/15 split for train/test/validation setups.