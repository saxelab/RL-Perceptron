# RL-Perceptron
Generalisation dynamics of policy learning in high-dimensions

## Simulations
The code for numerically solving the ODEs and their corresponding simulations can be run in ODEs_and_Simulations.ipynb

## Procgen

Supported Platforms:
- Windows 10
- macOS 10.14 (Mojave), 10.15 (Catalina)
- Linux (manylinux2010)

Supported Pythons:
- 3.7 64-bit
- 3.8 64-bit
- 3.9 64-bit
- 3.10 64-bit

### Install procgen from cource

```
cd procgen
conda env update --name procgen --file environment.yml
conda activate procgen
pip install -e .
# this should say "building procgen...done"
```

### Bossfight

In the procgen environment, the bossfight training and evalutation can be run from the train_eval_bossfight2 notebook.

## Pong

In the procgen environment (instructions to create above), install the `ale-py` package distributed via PyPI:
```
pip install ale-py
```
Follow the instructions for installing the Pong ROMs on [https://github.com/mgbellemare/Arcade-Learning-Environment/tree/master].
```
ROMs

In order to import ROMS, you need to download Roms.rar from the Atari 2600 VCS ROM Collection (http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)and extract the .rar file. Once you've done that, run:

python -m atari_py.import_roms <path to folder>

This should print out the names of ROMs as it imports them. The ROMs will be copied to your atari_py installation directory.

```
The ROMs can be found on [http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html]
