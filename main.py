from LER import LER
from pathlib import Path


if __name__ == '__main__':
    ler = LER()
    datadir = Path('data')

    for imgpath in sorted(list(datadir.iterdir())):
        stats = ler.analyse(imgpath)
        print(imgpath.stem, stats[:,2].mean())
