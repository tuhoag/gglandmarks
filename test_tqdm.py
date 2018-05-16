from time import sleep
from tqdm import trange, tqdm
from multiprocessing import Pool, freeze_support, RLock

L = list(range(9))


text = ""
for char in tqdm(["a", "b", "c", "d"]):
    sleep(1)
    text = text + char
