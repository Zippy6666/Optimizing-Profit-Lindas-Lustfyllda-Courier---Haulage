import csv
import random
from pathlib import Path
from tqdm import tqdm


def seed_packages(n_iter: int = 100, target_path: Path = Path('lagerstatus.csv')) -> None:
    assert 0 < n_iter, 'n_iter needs to be a positive integer'
    assert n_iter < 9_000_000_000, 'n_iter needs to be less than 9 billion'

    entries = []
    id_num = random.randint(1_000_000_000, 9_999_999_999 - n_iter)
    for _ in tqdm( range(n_iter), total=n_iter, desc="Seeding..." ):
        id_num += 1
        weight = round((random.randint(10, 150) + random.randint(10, 80)) / 20, 1)
        profit = int((random.randint(1, 10) + random.randint(1, 10)) / 2)
        deadline = int((random.randint(-1, 7) + random.randint(-3, 3)) / 2)
        entries.append(
            {
                'Paket_id': id_num,
                'Vikt': weight,
                'Förtjänst': profit,
                'Deadline': deadline
            }
        )

    print("Writing CSV...")

    fieldnames = 'Paket_id','Vikt','Förtjänst','Deadline'
    with target_path.open('w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(entries)


if __name__ == '__main__':
    seed_packages()