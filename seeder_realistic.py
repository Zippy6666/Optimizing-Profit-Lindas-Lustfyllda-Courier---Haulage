import csv
import random
from pathlib import Path
from tqdm import tqdm


def seed_packages(n_iter: int = 100, target_path: Path = Path('lagerstatus.csv')) -> None:
    assert 0 < n_iter, 'n_iter needs to be a positive integer'
    assert n_iter < 9_000_000_000, 'n_iter needs to be less than 9 billion'

    entries = []
    id_num = random.randint(1_000_000_000, 9_999_999_999 - n_iter)
    for _ in tqdm( range(n_iter), total=n_iter, desc="Seeding...", leave=False ):
        id_num += 1

        if random.random() > 0.55:
            # Heavy packages are a bit more rare
            weight = round( random.randint(5, 11) + random.random(),  1)
        else:
            weight = round( random.randint(1, 4) + random.random(), 1)


        if random.random() > 0.84:
            profit = random.randint(3, 6)
        else:
            profit = random.randint(1, 9)

        if random.random() > 0.13:
            deadline = random.randint(0, 9)
        else:
            deadline = random.randint(-5, -1)


        entries.append(
            {
                'Paket_id': id_num,
                'Vikt': weight,
                'Förtjänst': profit,
                'Deadline': deadline
            }
        )

    fieldnames = 'Paket_id','Vikt','Förtjänst','Deadline'
    with target_path.open('w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(entries)

if __name__ == '__main__':
    seed_packages()