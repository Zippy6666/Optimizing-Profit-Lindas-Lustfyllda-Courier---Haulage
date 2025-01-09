import numpy as np
import pandas as pd
import main


def make_gene( result: dict ) -> np.ndarray:
    """
    ### Args:
    `result`: A result dictionary from main.package_vans

    ### Returns
    `gene`: A gene in the form of a array
    """
    return result["df"]["Delivered"].to_numpy()


def fill_vans_according_to_gene( vans: list[main.DeliveryVan], df: pd.DataFrame, gene: np.ndarray ) -> np.float64:
    main.clear_vans(vans)
    
    df["Delivered"] = gene

    df_np = df.to_numpy()

    weight_idx = df.columns.get_loc("Vikt")  
    profit_idx = df.columns.get_loc("Förtjänst")  
    deadline_idx = df.columns.get_loc("Deadline")
    delivered_idx = df.columns.get_loc("Delivered")  

    for van_idx, van in enumerate( vans ):
        for package in df_np:
            if package[delivered_idx] == van_idx:
                van.load_package(package[weight_idx], main.get_real_profit(package[profit_idx], package[deadline_idx]))

    return sum(van.get_profit() for van in vans)


def mutate(gene: np.ndarray, rate: np.float64 = np.float64(0.01)) -> np.ndarray:
    """
    ### Args:
    `rate`: The rate of mutation
    `gene`: The gene to mutate

    ### Returns
    `mutated_gene` The mutated gene
    """
    mutated_gene = gene.copy()
    maxrand = len(gene)-1
    for i in range(len(gene)):
        if np.random.rand() < rate:
            mutated_gene[i] = np.random.randint(-1, maxrand)
    return mutated_gene


def evolve( df: pd.DataFrame, vans: list[main.DeliveryVan], gene: np.ndarray ) -> np.ndarray:
    """
    ### Args:
    `df`: The dataframe containing the packages
    `vans`: The vans to fill
    """
    best_profit = 0
    unmutated_profit = fill_vans_according_to_gene(vans, df, gene)

    while True:
        mutated_gene = mutate(gene)
        
        profit = fill_vans_according_to_gene(vans, df, mutated_gene)

        if profit > best_profit:
            print("new better mutated gene")
            gene = mutated_gene
            best_profit = profit

        # print(vans)
        # print("best mutated profit: ", best_profit)
        # print("unmutated profit: ", unmutated_profit)


if __name__ == "__main__":
    packages = pd.read_csv("lagerstatus.csv")

    vans = [ main.DeliveryVan( "bil_"+str(i) ) for i in range(1, 11) ]

    result = main.package_vans(len(packages), packages)

    gene = make_gene(result)

    evolve(result["df"], vans, gene)
