from main import df, delivery_vans


if __name__ == "__main__":
    df_no_penelty = df[df['Deadline'] >= 0].copy()

    # while df_no_penelty["Vikt"].sum() > 800*len(delivery_vans):

    #     best_package_goodness = float("-inf")

    #     for row in df_no_penelty.iterrows():
    #         package_goodness = row["Förtjänst"] - row["Vikt"]
    #         if package_goodness > best_package_goodness:
    #             best_package_goodness = package_goodness

