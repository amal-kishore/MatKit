from matkit.data import fetch_data

if __name__ == "__main__":
    df = fetch_data(stoichiometry='ABC', sources=['materials_project'], save=True, filename="abc_raw_data.csv")
    print(f"Fetched {len(df)} entries with ABC stoichiometry.")
