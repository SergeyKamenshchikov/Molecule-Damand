from molecule_demand_short import compute_molecule
import asyncio
from dotenv import load_dotenv
from pathlib import Path


if __name__ == "__main__":
    print(load_dotenv())
    query = "3D печать керамикой"
    df = asyncio.run(compute_molecule(query, TEST=True, verbose=True))
    result_path = Path('result_data')
    result_path.mkdir(exist_ok=True)
    df.to_excel(result_path/'result.xlsx', index=False)