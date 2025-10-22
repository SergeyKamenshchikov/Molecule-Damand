from molecule_demand_short import compute_molecule, PERPLEXITY_API_KEY
import asyncio
from dotenv import load_dotenv


if __name__ == "__main__":
    print(load_dotenv())
    query = "3D печать керамикой"
    print(PERPLEXITY_API_KEY)
    df = asyncio.run(compute_molecule(query, TEST=True, verbose=True))
    df.to_excel('data/resul.xlsx', index=False)