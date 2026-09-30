# Semantic Search for Fashion Products

Search a catalogue of ~12,700 fashion products by meaning, not exact keywords. Type "black track pants" or "summer floral dress" and get the closest matching products, with images, from a vector database.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nimishac28/Semantic_search/blob/main/Semantic_search_fashionproduct.ipynb)

<!-- Add a screenshot or GIF of the Streamlit app here -->
<!-- ![Demo](assets/demo.png) -->

## Why

Keyword search fails when the shopper's words don't match the catalogue's words. Semantic search converts both the query and every product description into embeddings (vectors), then returns the products whose vectors are closest to the query. "Joggers" can find "track pants" even though the words never overlap.

## How it works

```
Styles.csv ──► Clean data ──► Encode descriptions ──► Store in ChromaDB
                              (all-MiniLM-L6-v2,        (persistent,
                               384-dim vectors)          ./fashion_db)
                                                              │
User query ──► Encode query ──► Nearest-neighbour search ◄────┘
                                        │
                                        ▼
                         Streamlit UI: top 5 products + images
```

1. **Load and clean** – read `Styles.csv`, drop the empty `Unnamed: 4` column, switch image links from `http` to `https` so they render in the browser.
2. **Embed** – encode all 12,758 product descriptions with `sentence-transformers/all-MiniLM-L6-v2` → a `(12758, 384)` matrix.
3. **Index** – store documents, embeddings, and row IDs in a persistent ChromaDB collection, added in batches of 5,000.
4. **Query** – encode the search text, retrieve the 5 nearest products.
5. **Serve** – a Streamlit app shows each result's image, description, and category. In Colab, the app is exposed publicly through an ngrok tunnel.

## Tech stack

| Layer | Tool |
|---|---|
| Data handling | pandas |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent client) |
| UI | Streamlit |
| Tunnel (Colab only) | pyngrok |

## Dataset

`Styles.csv` – 12,758 products with these columns:

| Column | Example |
|---|---|
| `Product_ID` | 15970 |
| `Gender` | Men / Women / Boys / Girls / Unisex |
| `Category` | Topwear, Bottomwear, Watches … (42 categories) |
| `Product Description` | Turtle Check Men Navy Blue Shirt |
| `link` | Product image URL |

<!-- TODO: add the dataset source and licence (e.g. Kaggle link) -->

## Example results

**Query:** `black track pants`
```
ADIDAS Men Black Track Pants
Nike Men Black Track Pants
Nike Men Black Track Pants
Proline Men Black Track Pants
ADIDAS Men Cress 3spespt Black Track Pants
```

**Query:** `summer floral dress`
```
Mineral Blue Dress
Forever New Women Blossom Silk Cream Dress
Remanika Women Purple Dress
Forever New Women's Cream Floral Print Top
Forever New Women Pink Sleeveless Dress
```

The first query is a clean hit. The second shows the model's limits – see below.

## Getting started

### Run in Google Colab (easiest)

1. Open the notebook with the badge above.
2. Upload `Styles.csv` to the Colab session.
3. Run all cells up to the ChromaDB indexing step.
4. Add your own ngrok auth token (get one free at [ngrok.com](https://ngrok.com)):
   ```python
   from google.colab import userdata
   !ngrok config add-authtoken {userdata.get('NGROK_TOKEN')}
   ```
   Store the token in Colab's **Secrets** panel – never paste it directly into the notebook.
5. Run the Streamlit + ngrok cells and open the printed public URL.

### Run locally

```bash
git clone https://github.com/nimishac28/Semantic_search.git
cd Semantic_search
pip install pandas sentence-transformers chromadb streamlit
```

Build the vector index once by running the notebook cells up to the indexing step (this creates `./fashion_db`), then:

```bash
streamlit run app.py
```

Open http://localhost:8501. No ngrok needed locally.

## Project structure

```
├── Semantic_search_fashionproduct.ipynb   # data prep, embedding, indexing, testing
├── app.py                                 # Streamlit search UI
├── Styles.csv                             # product data
└── fashion_db/                            # ChromaDB store (generated, not committed)
```

## Limitations

- **Short descriptions limit what the model can match.** Product titles rarely mention season, occasion, fabric, or fit, so "summer" has little to match against. The floral query returned plain dresses and a floral *top* because that information simply isn't in the text.
- **Text only.** Search ignores the product images, so visual attributes (print, cut, colour shade) only count if they appear in the title.
- **No filters.** Gender and category are in the data but not used in search, so a "women's" query can still return men's products.
- **No evaluation yet.** Results have been checked by eye, not measured with a metric like precision@k or recall@k.
- **Query embedding path differs from indexing.** The notebook indexes with `sentence-transformers`, while `app.py` uses `query_texts`, which relies on ChromaDB's default embedding function. Both use MiniLM-L6-v2 today, but passing the same model explicitly would be safer.

## Next steps

- [ ] Add gender/category filters using ChromaDB metadata
- [ ] Combine product description + category + gender into richer text before embedding
- [ ] Try multimodal embeddings (e.g. CLIP) so images contribute to search
- [ ] Hybrid search: BM25 keyword scores + vector similarity
- [ ] Build a small labelled query set and report precision@5
- [ ] Deploy on Streamlit Community Cloud instead of an ngrok tunnel

## Author

**Nimisha** – [GitHub](https://github.com/nimishac28) <!-- add LinkedIn -->
