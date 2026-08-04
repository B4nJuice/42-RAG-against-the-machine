import json
from pathlib import Path
from chunking.chunker_selector import ChunkerSelector
from indexing.bm25 import BM25


def normalize_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(Path.cwd())
        except ValueError:
            pass

    return p.as_posix().lstrip("./")

def evaluate_bm25_on_dataset():
    """Évalue le BM25 sur toutes les questions du dataset et affiche les statistiques."""
    
    # Charger les données
    selector = ChunkerSelector()
    chunks = selector.folder_chunking("./data/raw/vllm-0.10.1")
    
    # Initialiser BM25
    bm25 = BM25(chunks, b=0.6, k=1.2, l=0.9)
    bm25.compute_stats()
    print(bm25.n_chunks)
    
    # Charger le dataset de questions
    with open("./data/public/AnsweredQuestions/dataset_code_public.json", "r") as f:
        dataset = json.load(f)
    
    questions = dataset["rag_questions"]
    
    rankings = []  # Liste des positions du chunk correct
    successes = 0  # Nombre de fois où le chunk correct est trouvé
    
    for idx, q in enumerate(questions):
        question = q["question"]
        expected_file_paths = {
            normalize_path(source["file_path"])
            for source in q.get("sources", [])
            if source.get("file_path")
        }
        
        # Obtenir les chunks classés par score
        scores = bm25.get_best_chunk(question)
        ranked_chunks = sorted(scores, key=lambda x: -scores[x])
        
        # Trouver la position du chunk attendu
        position = None
        for rank, chunk in enumerate(ranked_chunks):
            chunk_path = normalize_path(chunk.file_path)
            if chunk_path in expected_file_paths:
                position = rank + 1  # Position 1-indexed
                break
        
        if position:
            rankings.append(position)
            if position == 1:
                successes += 1
            print(f"Q{idx+1}: '{question[:60]}...' -> Position {position}")
        else:
            top_path = normalize_path(ranked_chunks[0].file_path) if ranked_chunks else "N/A"
            expected_preview = next(iter(expected_file_paths), "N/A")
            print(
                f"Q{idx+1}: '{question[:60]}...' -> CHUNK NOT FOUND "
                f"(attendu: {expected_preview}, top-1: {top_path})"
            )
    
    # Calculer les statistiques
    if rankings:
        avg_ranking = sum(rankings) / len(rankings)
        success_rate = (successes / len(questions)) * 100
        top3_successes = sum(1 for r in rankings if r <= 3)
        top5_successes = sum(1 for r in rankings if r <= 5)
        top10_successes = sum(1 for r in rankings if r <= 10)
        top3_rate = (top3_successes / len(questions)) * 100
        top5_rate = (top5_successes / len(questions)) * 100
        top10_rate = (top10_successes / len(questions)) * 100
        
        print("\n" + "="*60)
        print(f"RÉSULTATS:")
        print(f"  Nombre de questions: {len(questions)}")
        print(f"  Classement moyen: {avg_ranking:.2f}")
        print(f"  Taux de réussite (top-1): {success_rate:.2f}%")
        print(f"  Succès (top-1): {successes}/{len(questions)}")
        print(f"  Taux de réussite (top-3): {top3_rate:.2f}%")
        print(f"  Succès (top-3): {top3_successes}/{len(questions)}")
        print(f"  Taux de réussite (top-5): {top5_rate:.2f}%")
        print(f"  Succès (top-5): {top5_successes}/{len(questions)}")
        print(f"  Taux de réussite (top-10): {top10_rate:.2f}%")
        print(f"  Succès (top-10): {top10_successes}/{len(questions)}")
        print("="*60)
    else:
        print("Aucun résultat!")

if __name__ == "__main__":
    evaluate_bm25_on_dataset()
    
    # Ou mode interactif si on préfère
    # selector = ChunkerSelector()
    # chunks = selector.folder_chunking("./data/raw/vllm-0.10.1")
    # bm25 = BM25(chunks, b=0.6, k=1.2, l=0.9)
    # bm25.compute_stats()
    # c = bm25.get_best_chunk(input())
    # c2 = sorted(c, key=lambda x: -c[x])
    # for i in range(15):
    #     print(i, c2[i].file_path, c[c2[i]])