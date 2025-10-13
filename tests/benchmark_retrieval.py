#!/usr/bin/env python3
"""Retrieval accuracy benchmark for Cactus Nomic embeddings (FP16 vs INT8)

Usage:
    python benchmark_retrieval.py                    # Run full benchmark
    python benchmark_retrieval.py --samples 100     # Quick test
    python benchmark_retrieval.py --verify          # Verify setup only
"""

import argparse, ctypes, json, numpy as np, sys, time, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_100: float
    mrr: float
    ndcg_at_10: float
    
    def __str__(self):
        return f"R@1: {self.recall_at_1:.4f}, R@10: {self.recall_at_10:.4f}, MRR: {self.mrr:.4f}, NDCG@10: {self.ndcg_at_10:.4f}"


class CactusModel:
    def __init__(self, lib_path: str, model_path: str, context_size: int = 512):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.cactus_init.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        self.lib.cactus_init.restype = ctypes.c_void_p
        self.lib.cactus_embed.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cactus_embed.restype = ctypes.c_int
        self.lib.cactus_destroy.argtypes = [ctypes.c_void_p]
        self.lib.cactus_reset.argtypes = [ctypes.c_void_p]
        
        self.model = self.lib.cactus_init(model_path.encode('utf-8'), context_size)
        if not self.model:
            raise RuntimeError(f"Failed to initialize model from {model_path}")
    
    def embed(self, text: str) -> np.ndarray:
        max_dim = 8192
        buffer = (ctypes.c_float * max_dim)()
        dim = ctypes.c_size_t()
        result = self.lib.cactus_embed(self.model, text.encode('utf-8'), buffer, max_dim * 4, ctypes.byref(dim))
        if result < 0:
            raise RuntimeError(f"Embedding failed: {result}")
        embedding = np.array(buffer[:dim.value], dtype=np.float32)
        self.lib.cactus_reset(self.model)
        return embedding
    
    def __del__(self):
        if hasattr(self, 'model') and self.model:
            self.lib.cactus_destroy(self.model)


def build_shared_library(project_root: Path) -> Path:
    dylib = project_root / "cactus/build/libcactus.dylib"
    so = project_root / "cactus/build/libcactus.so"
    if dylib.exists(): return dylib
    if so.exists(): return so
    
    static = project_root / "cactus/build/libcactus.a"
    if not static.exists():
        subprocess.run([str(project_root / "cactus/build.sh")], check=True)
    
    print("Building shared library...")
    if sys.platform == "darwin":
        subprocess.run(["c++", "-dynamiclib", "-o", str(dylib), "-Wl,-all_load", str(static), "-framework", "Accelerate"], check=True)
        return dylib
    else:
        subprocess.run(["c++", "-shared", "-o", str(so), "-Wl,--whole-archive", str(static), "-Wl,--no-whole-archive"], check=True)
        return so


def compute_similarity_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    docs_norm = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    return np.dot(queries_norm, docs_norm.T)

def compute_dcg(relevances: List[int], k: int) -> float:
    relevances = relevances[:k]
    if not relevances: return 0.0
    dcg = relevances[0]
    for i in range(1, len(relevances)):
        dcg += relevances[i] / np.log2(i + 2)
    return dcg

def compute_ndcg(relevances: List[int], k: int) -> float:
    dcg = compute_dcg(relevances, k)
    idcg = compute_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(similarity_matrix: np.ndarray, ground_truth: List[List[int]], k_values=[1,5,10,100]) -> RetrievalMetrics:
    recalls = {k: [] for k in k_values}
    reciprocal_ranks, ndcg_scores = [], []
    
    for i in range(similarity_matrix.shape[0]):
        ranked_docs = np.argsort(-similarity_matrix[i])
        relevant_docs = set(ground_truth[i])
        if not relevant_docs: continue
        
        for k in k_values:
            recalls[k].append(len(set(ranked_docs[:k]) & relevant_docs) / len(relevant_docs))
        
        for rank, doc_id in enumerate(ranked_docs, 1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
        
        relevances = [1 if doc_id in relevant_docs else 0 for doc_id in ranked_docs[:10]]
        ndcg_scores.append(compute_ndcg(relevances, 10))
    
    return RetrievalMetrics(
        recall_at_1=np.mean(recalls[1]) if 1 in recalls and recalls[1] else 0.0,
        recall_at_5=np.mean(recalls[5]) if 5 in recalls and recalls[5] else 0.0,
        recall_at_10=np.mean(recalls[10]) if 10 in recalls and recalls[10] else 0.0,
        recall_at_100=np.mean(recalls[100]) if 100 in recalls and recalls[100] else 0.0,
        mrr=np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        ndcg_at_10=np.mean(ndcg_scores) if ndcg_scores else 0.0
    )


def load_retrieval_dataset(max_samples: int = 1000) -> Tuple[List[str], List[str], List[List[int]]]:
    if HAS_DATASETS:
        try:
            print("Loading MS MARCO dataset...")
            dataset = load_dataset("ms_marco", "v1.1", split="validation", streaming=True)
            queries, documents, ground_truth, doc_map, doc_id = [], [], [], {}, 0
            
            for i, ex in enumerate(dataset):
                if i >= max_samples: break
                query, passages = ex.get('query', ''), ex.get('passages', {})
                if not query or not passages: continue
                
                queries.append(query)
                relevant_ids = []
                for text, selected in zip(passages.get('passage_text', []), passages.get('is_selected', [])):
                    if text not in doc_map:
                        doc_map[text], doc_id = doc_id, doc_id + 1
                        documents.append(text)
                    if selected:
                        relevant_ids.append(doc_map[text])
                ground_truth.append(relevant_ids)
            
            print(f"Loaded {len(queries)} queries, {len(documents)} documents")
            return queries, documents, ground_truth
        except Exception as e:
            print(f"MS MARCO failed: {e}. Using synthetic data...")
    
    print("Creating synthetic dataset...")
    np.random.seed(42)
    topics = ["AI/ML", "climate", "quantum", "medical", "space", "energy", "security", "biotech", "robotics", "neuro"]
    documents = [f"Doc about {t}: aspect {i}" for t in topics for i in range(100)]
    queries = [f"Research on {topics[i%len(topics)]}" for i in range(min(max_samples, 1000))]
    ground_truth = [list(range((i%len(topics))*100, ((i%len(topics))+1)*100)) for i in range(len(queries))]
    print(f"Created {len(queries)} queries, {len(documents)} documents")
    return queries, documents, ground_truth


def run_benchmark(model: CactusModel, queries: List[str], documents: List[str], ground_truth: List[List[int]]) -> Tuple[RetrievalMetrics, float, float]:
    print(f"Embedding {len(queries)} queries...")
    t0 = time.time()
    query_embs = [model.embed(q) for q in (tqdm(queries) if HAS_TQDM else queries)]
    query_time = time.time() - t0
    
    print(f"Embedding {len(documents)} documents...")
    t0 = time.time()
    doc_embs = [model.embed(d) for d in (tqdm(documents) if HAS_TQDM else documents)]
    doc_time = time.time() - t0
    
    print("Computing metrics...")
    sim_matrix = compute_similarity_matrix(np.array(query_embs), np.array(doc_embs))
    metrics = evaluate_retrieval(sim_matrix, ground_truth)
    return metrics, query_time, doc_time


def verify_setup(lib_path: Path, fp16_path: Path, int8_path: Path) -> bool:
    """Verify benchmark setup"""
    print("="*60)
    print("Verifying Setup")
    print("="*60)
    
    ok = True
    if not lib_path.exists():
        print(f"✗ Library not found: {lib_path}")
        ok = False
    else:
        print(f"✓ Library: {lib_path}")
    
    if not fp16_path.exists():
        print(f"✗ FP16 model not found: {fp16_path}")
        ok = False
    else:
        print(f"✓ FP16 model: {fp16_path}")
    
    if not int8_path.exists():
        print(f"✗ INT8 model not found: {int8_path}")
        ok = False
    else:
        print(f"✓ INT8 model: {int8_path}")
    
    if ok:
        try:
            model = CactusModel(str(lib_path), str(int8_path))
            emb = model.embed("test")
            print(f"✓ Embedding generation works (dim={len(emb)})")
        except Exception as e:
            print(f"✗ Embedding test failed: {e}")
            ok = False
    
    print("="*60)
    return ok

def main():
    parser = argparse.ArgumentParser(description="Benchmark Nomic embeddings: FP16 vs INT8")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parent.parent)
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--lib-path", type=Path, help="Path to libcactus")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Verify setup only")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    lib_path = args.lib_path or build_shared_library(project_root)
    fp16_path = project_root / "weights/nomic-embed-text-v2-moe-fp16"
    int8_path = project_root / "weights/nomic-embed-text-v2-moe"
    
    if args.verify:
        sys.exit(0 if verify_setup(lib_path, fp16_path, int8_path) else 1)
    
    if not lib_path.exists() or (not fp16_path.exists() and not int8_path.exists()):
        print("Setup incomplete. Run with --verify to check")
        sys.exit(1)
    
    queries, documents, ground_truth = load_retrieval_dataset(args.samples)
    results = {}
    
    for precision, path in [('fp16', fp16_path), ('int8', int8_path)]:
        if (precision == 'fp16' and args.skip_fp16) or (precision == 'int8' and args.skip_int8):
            continue
        if not path.exists():
            print(f"Skipping {precision}: model not found")
            continue
        
        print(f"\n{'='*60}\nBENCHMARKING {precision.upper()}\n{'='*60}")
        try:
            model = CactusModel(str(lib_path), str(path))
            metrics, qt, dt = run_benchmark(model, queries, documents, ground_truth)
            results[precision] = {'metrics': metrics, 'query_time': qt, 'doc_time': dt}
            print(f"\n{metrics}")
            print(f"Throughput: {len(queries)/qt:.1f} q/s, {len(documents)/dt:.1f} d/s")
        except Exception as e:
            print(f"Error: {e}")
    
    if 'fp16' in results and 'int8' in results:
        print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
        fp16, int8 = results['fp16']['metrics'], results['int8']['metrics']
        for name, attr in [('R@10', 'recall_at_10'), ('MRR', 'mrr'), ('NDCG@10', 'ndcg_at_10')]:
            f, i = getattr(fp16, attr), getattr(int8, attr)
            print(f"{name:<10} FP16:{f:.4f} INT8:{i:.4f} Diff:{i-f:+.4f}")
        
        qps_f, qps_i = len(queries)/results['fp16']['query_time'], len(queries)/results['int8']['query_time']
        print(f"\nSpeedup: {qps_i/qps_f:.2f}x ({qps_f:.1f} -> {qps_i:.1f} q/s)")
    
    with open(project_root / "tests/benchmark_results.json", 'w') as f:
        json.dump({
            'samples': len(queries),
            'results': {p: {
                'metrics': {k: float(v) for k, v in d['metrics'].__dict__.items()},
                'query_time': d['query_time'],
                'doc_time': d['doc_time']
            } for p, d in results.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main()

