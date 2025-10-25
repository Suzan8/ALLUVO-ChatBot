from app.retriever import FaissRetriever
from app.llm import generate_answer

def format_context(results):
    lines = []
    for r in results:
        meta = r.get('meta', {})
        original = meta.get('original', {})
        name = original.get('name') or original.get('brand_name') or ''
        price = original.get('price', '')
        lines.append(f"id:{meta.get('source_id')} | name:{name} | price:{price} | snippet:{r.get('doc')}")
    return "\n".join(lines)

def rag_answer(query: str, top_k: int = 5, index_name: str = "products_index") -> str:
    retriever = FaissRetriever(index_name)
    hits = retriever.search(query, top_k=top_k)
    if not hits:
        return "لم أجد معلومات متاحة عن هذا الاستعلام في قاعدة بياناتنا. هل تريد أن أبحث في مجموعة أخرى؟"
    context = format_context(hits)
    answer = generate_answer(context, query)
    return answer
