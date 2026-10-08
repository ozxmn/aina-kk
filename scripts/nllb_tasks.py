import json, random
ru=[json.loads(l) for l in open('data/human/ru_pool.jsonl',encoding='utf-8')]
random.Random(31).shuffle(ru)
used=set()
for fn in ['data/machine/gemini__gemini-3.1-flash-lite.tasks.jsonl','data/machine/groq__qwen_qwen3.8-27b.tasks.jsonl']:
    for l in open(fn,encoding='utf-8'):
        t=json.loads(l)
        if t['condition']=='C4': used.add(t['source_id'])
free=[r for r in ru if r['id'] not in used]
with open('kaggle/gen_inputs/nllb.tasks.jsonl','w',encoding='utf-8') as f:
    for r in free[:200]:
        f.write(json.dumps({"task_id":f"nllb|C5|{r['id']}","generator":"hf:facebook/nllb-200-distilled-600M","condition":"C5","source_id":r['id'],
                            "group":r['group'],"domain":"web-ru","split":r['split'],"ru_source":r['text']},ensure_ascii=False)+"\n")
print('LLM-translated sources', len(used), 'NLLB tasks', min(200,len(free)))
