# A25 corpus-basis rank sweep -- Qwen/Qwen2.5-0.5B, eval T~1800, seed 11

| doc | basis | recall | gold-NLL |
|---|---|---|---|
| prose:frankenstein | full KV | 14/14 | 0.105 |
| prose:frankenstein | shared r6144 | 14/14 | 0.111 |
| prose:frankenstein | shared r3072 | 14/14 | 0.125 |
| prose:frankenstein | shared r1536 | 12/14 | 0.250 |
| prose:frankenstein | per-doc r1536 | 14/14 | 0.100 |
| prose:frankenstein | shared r768 | 12/14 | 0.384 |
| prose:frankenstein | per-doc r768 | 14/14 | 0.093 |
| dialogue:earnest | full KV | 14/14 | 0.062 |
| dialogue:earnest | shared r6144 | 14/14 | 0.062 |
| dialogue:earnest | shared r3072 | 13/14 | 0.118 |
| dialogue:earnest | shared r1536 | 14/14 | 0.109 |
| dialogue:earnest | per-doc r1536 | 14/14 | 0.063 |
| dialogue:earnest | shared r768 | 12/14 | 0.285 |
| dialogue:earnest | per-doc r768 | 14/14 | 0.060 |
| technical:origin_species | full KV | 13/14 | 0.144 |
| technical:origin_species | shared r6144 | 13/14 | 0.148 |
| technical:origin_species | shared r3072 | 14/14 | 0.129 |
| technical:origin_species | shared r1536 | 12/14 | 0.229 |
| technical:origin_species | per-doc r1536 | 13/14 | 0.141 |
| technical:origin_species | shared r768 | 11/14 | 0.397 |
| technical:origin_species | per-doc r768 | 14/14 | 0.144 |
| code:code_eval | full KV | 14/14 | 0.088 |
| code:code_eval | shared r6144 | 13/14 | 0.091 |
| code:code_eval | shared r3072 | 13/14 | 0.148 |
| code:code_eval | shared r1536 | 13/14 | 0.185 |
| code:code_eval | per-doc r1536 | 14/14 | 0.090 |
| code:code_eval | shared r768 | 11/14 | 0.266 |
| code:code_eval | per-doc r768 | 14/14 | 0.089 |
| structured:logs | full KV | 14/14 | 0.073 |
| structured:logs | shared r6144 | 14/14 | 0.077 |
| structured:logs | shared r3072 | 14/14 | 0.116 |
| structured:logs | shared r1536 | 14/14 | 0.180 |
| structured:logs | per-doc r1536 | 14/14 | 0.075 |
| structured:logs | shared r768 | 10/14 | 0.283 |
| structured:logs | per-doc r768 | 14/14 | 0.074 |
