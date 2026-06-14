# Análise interpretativa (etapa 6)

Resumo automático dos casos de convergência e divergência entre agrupamentos
estruturais e semânticos.

## Contexto

- K primário (radicais): 3
- K primário (caracteres): 60

## Métricas globais (K primário)

- Radicais: ARI=-0.0007, NMI=0.0172
- Caracteres: ARI=0.0062, NMI=0.1130

Valores baixos de ARI/NMI indicam pouco alinhamento global entre as duas visões.
A análise abaixo destaca onde ainda há sobreposição local (convergência) e onde
estrutura e semântica se separam (divergência).

## Convergência (amostra)

- **radicals** 人 (struct=0, sem=0, overlap=64): 
- **radicals** 力 (struct=0, sem=0, overlap=64): 
- **radicals** 卜 (struct=0, sem=0, overlap=64): 
- **radicals** 厶 (struct=0, sem=0, overlap=64): 
- **radicals** 土 (struct=0, sem=0, overlap=64): 
- **radicals** 一 (struct=1, sem=2, overlap=1): 
- **radicals** 口 (struct=2, sem=1, overlap=1): 
- **radicals** 丨 (struct=0, sem=1, overlap=83): 

## Divergência (amostra)

- **radicals** 人 (struct_together_sem_split, struct=0, sem=0): struct_together_sem_split
- **radicals** 儿 (struct_together_sem_split, struct=0, sem=2): struct_together_sem_split
- **radicals** 几 (struct_together_sem_split, struct=0, sem=2): struct_together_sem_split
- **radicals** 口 (sem_together_struct_split, struct=2, sem=1): sem_together_struct_split
- **radicals** 一 (sem_together_struct_split, struct=1, sem=2): sem_together_struct_split
- **characters** 㒈 (struct_together_sem_split, struct=0, sem=18): struct_together_sem_split — dangerous; lofty; steep; high and dangerous
- **characters** 㔌 (struct_together_sem_split, struct=0, sem=52): struct_together_sem_split — to cut off; to mince, to cut up firewood
- **characters** 㖩 (struct_together_sem_split, struct=0, sem=20): struct_together_sem_split — not pure, immodest, to urge, (same as 嗾) to set a dog on
- **characters** 㐖 (struct_together_sem_split, struct=1, sem=32): struct_together_sem_split — 㐖毒, an old name for India
- **characters** 㐭 (struct_together_sem_split, struct=1, sem=48): struct_together_sem_split — (same as 廩) a granary, to supply (foodstuff), to stockpile
