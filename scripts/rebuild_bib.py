# 병합(merge=union)으로 섞인 data/references.bib를 git 기록에서 다시 만든다.
# 각 키는 그 키가 처음 나타난(덧붙인 에이전트의) 커밋의 온전한 항목을 쓴다.
# 사용: python3 scripts/rebuild_bib.py data/references.bib
import re, subprocess, sys
def entries(text):
    out=[]
    for chunk in re.split(r'\n(?=@\w+\{)', text):
        m=re.match(r'@\w+\{([^,\s]+),', chunk)
        if m: out.append((m.group(1), chunk.rstrip()))
    return out
def valid(e):
    body=e
    if body.count('{')!=body.count('}'): return False
    if re.search(r'\n@', body[1:]): return False
    fields=re.findall(r'^\s*(\w+)\s*=', body, re.M)
    if len(fields)!=len(set(fields)): return False
    lines=[l for l in body.split('\n')[1:-1] if l.strip()]
    for a,b in zip(lines, lines[1:]):
        if re.match(r'^\s*\w+\s*=', b) and not a.rstrip().endswith(','): return False
    return True
commits=subprocess.run(['git','log','--no-merges','--reverse','--format=%H','--','data/references.bib'],capture_output=True,text=True).stdout.split()
best={}; order=[]
for c in commits:
    t=subprocess.run(['git','show',f'{c}:data/references.bib'],capture_output=True,text=True).stdout
    for k,e in entries(t):
        if not valid(e): continue
        if k not in best: order.append(k)
        best.setdefault(k,e)
cur=open('data/references.bib').read()
curkeys=[k for k,_ in entries(cur)]
header=cur.split('@',1)[0]
missing=[k for k in curkeys if k not in best]
print('valid keys',len(best),'current keys',len(set(curkeys)),'missing from history',missing)
open(sys.argv[1],'w').write(header.rstrip()+'\n\n'+'\n\n'.join(best[k] for k in order)+'\n')
